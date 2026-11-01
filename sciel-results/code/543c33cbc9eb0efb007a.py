"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import product

def construct(k, g):
    deadline = time.time() + 145  # Leave 5s safety margin
    
    # For k=5, g=9: use voltage graph over K6
    if k == 5 and g == 9:
        result = _voltage_graph_k5_g9(deadline)
        if result is not None:
            return result
        # Fallback to LUW
        return _luw_construct(k, g)
    
    # Fallback to LUW for other parameters
    return _luv_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# Alias to avoid typo
_luv_construct = _luw_construct

def _voltage_graph_k5_g9(deadline):
    """Construct a 5-regular graph of girth >= 9 using voltage graph over K6."""
    
    # K6 has 6 vertices, 15 edges
    base_vertices = 6
    base_edges = []
    edge_to_idx = {}
    for i in range(6):
        for j in range(i + 1, 6):
            edge_to_idx[(i, j)] = len(base_edges)
            base_edges.append((i, j))
    
    # Enumerate all non-backtracking closed walks in K6 of length 3 to 8
    # A non-backtracking closed walk is v0, v1, ..., vL=v0 where:
    # - v_i != v_{i+1} (consecutive vertices differ)
    # - v_{i+1} != v_{i-1} (no immediate backtracking)
    # For the voltage sum: for each step v_i -> v_{i+1}, if v_i < v_{i+1} add phi, else subtract phi
    
    # To avoid duplicates, we fix v0 = 0 (by symmetry) and v1 < v2 (to avoid reversal duplicates)
    # Actually, let's just enumerate all and use the voltage constraint
    
    # For efficiency, let's precompute the voltage sum expression for each walk
    # Each walk gives a linear combination of the 15 edge voltages
    # We need this sum != 0 mod m for all walks of length 3..8
    
    # Enumerate walks
    def enumerate_walks(max_len):
        """Enumerate non-backtracking closed walks starting and ending at vertex 0."""
        walks = []
        # Walk: v0=0, v1, v2, ..., vL=0
        # v1 can be any of 1,2,3,4,5
        # For each step, next vertex != current and != previous
        
        for v1 in range(1, 6):
            # Build walk step by step
            def extend(path, prev, current, length):
                if length == 0:
                    # We need to return to 0
                    if current != 0:
                        # Check if 0 is a valid next step
                        if 0 != prev:
                            walks.append(path + [0])
                    return
                
                for next_v in range(6):
                    if next_v == current or next_v == prev:
                        continue
                    if length == 1:
                        # This is the last step before returning to 0
                        if next_v != 0:
                            continue
                        else:
                            # Check that 0 != prev (no backtracking)
                            if 0 != prev:
                                walks.append(path + [0])
                    else:
                        extend(path + [next_v], current, next_v, length - 1)
            
            # Start: path=[0, v1], prev=0, current=v1, need L-2 more steps to reach 0
            for L in range(3, max_len + 1):
                # We need L-1 steps from v1 to 0
                extend([0, v1], 0, v1, L - 2)
        
        return walks
    
    # Actually, let me use a cleaner approach
    def get_walks(max_len):
        """Get all non-backtracking closed walks of length 3..max_len in K6, starting at 0."""
        walks = []
        
        for L in range(3, max_len + 1):
            # v0 = 0, v1 in 1..5
            for v1 in range(1, 6):
                # Recursively build the walk
                def build(path, prev, cur, remaining):
                    if remaining == 0:
                        # Must return to 0
                        if cur != 0 and 0 != prev:
                            walks.append(path + [0])
                        return
                    
                    for nxt in range(6):
                        if nxt == cur or nxt == prev:
                            continue
                        if remaining == 1:
                            # Last step: must go to 0
                            if nxt == 0 and 0 != prev:
                                walks.append(path + [0])
                        else:
                            build(path + [nxt], cur, nxt, remaining - 1)
                
                build([0, v1], 0, v1, L - 2)
        
        return walks
    
    # For each walk, compute the voltage sum as a vector of coefficients
    # coeff[e] = +1 if we traverse edge e in forward direction, -1 if backward, 0 otherwise
    
    def walk_voltage_coeffs(walk):
        """Compute voltage sum coefficients for a walk."""
        coeffs = [0] * 15
        for i in range(len(walk) - 1):
            u, v = walk[i], walk[i + 1]
            if u < v:
                idx = edge_to_idx[(u, v)]
                coeffs[idx] += 1
            else:
                idx = edge_to_idx[(v, u)]
                coeffs[idx] -= 1
        return coeffs
    
    # Precompute all walk coefficient vectors
    all_walks = get_walks(8)
    walk_coeffs = [walk_voltage_coeffs(w) for w in all_walks]
    
    # Remove duplicate coefficient vectors (same constraint)
    unique_coeffs = []
    seen = set()
    for c in walk_coeffs:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_coeffs.append(c)
    
    # Now search for voltage assignments
    # For each m, try to find voltages in Z_m such that for all walks, sum != 0 mod m
    
    random.seed(42)
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (5,9): n >= 2*(1+5+25+125) = 362
    # So m >= 61 (since n = 6m)
    
    for m in range(61, 200):
        if time.time() > deadline:
            break
        
        # Try random voltage assignments
        for trial in range(100):
            if time.time() > deadline:
                break
            
            voltages = [random.randint(0, m - 1) for _ in range(15)]
            
            # Check all walks
            valid = True
            for coeffs in unique_coeffs:
                total = sum(c * v for c, v in zip(coeffs, voltages)) % m
                if total == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Build the graph
            n = 6 * m
            edges = []
            adj = [[] for _ in range(n)]
            
            for (u, v), vol in zip(base_edges, voltages):
                for x in range(m):
                    u_lift = u * m + x
                    v_lift = v * m + ((x + vol) % m)
                    if u_lift < v_lift:
                        edges.append((u_lift, v_lift))
                    adj[u_lift].append(v_lift)
                    adj[v_lift].append(u_lift)
            
            # Verify regularity
            is_regular = all(len(adj[i]) == 5 for i in range(n))
            if not is_regular:
                continue
            
            # Verify girth using non-backtracking BFS
            if _check_girth(adj, n, 9):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    if n < 500:
                        return n, edges
                    break  # Found valid for this m, try smaller m
    
    if best_edges is not None:
        return best_n, best_edges
    
    return None

def _check_girth(adj, n, target_girth):
    """Check if graph has girth >= target_girth using non-backtracking BFS."""
    max_depth = (target_girth - 1) // 2
    
    for start in range(n):
        # BFS with parent tracking
        visited = [[-1] * n for _ in range(max_depth + 1)]
        visited[0][start] = -2
        
        queue = [(start, 0, -1)]
        
        while queue:
            v, depth, parent = queue.pop(0)
            if depth >= max_depth:
                continue
            
            for u in adj[v]:
                if u == parent:
                    continue
                if visited[depth + 1][u] != -1:
                    return False
                visited[depth + 1][u] = v
                queue.append((u, depth + 1, v))
    
    return True
# EVOLVE-BLOCK-END
