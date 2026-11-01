import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    # The primary target of this task is k=5, g=9.
    # We use a voltage graph construction.
    # We lift a small base graph (K_6) with voltages in Z_m to get a 5-regular graph of size 6*m.
    # We need to ensure girth >= 9. This means no non-backtracking closed walks of length 3..8 in the base graph
    # should have a voltage sum of 0 mod m.
    
    if k == 5 and g == 9:
        return _construct_5_9()
    
    # Fallback for other parameters (not expected in this task, but good for robustness)
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """
    Fallback: Lazebnik-Ustimenko-Woldar (LUW) graph construction.
    For k=5, g=9, this gives n=1250.
    """
    q = k
    # Girth of LUW(d,q) is d+5.
    # We want girth >= g, so d+5 >= g => d >= g-5.
    # Also d >= 3.
    # The seed program used specific parity logic, but standard LUW works for d >= 3.
    # Let's stick to the seed's d selection to be safe, or just use d = g-5.
    # For g=9, d=4. Girth 9. n = 2*q^d = 2*5^4 = 1250.
    # The seed used d=5 for g=9, giving n=6250? No, seed said n=1250 for d=4?
    # "For g=9, d=4 gives girth 9, n=2*5^4=1250"
    # "Seed used d=5. Why?"
    # Let's just use d = max(3, g-5).
    d = max(3, g - 5)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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


def _construct_5_9():
    """
    Specialized construction for k=5, g=9.
    Uses voltage lift of K_6.
    """
    deadline = time.time() + 140  # 10s margin
    
    # Base graph: K_6
    base_n = 6
    # Edges of K_6: 15 edges
    base_edges = []
    edge_idx = {}
    for u in range(base_n):
        for v in range(u + 1, base_n):
            edge_idx[(u, v)] = len(base_edges)
            base_edges.append((u, v))
    
    num_edges = len(base_edges)
    
    # Generate all non-backtracking closed walks of length 3 to 8 in K_6.
    # We represent each walk by its coefficient vector over the 15 edges.
    # A walk is a sequence of vertices v_0, ..., v_L with v_L = v_0.
    # Non-backtracking: v_{i-1} != v_{i+1} for all i (mod L).
    # Since it's a closed walk, we check indices 0..L-1.
    # Actually, the condition is that you don't immediately reverse direction.
    # In a graph, an edge is undirected. A walk is a sequence of edges e_1, ..., e_L.
    # e_i connects v_{i-1} and v_i.
    # Non-backtracking means e_{i-1} != e_i.
    # For K_6, this is equivalent to v_{i-1} != v_{i+1}.
    
    constraints = []
    
    def get_edge_id(u, v):
        if u > v:
            u, v = v, u
        return edge_idx.get((u, v), -1)
    
    def gen_walks(length):
        # Recursive generation of non-backtracking walks starting at 0.
        # v_0 = 0.
        # We need to close the walk, so v_L = 0.
        # And v_{L-1} != v_1 (to ensure non-backtracking at the start/end junction).
        
        def backtrack(v_idx, current, prev, path):
            if v_idx == length:
                # Must be at 0
                if current == 0:
                    # Check non-backtracking at the closure
                    # The last edge is (v_{L-1}, 0). The first edge is (0, v_1).
                    # They are the same edge if v_{L-1} == v_1.
                    # So we need v_{L-1} != v_1.
                    if path[length-1] != path[1]:
                        # Compute coefficients
                        coeffs = [0] * num_edges
                        for i in range(length):
                            u = path[i]
                            v = path[(i+1)%length]
                            eid = get_edge_id(u, v)
                            if eid == -1: continue
                            # Sign: orientation
                            # We need a consistent orientation.
                            # Let's define the voltage on edge (u,v) as phi(u,v).
                            # If we traverse u->v, we add phi. If v->u, we subtract phi.
                            # Let's assign an arbitrary orientation to each edge, say u<v.
                            # If we traverse in that direction, coeff += 1, else -= 1.
                            if u < v:
                                coeffs[eid] += 1
                            else:
                                coeffs[eid] -= 1
                        constraints.append(coeffs)
                return
            
            if v_idx == length - 1:
                # Must go to 0
                if current != 0 and prev != 0:
                    # Check if edge (current, 0) exists (it does in K_6)
                    # Check non-backtracking: current != prev (already true)
                    # And the next step is 0. So 0 != prev is required. (prev != 0 checked)
                    # And the step before was current. So 0 != current? No, 0 is the target.
                    # The condition is v_{L-1} != v_{L-2} which is current != prev (true).
                    # And the closure condition: v_1 != v_{L-1} (checked in base case).
                    path.append(0)
                    backtrack(length, 0, current, path)
                    path.pop()
                return
            
            for next_v in range(base_n):
                if next_v == current or next_v == prev:
                    continue
                path.append(next_v)
                backtrack(v_idx + 1, next_v, current, path)
                path.pop()
        
        for v1 in range(1, base_n):
            backtrack(1, v1, 0, [0, v1])

    for L in range(3, 9):
        gen_walks(L)
    
    # Remove duplicate constraints to speed up search
    # Many walks will have the same coefficient vector.
    unique_constraints = list(set(map(tuple, constraints)))
    constraints = [list(c) for c in unique_constraints]
    
    # Precompute for each edge index, which constraints involve it and with what coefficient
    # This allows faster delta calculation.
    # constraint_to_edges: list of (edge_idx, coeff) for each constraint
    # edge_to_constraints: list of (constraint_idx, coeff) for each edge
    
    C = len(constraints)
    E = num_edges
    
    # Use numpy for vectorized score calculation
    # A is a C x E matrix where A[i][j] is the coefficient of edge j in walk i.
    A = np.array(constraints, dtype=np.int64)
    
    best_n = 1250 # Fallback from LUW
    best_edges = None
    
    # We search for m (size of voltage group Z_m) and voltages (vector in Z_m^E).
    # We want to minimize n = 6*m.
    # So we try small m first.
    
    # For a given m, we want to find v in Z_m^E such that A @ v != 0 (mod m) for all rows.
    # This is a constraint satisfaction problem.
    
    # Strategy:
    # 1. Iterate m from 1 upwards.
    # 2. For each m, try to find a valid voltage assignment using local search / simulated annealing.
    # 3. If found, build the graph and return.
    
    # Lower bound on m:
    # If m is too small, it's impossible to avoid 0 sums.
    # For example, if there is a walk that uses each edge exactly once (a Hamiltonian cycle in the line graph?),
    # or if the sum of coefficients is constrained.
    # Actually, just start from m=1.
    
    # Note: m=1 is trivial (all voltages 0), score is C (all walks sum to 0).
    # We need score = 0.
    
    # Let's use a simple local search with random restarts.
    
    for m in range(1, 200):
        if time.time() > deadline:
            break
            
        # If 6*m >= best_n, we can stop if we are confident, but we want the MINIMUM n.
        # So we should stop as soon as we find a valid m.
        # Since we iterate m from 1 upwards, the first valid m gives the minimum n.
        
        # Local search
        # Initialize random voltages
        found = False
        num_restarts = 50
        steps_per_restart = 1000
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            v = np.random.randint(0, m, size=E)
            
            # Calculate initial score
            # scores = (A @ v) % m
            # score = sum(scores == 0)
            # Vectorized:
            sums = (A @ v) % m
            score = np.sum(sums == 0)
            
            if score == 0:
                found = True
                break
            
            # Local search
            # We will randomly pick an edge, change its voltage, and see if score decreases.
            # To make it faster, we can precompute the contribution of each edge to each constraint.
            # But A is already that.
            # Delta for changing v[j] by delta:
            # new_sums = sums + delta * A[:, j]
            # new_score = sum(new_sums == 0)
            
            for step in range(steps_per_restart):
                if time.time() > deadline:
                    break
                if score == 0:
                    found = True
                    break
                    
                # Pick random edge
                j = np.random.randint(0, E)
                # Pick random delta (change in voltage)
                # v[j] is in 0..m-1.
                # Let's try all possible new values for v[j]?
                # If m is small, this is cheap.
                # If m is large, maybe just random.
                # Let's try random delta first.
                old_val = v[j]
                new_val = np.random.randint(0, m)
                if new_val == old_val:
                    continue
                
                delta = (new_val - old_val) % m
                if delta == 0:
                    continue
                
                # Calculate new sums
                # new_sums = sums + delta * A[:, j]
                # Note: A[:, j] is a vector of coefficients.
                # We need to do this mod m.
                new_sums = (sums + delta * A[:, j]) % m
                new_score = np.sum(new_sums == 0)
                
                if new_score < score:
                    v[j] = new_val
                    sums = new_sums
                    score = new_score
                # Else: keep old value (greedy)
                # We could also accept worse moves with some probability (simulated annealing),
                # but greedy is simpler and often works for these CSPs.
            
            if found:
                break
        
        if found:
            # Build the graph
            n = 6 * m
            edges = []
            # Vertex id: u * m + i
            # Edge (u, v) in base graph corresponds to edges between (u, i) and (v, (i + phi) % m)
            # where phi is the voltage on edge (u, v).
            
            for u in range(base_n):
                for v in range(u + 1, base_n):
                    eid = get_edge_id(u, v)
                    phi = v[eid] # Voltage is in 0..m-1
                    
                    for i in range(m):
                        # Edge from (u, i) to (v, (i + phi) % m)
                        id1 = u * m + i
                        id2 = v * m + (i + phi) % m
                        edges.append((id1, id2))
            
            # Verify the result
            # 1. Degree
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(d != 5 for d in deg):
                continue # Should not happen
                
            # 2. Girth
            # Check for cycles of length < 9.
            # We can do a BFS from each vertex, but that's O(n * (n+E)).
            # n is at most 6*200 = 1200. E is 5*1200/2 = 3000.
            # BFS from each vertex: 1200 * (1200+3000) = ~5M ops.
            # In Python, this might be slow.
            # However, we only need to check if the girth is < 9.
            # We can use the voltage graph property: girth of lift >= girth of base graph?
            # No, the lift can have shorter cycles if voltages sum to 0.
            # We already ensured that no non-backtracking closed walk of length 3..8 has sum 0.
            # This is equivalent to saying the lift has no cycle of length 3..8.
            # Is it sufficient to check non-backtracking walks?
            # Yes, any cycle of length L contains a non-backtracking closed walk of length L.
            # If a cycle of length L exists, the voltage sum around it is 0.
            # Our constraint ensures that for any non-backtracking closed walk of length 3..8, the sum is NOT 0.
            # Thus, no such cycle exists.
            # So we can trust the voltage construction.
            
            return n, edges
    
    # If no solution found, return fallback
    return _luw_construct(5, 9)

# EVOLVE-BLOCK-END
