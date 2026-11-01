"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(k, g):
    # Voltage graph construction: use a small base graph and assign voltages from Z_m
    # to create a k-regular graph with girth >= g.
    
    deadline = time.time() + 145  # Leave 5s safety margin
    
    # For k=5, g=9: use a 5-cycle as base graph with voltage assignments
    # The lift will be 5-regular if we use a 5-cycle base with one edge per vertex
    
    # Strategy: Use a base graph B with n_b vertices, each of degree k.
    # Assign voltages in Z_m to edges. The lift has n_b * m vertices.
    # Girth of lift >= girth of B, but we need to ensure no short cycles in the lift.
    
    # For k=5: Use K6 (complete graph on 6 vertices) as base. Each vertex has degree 5.
    # Assign random voltages in Z_m and check the resulting graph.
    
    def check_girth(adj, n, target_girth, max_depth=None):
        """Check if graph has girth >= target_girth using BFS from each vertex."""
        if max_depth is None:
            max_depth = (target_girth - 1) // 2
        
        for start in range(n):
            # BFS with parent tracking to avoid backtracking
            visited = [[-1] * n for _ in range(max_depth + 1)]
            visited[0][start] = -2  # -2 marks start
            
            queue = [(start, 0, -1)]  # (vertex, depth, parent)
            
            while queue:
                v, depth, parent = queue.pop(0)
                if depth >= max_depth:
                    continue
                
                for u in adj[v]:
                    if u == parent:
                        continue  # No backtracking
                    if visited[depth + 1][u] != -1:
                        return False  # Found cycle of length <= 2*(depth+1)
                    visited[depth + 1][u] = v
                    queue.append((u, depth + 1, v))
        
        return True
    
    def build_lift(base_edges, voltages, m):
        """Build the voltage graph lift."""
        n_b = max(v for e in base_edges for v in e) + 1
        n = n_b * m
        adj = [[] for _ in range(n)]
        
        for (u, v), vol in zip(base_edges, voltages):
            # Edge from u to v with voltage vol
            # In the lift: (u, x) connected to (v, x + vol) mod m
            for x in range(m):
                u_lift = u * m + x
                v_lift = v * m + ((x + vol) % m)
                adj[u_lift].append(v_lift)
                # Undirected: also add reverse
                # But we need to be careful not to duplicate
                # Actually for undirected graphs, we add both directions
                # But in the lift, the reverse edge has voltage -vol
        
        # For undirected graphs, each edge in base gives two directed edges in lift
        # Let's rebuild properly
        adj = [[] for _ in range(n)]
        for (u, v), vol in zip(base_edges, voltages):
            for x in range(m):
                u_lift = u * m + x
                v_lift = v * m + ((x + vol) % m)
                adj[u_lift].append(v_lift)
                # Reverse direction: from (v, y) to (u, y - vol) mod m
                # where y = (x + vol) % m, so x = (y - vol) % m
                # From (v, y) to (u, (y - vol) % m)
                for y in range(m):
                    pass  # This is getting complicated, let's use a simpler approach
        
        # Simpler: for each undirected edge (u,v) with voltage vol,
        # in the lift we have edges: (u, x) - (v, x+vol) for all x
        # This automatically handles both directions since if (u,x)-(v,x+vol),
        # then from v's perspective, (v, x+vol) connects to (u, x) = (u, (x+vol)-vol)
        
        adj = [[] for _ in range(n)]
        for (u, v), vol in zip(base_edges, voltages):
            for x in range(m):
                u_lift = u * m + x
                v_lift = v * m + ((x + vol) % m)
                adj[u_lift].append(v_lift)
        
        # Verify regularity
        for i in range(n):
            if len(adj[i]) != k:
                return None, None
        
        # Convert to edge list
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        return n, edges
    
    # Base graph: K6 for k=5
    base_edges = []
    for i in range(6):
        for j in range(i+1, 6):
            base_edges.append((i, j))
    
    # Try different values of m and random voltage assignments
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (5,9): n >= 2 * (1 + 5 + 5^2 + 5^3) = 2 * 181 = 362
    # But for bipartite it's different. Let's just search.
    
    # Try m values that give reasonable n
    # n = 6 * m, so m should be at least 62 to get n >= 372
    # Let's try m from 60 to 100
    
    random.seed(42)
    
    for m in range(50, 120):
        if time.time() > deadline:
            break
        
        # Try a few random voltage assignments
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Random voltages in Z_m
            voltages = [random.randint(0, m-1) for _ in base_edges]
            
            n, edges = build_lift(base_edges, voltages, m)
            if n is None:
                continue
            
            if n >= best_n:
                continue
            
            # Check girth
            # Build adjacency list for girth check
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            if check_girth(adj, n, g):
                best_n = n
                best_edges = edges
                # If we found a good one, we can break early
                if n < 400:  # Well below the known best
                    return n, edges
    
    # Fallback to the original LUW construction if voltage graph didn't work well
    if best_edges is not None:
        return best_n, best_edges
    
    # Original LUW construction as fallback
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
# EVOLVE-BLOCK-END
