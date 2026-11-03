import time
import sys
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # If k is not 3, we might need a different strategy, but the task specifies k=3, g=16.
    # The prompt says "the evaluator passes this task's degree k and girth g".
    # So k and g are variables.
    
    if k != 3:
        # Fallback for other k: use LUBW if possible, or random
        # For now, focus on k=3
        pass

    q = k
    if q != 3:
        # If k is not 3, the GP search might not be applicable or optimal.
        # We'll stick to LUBW for non-3 degrees as a safe bet, 
        # though the task is specifically for the (3,16) cage.
        pass

    def girth_of_d(d):
        # Approximate girth for LUBW
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1
        
    # GF(q) operations for LUBW
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    # Rule for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    def build_luw():
        index = {}
        order = []
        edges = []
        
        start = (0, (None,) + (0,) * d)
        index[start] = 0
        order.append(start)
        
        for i, vertex in enumerate(order):
            if time.time() > deadline:
                break
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                
                neighbor = (1 - side, tuple(y))
                
                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)
                
                if side == 0:
                    edges.append((index[vertex], index[neighbor]))
                    
        n = len(order)
        return n, edges

    def check_girth(n, edges, min_girth):
        """
        Checks if the graph has girth at least min_girth.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_depth = min_girth // 2 - 1
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            
            while queue:
                curr = queue.popleft()
                if dist[curr] > max_depth:
                    continue
                for neighbor in adj[curr]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                    elif neighbor != parent[curr]:
                        # Found a cycle
                        if dist[curr] + dist[neighbor] + 1 < min_girth:
                            return False
        return True

    # Strategy:
    # 1. Try to find a Generalized Petersen Graph with girth >= g.
    #    GP(n, k) has 2n vertices.
    #    We search n from 5 upwards.
    # 2. If found, return it.
    # 3. If not found within time, return LUBW graph.

    if k == 3:
        # Search for GP graphs
        # Max n to search: 
        # LUBW size is approx 2 * 3^d. For g=16, d=7, size=4374.
        # So we want 2n < 4374 => n < 2187.
        # We can search up to n=2000.
        
        max_n_search = 2000
        
        for n in range(5, max_n_search + 1):
            if time.time() > deadline - 10:  # Leave 10s for LUBW fallback
                break
                
            # k_gp must be coprime to n for the graph to be connected?
            # Actually, GP(n,k) is connected iff gcd(n,k)=1.
            # If gcd(n,k) != 1, it's a union of components.
            # The girth of a disconnected graph is the min girth of components.
            # But we want a single graph. Usually "graph" implies connected.
            # The task says "a 3-regular graph". It doesn't explicitly say connected,
            # but cage problems usually imply connected.
            # If disconnected, the girth is the min of the components.
            # We will only consider connected GP graphs (gcd(n,k)=1).
            
            for k_gp in range(1, n // 2 + 1):
                if time.time() > deadline - 10:
                    break
                
                import math
                if math.gcd(n, k_gp) != 1:
                    continue
                
                # Construct edges
                edges_gp = []
                # Outer cycle: u_i to u_{i+1}
                for i in range(n):
                    edges_gp.append((i, (i + 1) % n))
                # Spokes: u_i to v_i
                for i in range(n):
                    edges_gp.append((i, n + i))
                # Inner cycle: v_i to v_{i+k_gp}
                for i in range(n):
                    edges_gp.append((n + i, n + (i + k_gp) % n))
                
                # Check girth
                # Optimization: We can check girth quickly.
                # If girth < g, skip.
                # To save time, we can do a quick BFS check.
                # But check_girth is O(n). 
                # 2000 * 1000 * 2000 = 4e9 ops? Too slow.
                # We need a faster girth check or limit the search.
                
                # Let's limit the search to n < 500 first.
                # If we don't find one, we try LUBW.
                # Actually, 2000 is too many.
                # Let's reduce max_n_search to 500.
                # 500 * 250 * 1000 (avg check cost) = 1.25e8. Feasible.
                
                if n > 500:
                    break # Inner break logic needs to be handled outside or via flag
                
                if check_girth(2 * n, edges_gp, g):
                    return 2 * n, edges_gp

    # Fallback: LUBW
    n_luw, edges_luw = build_luw()
    return n_luw, edges_luw
