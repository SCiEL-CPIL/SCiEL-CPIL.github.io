import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Construct LUW graph as baseline
    # This is the seed program's logic, optimized slightly
    q = k
    # For LUW: girth >= d + 5 if d is odd, d + 6 if d is even
    # We need girth >= g
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        mult_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        def add(x, y): return add_table[x][y]
        def sub(x, y): return add_table[x][y]
        def mul(x, y): return mult_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    rule = [None] * (d + 1)
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0: rule[c] = (1, c0 - 2)
        elif r == 1: rule[c] = (c0 - 1, 1)
        elif r == 2: rule[c] = (c0, 1)
        else: rule[c] = (1, c0 + 1)
    
    def get_neighbours(side, x):
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = 0
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    luw_edges = []
    
    for vertex in order:
        if time.time() > deadline: break
        side, x = vertex
        for other in get_neighbours(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                luw_edges.append((index[vertex], index[other]))
                
    luw_n = len(order)
    
    # 2. Try to find a smaller graph via random search
    # Only try if we have time and if luw_n is large enough to warrant the risk
    # We search for N in [luw_n - 50, luw_n - 1]
    # This is a gamble. If we find one, we win big. If not, we return LUW.
    
    best_n = luw_n
    best_edges = luw_edges
    
    # Helper to check girth
    def check_girth(adj, n, g):
        # BFS with non-backtracking
        # dist[v][prev] = min distance
        # Use a simple BFS from each vertex? No, too slow for n=1000.
        # Use the non-backtracking walk idea from inspiration 3.
        # State: (u, prev)
        # dist[state]
        # This is O(n * k^g) which is fine.
        
        # For efficiency, we can just check if there are cycles of length < g.
        # A cycle of length L < g implies a non-backtracking walk of length L.
        
        # Implementation of non-backtracking BFS
        # We can stop early if we find a cycle.
        
        # Actually, for random graphs, 4-cycles are the main issue.
        # We can just count 4-cycles? No, need girth >= g.
        
        # Let's use the numpy approach from inspiration 3 for speed if available.
        # But standard library is safer.
        
        # Simple BFS from each vertex is O(n * (n+k)) = O(n^2).
        # For n=1000, 1,000,000 ops. 1000 vertices -> 1e9 ops. Too slow in Python.
        
        # We need the non-backtracking method.
        # States: (u, prev). Total n * (n+1) states? No, prev is a neighbor.
        # n * k states.
        # For each state, expand to k-1 neighbors.
        # Depth g.
        # n * k * (k-1)^(g/2).
        # 1000 * 5 * 4^4 = 1000 * 5 * 256 = 1,280,000. Very fast.
        
        # We need to detect if a state (u, prev) is reached at distance < g 
        # such that it forms a cycle.
        # Actually, if we reach (u, prev) and (u, prev) was visited at distance d1, 
        # and current distance is d2, and d1 + d2 < g? No.
        
        # Standard algorithm:
        # For each start vertex s:
        #   BFS on non-backtracking states.
        #   If we reach (s, t) where t is a neighbor of s, and dist < g, cycle found.
        
        # This is O(n * (n*k * (k-1)^(g/2)))? No.
        # We do BFS from each s.
        # Total work: n * (states explored per s).
        # States explored per s is bounded by k * (k-1)^(g/2).
        # Total: n * k * (k-1)^(g/2).
        # For n=1000, k=5, g=9: 1000 * 5 * 4^4 = 1.28e6.
        # This is fast enough in Python if optimized.
        
        def has_short_cycle(adj, n, g):
            # adj is list of lists
            for s in range(n):
                # BFS from s
                # State: (u, prev)
                # dist: dict or array
                # We can use a 2D array dist[u][prev_index] but prev is vertex id.
                # Use dict for sparse? No, n*k is small.
                # dist[u][v] where v is prev.
                # Initialize dist[s][s] = 0? No, prev is neighbor.
                # Start states: (s, t) for t in adj[s], dist = 1.
                
                # If we reach (s, t) again? No, that's backtracking.
                # We stop if we reach (s, t) with t != prev?
                # Actually, a cycle of length L starting at s is a walk s -> ... -> s.
                # In non-backtracking, we start at (s, t1), end at (s, tL).
                # If tL == t1, it's a backtrack? No.
                
                # Let's use the standard "girth >= g" check.
                # If there is a cycle of length L < g, then for some s, 
                # there is a non-backtracking walk of length L from s to s.
                
                # We perform BFS from each (s, t) for t in adj[s].
                # If we reach (s, u) where u != t (the start prev) and dist < g, 
                # we have a cycle of length dist + 1?
                
                # This is getting complex. Let's use the numpy vectorized approach 
                # from Inspiration 3 if it's fast.
                
                # For now, return False (assume no short cycle) to allow the 
                # random graph to be accepted? NO, that's invalid.
                # We MUST verify.
                
                # Let's implement a simple BFS check for cycles < g.
                # For each start s:
                #   Queue: [(s, -1, 0)]
                #   Visited: set of (u, prev)
                #   While Queue:
                #     u, prev, dist = pop
                #     If dist >= g: continue
                #     For v in adj[u]:
                #       If v == prev: continue
                #       If v == s and dist + 1 >= 3:
                #          # Found cycle of length dist + 1
                #          if dist + 1 < g: return True
                #       If (v, u) not in Visited:
                #          Visited.add((v, u))
                #          Queue.append((v, u, dist + 1))
                #   If no cycle found from s, continue
                # Return False
                
                # This is O(n * (n*k * (k-1)^(g/2)))?
                # The number of states is n*k.
                # We do this for each s.
                # Total O(n^2 * k * (k-1)^(g/2)).
                # 1000^2 * 5 * 256 = 1.28e9. Too slow in Python.
                
                # We need to be smarter.
                # We can just check 4-cycles and 6-cycles?
                # For g=9, we need to check 4, 6, 8.
                # Counting 4-cycles is O(n * k^3).
                # 1000 * 125 = 125,000. Fast.
                # Counting 6-cycles is O(n * k^5)?
                # 1000 * 3125 = 3e6. Fast.
                # Counting 8-cycles?
                # This is getting complicated.
                
                # Given the time, I will NOT implement the random search verification 
                # unless I am sure it's fast.
                # I will just return the LUW graph.
                
                return False # Placeholder

            return False
        
        # Since verification is hard to make fast in pure Python for n=1000,
        # and random graphs likely have short cycles, the random search is 
        # likely to fail or produce invalid graphs.
        
        # Therefore, the best strategy is to return the LUW graph.
        
    return luw_n, luw_edges
