import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Baseline: LUW Construction
    # This is fast and guarantees a valid graph, though large.
    n_luw, edges_luw = _luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. Heuristic Search for Smaller Graphs
    # Moore Bound
    # n >= 1 + k * sum(k-1)^i for i=0 to (g-3)/2
    depth = (g - 3) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(depth + 1))
    
    # Search range
    # We start from moore_bound and go up.
    # Given the gap, we might not find one very close to the bound.
    # We will try a few specific n values or a range.
    
    # To be efficient, we will try n values in a strategic range.
    # If we find a valid graph, we update best.
    
    # Local Search Function
    def try_local_search(n, max_iter=20000, timeout=None):
        if timeout is None:
            timeout = time.time() + 5 # 5 seconds per attempt
        
        # Initialize random k-regular graph
        # Using configuration model or similar
        # For simplicity, we can use a random permutation swap method on a base graph
        # Or just generate random edges and check regularity? Hard.
        # Let's use a simple random regular graph generator.
        
        def generate_random_regular_graph(n, k):
            # This is a simplified generator. For k=5, n must be even?
            # Sum of degrees = nk. Must be even.
            if (n * k) % 2 != 0:
                return None
            # Simple method: Start with a cycle and add random chords?
            # Or use the "random permutation" method for regular graphs.
            # We'll use a heuristic: Start with a known regular graph (e.g., cycle + chords) 
            # or just random edges and fix degrees.
            
            # For robustness, let's use a simple "random edges" approach with repair?
            # Better: Use the "configuration model" conceptually but implement via swaps.
            # Start with a disjoint union of cycles (if possible) or a random graph and 
            # iteratively swap edges to make it regular.
            
            # Actually, for k=5, we can start with a "blow-up" of a small regular graph?
            # Or just generate a random graph and check.
            
            # Let's try a simple random graph generation:
            # Create list of stubs.
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            edges = set()
            valid = True
            for i in range(0, n * k, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    # Self loop, try to fix by swapping with another stub
                    # For simplicity, restart or mark invalid
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((min(u, v), max(u, v)))
                adj[u].append(v)
                adj[v].append(u)
            
            if not valid:
                return None
            
            return adj, list(edges)
        
        adj, edges = generate_random_regular_graph(n, k)
        if adj is None:
            return None
            
        # Local Search
        for _ in range(max_iter):
            if time.time() > timeout:
                break
                
            # Check girth
            girth = _calculate_girth(adj, n, g)
            if girth >= g:
                return n, edges
                
            # Find a short cycle to break
            # We can just do random swaps and hope to improve girth
            # Or target the shortest cycle.
            
            # Random Swap
            u = random.randint(0, n-1)
            if not adj[u]: continue
            v = random.choice(adj[u])
            
            # Find another edge to swap with
            # Pick random u2
            u2 = random.randint(0, n-1)
            if u2 == u or u2 == v: continue
            if not adj[u2]: continue
            v2 = random.choice(adj[u2])
            
            if u2 == v or v2 == u or u2 == v2 or u2 == v:
                continue
            
            # Check if swap is valid (no multi-edges, no loops)
            # Remove (u,v) and (u2,v2)
            # Add (u,v2) and (u2,v)
            
            # Check existence
            if v2 in adj[u] or v in adj[u2]:
                continue
                
            # Perform swap
            adj[u].remove(v)
            adj[u].append(v2)
            adj[v].remove(u)
            adj[v].append(u2)
            
            adj[u2].remove(v2)
            adj[u2].append(v)
            adj[v2].remove(u2)
            adj[v2].append(u)
            
            # Update edges set
            edges.remove((min(u, v), max(u, v)))
            edges.remove((min(u2, v2), max(u2, v2)))
            edges.add((min(u, v2), max(u, v2)))
            edges.add((min(u2, v), max(u2, v)))
            
        return None

    # Search Loop
    # We try n values.
    # Start from moore_bound, but maybe jump.
    # Given time, we can try a few specific n values.
    # If we find one, we return it.
    
    # To maximize chance, we can try n in a range.
    # But local search on large n is slow.
    # Let's try a few restarts for a few n values.
    
    # We will try n from moore_bound to moore_bound + 200
    # And maybe some larger values if time permits.
    
    # Actually, the LUW graph is ~6000.
    # If we find a graph of size 1500, that's a big improvement.
    
    # Let's try n values: moore_bound, moore_bound+10, ...
    # But girth 9 is very hard.
    # Maybe we should stick to the LUW if search fails?
    
    # Let's run search for a few seconds.
    
    # We will try a specific n that is known to be small?
    # No, we don't know.
    
    # Let's try a range of n values.
    # To be safe, we will only update best if we find a valid graph with n < best_n.
    
    # We will use the remaining time for search.
    
    # Strategy:
    # 1. Try n = moore_bound, moore_bound+1, ... up to some limit or time out.
    # 2. If not found, try larger n.
    
    # Given the difficulty, we might not find anything smaller than LUW.
    # But it's worth a shot.
    
    # We will try a few specific n values with limited time each.
    n_candidates = []
    # Add moore bound and some increments
    for i in range(0, 100, 5):
        n_candidates.append(moore_bound + i)
    # Add some larger values
    for i in range(500, 2000, 100):
        n_candidates.append(i)
        
    random.shuffle(n_candidates)
    
    for n in n_candidates:
        if time.time() > deadline - 5:
            break
        if n >= best_n:
            continue
            
        result = try_local_search(n, max_iter=5000, timeout=time.time() + 2)
        if result:
            n_found, edges_found = result
            if n_found < best_n:
                best_n = n_found
                best_edges = edges_found
                # If we found a very good one, we can stop or continue to improve
                # But we need to ensure validity.
                
    return best_n, best_edges

def _calculate_girth(adj, n, limit):
    # BFS from each vertex
    # Return shortest cycle length
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Cycle found
                    # Length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth < limit:
                        return min_girth # Early exit
    return min_girth if min_girth != float('inf') else 0

def _luw_construct(k, g):
    # Same as original, but cleaned up
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
