import random
import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # 1. Try voltage graph lifts
    # Base graph size v_b, group size m
    # Total vertices n = v_b * m
    
    # We want n to be as small as possible, but >= Moore bound ~ 426
    # Let's target n in [426, 1000]
    
    # Base graphs:
    # We need k-regular graphs.
    # Small k-regular graphs:
    # k=5: K6 (v=6), etc.
    
    # We will iterate over possible (v_b, m) pairs
    # v_b must be at least k+1 = 6
    # m must be an integer
    
    # Strategy:
    # For v_b from 6 to 20:
    #   Generate random v_b-regular graphs (or use known ones)
    #   For m from 50 to 200:
    #      If v_b * m > 1200: break
    #      Try random voltage assignments
    #      Check girth
    
    # To be efficient, we can check the girth of the base graph's closed walks.
    # A closed walk of length L in B corresponds to a closed walk in G.
    # Its voltage sum is the sum of voltages on the edges.
    # If sum == 0 mod m, it's a cycle in G of length L.
    # We need no such walk for L < g.
    
    # We can precompute all closed walks in B up to length g-1.
    # For each walk, the condition is sum(v_e) == 0 mod m.
    # This is a linear constraint on the voltages.
    # We can try to find voltages that violate as few constraints as possible, or just random search.
    
    # Random search is simpler.
    
    best_n = float('inf')
    best_edges = None
    
    # List of candidate (v_b, m)
    candidates = []
    for v_b in range(k + 1, 25):
        # v_b must allow a k-regular graph.
        # k * v_b must be even.
        if (k * v_b) % 2 != 0:
            continue
        for m in range(20, 200):
            n = v_b * m
            if n < 426: # Moore bound approx
                continue
            if n > 1500:
                break
            candidates.append((v_b, m))
            
    # Shuffle candidates to try diverse sizes
    random.shuffle(candidates)
    
    for v_b, m in candidates:
        if time.time() > deadline:
            break
            
        # Generate a random k-regular graph on v_b vertices
        # This is tricky for small v_b.
        # For v_b=6, k=5, it's K6.
        # For v_b=8, k=5, it's K8 minus a perfect matching? No, K8 is 7-reg.
        # We can use a simple random regular graph generator.
        
        # Since v_b is small, we can just try a few known structures or random permutations.
        # Let's try to construct a random k-regular graph.
        
        graph = _random_regular_graph(v_b, k)
        if graph is None:
            continue
            
        # Get edges
        edges_b = graph
        
        # Try random voltage assignments
        # We need to check if there is a closed walk of length < g with sum 0 mod m.
        # We can sample a few random assignments.
        
        # To make it faster, we can check the constraints.
        # But random search with early termination on girth check is fine for small v_b.
        
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Assign random voltages
            voltages = [random.randint(0, m - 1) for _ in range(len(edges_b))]
            
            # Check girth
            # We need to check all closed walks in B up to length g-1.
            # This is expensive if done naively for each attempt.
            # Better: precompute the walks.
            
            # For now, let's just construct the full graph and check girth?
            # No, constructing the full graph with n ~ 1000 is fast.
            # Checking girth of n=1000 graph is O(n * k^(g/2)) which is too slow for g=9?
            # k=5, g=9. BFS from each node up to depth 4.
            # 1000 * 5^4 = 625,000 ops. Very fast.
            
            # So:
            # 1. Construct derived graph.
            # 2. Check girth.
            # 3. If girth >= g, return.
            
            derived_edges = _construct_derived_graph(edges_b, voltages, m)
            
            # Check girth
            if _check_girth(derived_edges, v_b * m, k, g):
                return v_b * m, derived_edges
                
    # If no voltage graph found, fall back to L-U-W or random greedy?
    # The L-U-W is guaranteed to work but is large.
    # We can return the L-U-W result as a fallback.
    
    # Re-implement L-U-W for fallback
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

def _random_regular_graph(v, k):
    # Generate a random k-regular graph on v vertices
    # Simple method: random pairing of stubs
    if (v * k) % 2 != 0:
        return None
    stubs = []
    for i in range(v):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v2 = stubs[i], stubs[i+1]
        if u == v2:
            # Self loop, retry or swap
            # For simplicity, if self loop, swap with another
            # This might fail for small v, k
            return None 
        edge = (min(u, v2), max(u, v2))
        if edge in edges:
            return None
        edges.add(edge)
    return list(edges)

def _construct_derived_graph(base_edges, voltages, m):
    # Base graph has v_b vertices (implied by max id in base_edges + 1)
    # Derived graph has v_b * m vertices.
    # Vertex (u, a) is connected to (v, b) if (u, v) is an edge in base
    # and b = a + voltage(u, v) mod m (or a - voltage depending on direction)
    # Since graph is undirected, we need to be careful.
    # Let's say edge (u, v) has voltage w.
    # Then (u, a) is connected to (v, a + w) and (v, a - w) is connected to (u, a)?
    # Standard voltage graph:
    # If edge u-v has voltage g, then (u, a) ~ (v, a g).
    # For undirected, we usually define voltage on directed edges.
    # Let's assume we assign voltage to each undirected edge, and treat it as symmetric?
    # No, voltage must be antisymmetric or we define it on one direction.
    # Let's define voltage w for u < v.
    # Then (u, a) ~ (v, a + w) and (v, a) ~ (u, a - w).
    
    v_b = max([max(e) for e in base_edges]) + 1 if base_edges else 1
    derived_edges = []
    edge_map = {}
    for idx, (u, v) in enumerate(base_edges):
        w = voltages[idx]
        # Direction u -> v has voltage w
        # Direction v -> u has voltage -w mod m
        for a in range(m):
            # u, a to v, a + w
            u_idx = u * m + a
            v_idx = v * m + (a + w) % m
            derived_edges.append((u_idx, v_idx))
            
            # v, a to u, a - w
            v_idx2 = v * m + a
            u_idx2 = u * m + (a - w) % m
            derived_edges.append((v_idx2, u_idx2))
            
    # Remove duplicates?
    # The set of edges should be unique.
    # (u, a) -- (v, a+w) is the same as (v, a+w) -- (u, a).
    # In the loop, when we process (v, a') where a' = a+w, we add (v, a') -- (u, a' - w) = (u, a).
    # So we are adding each edge twice.
    # We should return a set or deduplicate.
    
    unique_edges = set(derived_edges)
    return list(unique_edges)

def _check_girth(edges, n, k, g):
    # Check if graph has girth >= g
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each node up to depth (g-1)//2
    # If we find a node at depth d1 from start that was also visited at depth d2 < d1
    # from the same start, we have a cycle of length d1 + d2.
    # Actually, standard girth check:
    # For each start node s:
    #   BFS, track parent and depth.
    #   If we encounter a node v that is already visited (not parent),
    #   cycle length = depth[u] + depth[v] + 1.
    #   If cycle length < g, return False.
    
    for s in range(n):
        visited = [-1] * n
        parent = [-1] * n
        queue = [(s, 0)]
        visited[s] = 0
        while queue:
            u, d = queue.pop(0)
            if d >= g: # We only care about cycles < g
                continue
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = d + 1
                    parent[v] = u
                    queue.append((v, d + 1))
                elif parent[v] != u and parent[u] != v:
                    # Found a cycle
                    # The cycle length is d + 1 + visited[v]
                    # Wait, this logic is for simple cycles.
                    # A back edge (u, v) creates a cycle of length d + 1 + visited[v]
                    # IF the path from s to v is simple and doesn't use u.
                    # This is true in BFS tree.
                    cycle_len = d + 1 + visited[v]
                    if cycle_len < g:
                        return False
    return True
