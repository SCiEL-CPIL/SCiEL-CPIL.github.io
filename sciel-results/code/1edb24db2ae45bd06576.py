import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 18
    # Graph representation: list of sets of neighbors
    # Or bitmask for speed
    # Let's use a list of sets for flexibility, and bitmask for speed in checks
    
    # Initialize a random K4-free graph
    # Start with empty graph
    graph = [set() for _ in range(n)]
    
    def add_edge(u, v):
        graph[u].add(v)
        graph[v].add(u)
        
    def remove_edge(u, v):
        graph[u].discard(v)
        graph[v].discard(u)
        
    def is_k4_free():
        # Check all 4-cliques
        # Too slow for frequent checks. 
        # Better to check incrementally when adding edges.
        pass
        
    def check_add_edge(u, v):
        # Check if adding (u, v) creates a K4
        # Common neighbors of u and v
        common = graph[u].intersection(graph[v])
        # If there is an edge (w, x) in common, then (u, v, w, x) is a K4
        for w in common:
            for x in common:
                if w < x and x in graph[w]:
                    return False
        return True

    # Generate initial graph
    # Try to add random edges
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges)
    for u, v in edges:
        if check_add_edge(u, v):
            add_edge(u, v)
            
    # Function to compute max triangle-free induced subgraph size
    def max_triangle_free_size():
        # Use branch and bound
        # State: set of available vertices
        # We want to find the max size of a subset S such that G[S] is triangle-free
        
        # Precompute triangles?
        # A set S is triangle-free if it contains no triangle.
        # Let's use a recursive search.
        
        # Heuristic: Greedy
        # Pick a vertex, include it, remove its neighbors' neighbors? No.
        # If we include v, we cannot include any pair (u, w) that forms a triangle with v.
        # i.e., if u, w in N(v) and (u, w) is an edge, we cannot include both u and w.
        
        # This is complex. Let's use a simpler approach for n=18.
        # Iterate over all subsets? 2^18 = 262144.
        # For each subset, check if it's triangle-free.
        # This is 262144 * (cost to check).
        # Cost to check: iterate over triangles in the subset.
        # Number of triangles is at most 216.
        # 262144 * 216 ~ 56M ops. Too slow in Python for every move.
        
        # Optimized check:
        # Represent subset as bitmask.
        # Precompute list of triangle bitmasks.
        # For a given subset mask, check if (mask & tri_mask) == tri_mask for any tri_mask.
        
        # Let's precompute triangle masks for the current graph.
        tri_masks = []
        for i in range(n):
            for j in range(i+1, n):
                if j in graph[i]:
                    common = graph[i].intersection(graph[j])
                    for k in common:
                        if k > j:
                            mask = (1 << i) | (1 << j) | (1 << k)
                            tri_masks.append(mask)
        
        # Now find max subset with no triangle
        # Use a greedy heuristic first to get a lower bound
        # Then branch and bound?
        
        # Actually, for n=18, we can just iterate k from 18 down to 1
        # and check if there is a triangle-free set of size k.
        # But checking all subsets of size k is C(18, k).
        # Max C(18, 9) is 48620.
        # For each subset, check triangles.
        # This is feasible for a few calls, but not thousands.
        
        # Let's use a recursive solver with pruning.
        # Sort vertices by degree?
        
        # Alternative: Use the fact that we are doing local search.
        # We can maintain the triangle masks incrementally?
        
        # Let's try a simple recursive solver.
        # max_tri_free(available_mask)
        # Pick a vertex v in available_mask.
        # Option 1: Exclude v.
        # Option 2: Include v. Then we must exclude any pair (u, w) in N(v) that are connected.
        # This is not a simple independent set.
        
        # Let's use the bitmask iteration method but optimized.
        # We want the max popcount of mask such that no tri_mask is subset of mask.
        
        # We can iterate masks in decreasing order of popcount.
        # Stop at the first valid one.
        
        # To speed up, we can use a heuristic to order masks?
        
        # Given the time limit, let's just do the C(18, k) check for k=18, 17, ...
        # And hope we find the answer quickly.
        
        from itertools import combinations
        
        for k in range(n, 0, -1):
            for comb in combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                is_tf = True
                for tm in tri_masks:
                    if (mask & tm) == tm:
                        is_tf = False
                        break
                if is_tf:
                    return k
        return 0

    # This is too slow.
    # Let's use a faster heuristic for the search, and exact only for final verification or when close.
    
    # Actually, let's use a different strategy.
    # Start with Turan graph T3(18).
    # Alpha2 = 12.
    # Try to improve.
    
    # Reset graph to T3(18)
    for i in range(n):
        graph[i].clear()
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    add_edge(u, v)
                    
    best_alpha = max_triangle_free_size()
    best_graph = [set(s) for s in graph]
    
    while time.time() < deadline:
        # Try random swaps
        # Pick random edge to remove, random non-edge to add
        u1, v1 = random.choice([(i, j) for i in range(n) for j in range(i+1, n) if j in graph[i]])
        u2, v2 = random.choice([(i, j) for i in range(n) for j in range(i+1, n) if j not in graph[i]])
        
        # Save state
        old_graph = [set(s) for s in graph]
        
        remove_edge(u1, v1)
        
        # Check if we can add u2, v2
        if check_add_edge(u2, v2):
            add_edge(u2, v2)
            new_alpha = max_triangle_free_size()
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_graph = [set(s) for s in graph]
            else:
                # Restore
                for i in range(n):
                    graph[i] = old_graph[i]
        else:
            # Restore
            for i in range(n):
                graph[i] = old_graph[i]

    # Return edges of best_graph
    edges = []
    for i in range(n):
        for j in best_graph[i]:
            if i < j:
                edges.append([i, j])
    return edges
