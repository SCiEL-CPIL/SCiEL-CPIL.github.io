import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        
        # If no edges, return empty
        if not edges:
            return []
            
        # Build adjacency bitmask for fast operations
        # adj_mask[i] is an integer where bit j is 1 if i and j are connected
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        # All vertices mask
        all_vertices = (1 << n) - 1
        
        # Greedy heuristic to get an initial upper bound
        # Pick vertex with max degree, remove it and its neighbors
        def greedy_vc():
            remaining = all_vertices
            cover = set()
            # Copy adjacency for dynamic removal? 
            # Simpler: just pick max degree from remaining, remove from remaining
            # But we need to ensure we cover all edges.
            # Standard greedy: while edges remain, pick vertex incident to most uncovered edges.
            # For small n, simple approach:
            # Iterate, find vertex with max degree in subgraph induced by 'remaining'?
            # Actually, MVC is not simply picking max degree. 
            # Let's use a simple valid cover: all vertices with degree > 0 is a valid cover? No, that's a dominating set or just a cover.
            # Yes, the set of all non-isolated vertices is a valid vertex cover.
            # Size is at most n.
            # Better: 
            # While there are edges, pick a vertex, add to cover, remove all incident edges.
            # To be efficient, we can just return a decent upper bound.
            # Let's use the "all non-isolated" as a safe, fast upper bound.
            non_isolated = 0
            for i in range(n):
                if adj_mask[i] != 0:
                    non_isolated |= (1 << i)
            return non_isolated, bin(non_isolated).count('1')

        initial_mask, best_size = greedy_vc()
        
        # Branch and Bound
        # State: remaining vertices mask, current cover mask, current size
        # We want to minimize current size + min_vc(remaining)
        
        # Lower bound: 
        # A simple lower bound is the size of a maximal matching in the remaining graph.
        # Finding a maximal matching is fast.
        def get_max_matching_lb(mask):
            # Greedy maximal matching
            matched = 0
            m = mask
            count = 0
            while m:
                # Pick lowest set bit
                u_bit = m & -m
                u = u_bit.bit_length() - 1
                m ^= u_bit
                # Find neighbor in m
                neigh = adj_mask[u] & m
                if neigh:
                    v_bit = neigh & -neigh
                    v = v_bit.bit_length() - 1
                    m ^= v_bit
                    count += 1
            return count

        def branch(mask, current_size):
            nonlocal best_size, initial_mask
            
            if current_size >= best_size:
                return
                
            if mask == 0:
                if current_size < best_size:
                    best_size = current_size
                    initial_mask = 0 # We don't need to track the mask if we just want size? 
                    # Wait, we need to return the vertices. 
                    # So we should track the path or just return the size and reconstruct?
                    # For n=40, tracking the set in recursion is fine if we pass a set or mask.
                    # Let's pass a mask for the cover as well? 
                    # Or just return the min size and then do a second pass?
                    # No, let's store the best mask.
                return

            # Lower bound
            lb = get_max_matching_lb(mask)
            if current_size + lb >= best_size:
                return

            # Pick an edge to branch on
            # Find any edge in the remaining graph
            # Iterate through vertices in mask
            m = mask
            edge_found = False
            u = -1
            v = -1
            while m:
                u_bit = m & -m
                u = u_bit.bit_length() - 1
                m ^= u_bit
                # Check neighbors in mask
                neigh = adj_mask[u] & mask
                if neigh:
                    v_bit = neigh & -neigh
                    v = v_bit.bit_length() - 1
                    edge_found = True
                    break
            
            if not edge_found:
                # No edges left, so we don't need to add more vertices
                if current_size < best_size:
                    best_size = current_size
                    # Store the solution? 
                    # We need to reconstruct the set. 
                    # It's easier to keep track of the chosen vertices in the recursion.
                    # Let's modify branch to take a list/set of chosen vertices.
                    pass
                return

            # Branch 1: Include u
            # Remove u and all its neighbors from 'mask' (they are covered)
            new_mask_1 = mask & ~(adj_mask[u] | (1 << u))
            branch(new_mask_1, current_size + 1)
            
            # Branch 2: Include v
            new_mask_2 = mask & ~(adj_mask[v] | (1 << v))
            branch(new_mask_2, current_size + 1)

        # To return the actual vertices, we need to track the path.
        # Let's use a global best_mask variable.
        best_mask = initial_mask
        
        def branch2(mask, current_size, current_mask):
            nonlocal best_size, best_mask
            
            if current_size >= best_size:
                return
                
            if mask == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_mask = current_mask
                return

            # Lower bound
            lb = get_max_matching_lb(mask)
            if current_size + lb >= best_size:
                return

            # Pick an edge
            m = mask
            edge_found = False
            u = -1
            v = -1
            while m:
                u_bit = m & -m
                u = u_bit.bit_length() - 1
                m ^= u_bit
                neigh = adj_mask[u] & mask
                if neigh:
                    v_bit = neigh & -neigh
                    v = v_bit.bit_length() - 1
                    edge_found = True
                    break
            
            if not edge_found:
                if current_size < best_size:
                    best_size = current_size
                    best_mask = current_mask
                return

            # Branch 1: Include u
            new_mask_1 = mask & ~(adj_mask[u] | (1 << u))
            branch2(new_mask_1, current_size + 1, current_mask | (1 << u))
            
            # Branch 2: Include v
            new_mask_2 = mask & ~(adj_mask[v] | (1 << v))
            branch2(new_mask_2, current_size + 1, current_mask | (1 << v))

        branch2(all_vertices, 0, 0)
        
        # Extract vertices from best_mask
        result = []
        for i in range(n):
            if best_mask & (1 << i):
                result.append(i)
                
        return sorted(result)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
