import random
import time
import sys

def construct():
    start_time = time.time()
    
    # Try n=40 first, then 39
    for n in [40, 39, 38]:
        if time.time() - start_time > 110:
            break
            
        # Attempt to construct a valid graph on n vertices
        # Strategy: Start with complete bipartite, add internal edges, repair triangles.
        
        success = False
        attempts = 0
        while not success and attempts < 50: # 50 attempts per n
            attempts += 1
            if time.time() - start_time > 110:
                break
                
            # Initialize
            # Split into two sets A and B
            mid = n // 2
            A = list(range(mid))
            B = list(range(mid, n))
            
            # adj_masks: list of ints
            adj_masks = [0] * n
            
            # Add all edges between A and B
            for u in A:
                for v in B:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    
            # Randomly add edges within A and B
            # Probability p to add edge
            p = 0.3
            for i in range(len(A)):
                for j in range(i+1, len(A)):
                    if random.random() < p:
                        u = A[i]
                        v = A[j]
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        
            for i in range(len(B)):
                for j in range(i+1, len(B)):
                    if random.random() < p:
                        u = B[i]
                        v = B[j]
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        
            # Repair phase: Remove edges to destroy triangles
            # We iterate until no triangles
            changed = True
            while changed:
                changed = False
                # Find a triangle
                # Iterate all pairs (u, v)
                # To make it faster, we can just check random edges or all edges
                # n=40, edges ~ 400. 400 checks is fast.
                for u in range(n):
                    if changed: break
                    # Only check edges where u < v to avoid duplicates
                    # We need to know if (u,v) is an edge
                    # adj_masks[u] & (1<<v)
                    for v in range(u+1, n):
                        if adj_masks[u] & (1 << v):
                            # Check common neighbors
                            common = adj_masks[u] & adj_masks[v]
                            if common:
                                # Triangle found: u, v, w
                                # Pick a random common neighbor w
                                # Get index of a set bit in common
                                # w = (common & -common).bit_length() - 1
                                # Remove edge (u, w) or (v, w) or (u, v)
                                # Removing (u, v) is simplest
                                adj_masks[u] &= ~(1 << v)
                                adj_masks[v] &= ~(1 << u)
                                changed = True
                                break
                                
            # Check if independent set of size 10 exists
            # If it does, we need to add edges to reduce alpha
            # We'll loop: if alpha >= 10, try to add an edge
            repair_alpha = True
            alpha_attempts = 0
            while repair_alpha and alpha_attempts < 100:
                alpha_attempts += 1
                if time.time() - start_time > 110:
                    break
                    
                # Check alpha
                # Use fast checker
                if not has_independent_set_10(adj_masks, n):
                    repair_alpha = False
                    success = True
                    break
                else:
                    # Find a maximal independent set to guide edge addition
                    # We can just pick two random non-adjacent vertices and try to add edge
                    # But we must ensure no triangle is created
                    # If we add edge (u, v), we need no common neighbor.
                    # So we can only add edge (u, v) if adj_masks[u] & adj_masks[v] == 0
                    
                    # Strategy: Pick a random vertex u, pick a random non-neighbor v of u.
                    # If adj_masks[u] & adj_masks[v] == 0, add edge.
                    # If not, skip.
                    
                    added = False
                    for _ in range(10): # Try a few times
                        u = random.randint(0, n-1)
                        # Get non-neighbors of u
                        # non_neighbors = ((1<<n)-1) & ~adj_masks[u] & ~(1<<u)
                        all_mask = (1 << n) - 1
                        non_neigh = all_mask & ~adj_masks[u] & ~(1 << u)
                        if non_neigh == 0:
                            continue
                        
                        # Pick random v from non_neigh
                        # v = (non_neigh & -non_neigh).bit_length() - 1
                        # Or random bit
                        # Let's just iterate
                        # To pick random bit:
                        # count = non_neigh.bit_count()
                        # idx = random.randint(0, count-1)
                        # ... too complex. Just pick lowest bit for now?
                        # Or iterate a few random indices
                        for _ in range(5):
                            v = random.randint(0, n-1)
                            if non_neigh & (1 << v):
                                # Check if adding (u,v) creates triangle
                                # i.e., if they have common neighbor
                                if (adj_masks[u] & adj_masks[v]) == 0:
                                    # Add edge
                                    adj_masks[u] |= (1 << v)
                                    adj_masks[v] |= (1 << u)
                                    added = True
                                    break
                            if added:
                                break
                        if added:
                            break
                    if not added:
                        # Couldn't add any edge, restart construction
                        break

        if success:
            # Build edge list
            edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj_masks[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
            
    # Fallback: Return a known valid small graph or the seed's fallback
    # Seed fallback was n=26
    n = 26
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (j-i) % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def has_independent_set_10(adj_masks, n):
    # Bitmask backtracking
    # adj_masks[i] is bitmask of neighbors
    
    def backtrack(candidates, size):
        if size == 10:
            return True
        if candidates == 0:
            return False
        
        # Upper bound: popcount
        if size + candidates.bit_count() < 10:
            return False
            
        # Heuristic: Pick vertex with highest degree in candidate subgraph
        # This is expensive. Let's just pick the lowest set bit for now.
        # Optimization: If we pick a vertex with high degree, the "include" branch
        # removes many candidates.
        
        # Let's try to pick a vertex.
        # To avoid deep recursion overhead, we can use a stack or just recursion.
        # Recursion depth max 10.
        
        # Pick lowest set bit
        lsb = candidates & -candidates
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New candidates: candidates not adjacent to v, and not v
        new_cand_incl = candidates & ~adj_masks[v] & ~(1 << v)
        if backtrack(new_cand_incl, size + 1):
            return True
            
        # Branch 2: Exclude v
        new_cand_excl = candidates & ~(1 << v)
        if backtrack(new_cand_excl, size):
            return True
            
        return False

    all_vertices = (1 << n) - 1
    return backtrack(all_vertices, 0)
