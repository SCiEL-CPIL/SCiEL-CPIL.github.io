import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def has_k4(n, adj):
        """Check if graph has K4 using bitmasks."""
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    # Check if any edge exists in common
                    c = common
                    while c:
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        # If k is connected to any other node in common, we have K4
                        # adj[k] & common contains neighbors of k in common
                        # If this is not 0, there is an edge
                        if (adj[k] & common) & ~lsb: # ~lsb removes k itself if it was there? 
                            # adj[k] does not contain k. So adj[k] & common is just neighbors of k in common.
                            # If this is non-zero, there is an edge (k, x) where x in common.
                            # Then i, j, k, x form K4.
                            return True
        return False

    def has_ind6(n, adj):
        """Check if graph has independent set of size 6."""
        all_mask = (1 << n) - 1
        # Precompute non-neighbor masks
        nn = [0] * n
        for i in range(n):
            nn[i] = (~adj[i]) & all_mask
            nn[i] &= ~(1 << i)
            
        # Recursive search for independent set of size 6
        # Use lru_cache? No, mask changes.
        # Simple recursion with pruning
        
        def search(mask, depth):
            if depth == 0:
                return True
            if mask == 0:
                return False
            if mask.bit_count() < depth:
                return False
                
            # Pick vertex with minimum non-neighbors in mask (most constrained)
            best_v = -1
            min_cnt = 100
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                cnt = (nn[v] & mask).bit_count()
                if cnt < min_cnt:
                    min_cnt = cnt
                    best_v = v
                    
            if best_v == -1:
                return False
            
            v = best_v
            # Branch 1: Include v
            if search(mask & nn[v], depth - 1):
                return True
            # Branch 2: Exclude v
            if search(mask & ~(1 << v), depth - 1):
                return True
            return False
            
        return search(all_mask, 6)

    def is_valid(n, adj):
        if has_k4(n, adj):
            return False
        if has_ind6(n, adj):
            return False
        return True

    def edges_from_masks(n, adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    best_n = 1
    best_adj = [0]
    
    # Try to find valid graphs for n from 35 down to 20
    for target_n in range(35, 19, -1):
        if time.time() > deadline - 10.0:
            break
            
        found = False
        # Try multiple restarts
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize random graph
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < 0.4:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Local Search (Simulated Annealing)
            # Objective: Minimize violations. 
            # Since we don't have a fast violation counter for Ind6, 
            # we will just try to reach a valid state.
            
            # Strategy: 
            # 1. Check if valid.
            # 2. If not, pick a random edge to flip.
            # 3. If new state is valid, accept.
            # 4. Else, accept with probability based on "improvement" in K4 count?
            #    Let's just do random walk with occasional backtracking.
            
            # Better: Use a heuristic score = K4_count + Ind5_count?
            # Let's just do random flips and check validity.
            
            current_adj = adj[:]
            for step in range(5000):
                if time.time() > deadline - 2.0:
                    break
                    
                if is_valid(target_n, current_adj):
                    found = True
                    best_n = target_n
                    best_adj = current_adj
                    break
                
                # Pick random edge to flip
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                
                old_state = (current_adj[u] >> v) & 1
                # Flip
                if old_state:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
                # Check validity
                if is_valid(target_n, current_adj):
                    found = True
                    best_n = target_n
                    best_adj = current_adj
                    break
                else:
                    # Revert? Or keep?
                    # If we keep, we might drift.
                    # Let's revert 50% of the time to stay close to random?
                    # Or just revert always to do a local search around valid states?
                    # Since we start random, we are far from valid.
                    # We need to move towards valid.
                    # Without a gradient, random walk is slow.
                    
                    # Let's try to reduce K4s.
                    # Calculate K4 count before and after?
                    # That's expensive.
                    
                    # Alternative: Just revert and try another edge.
                    if old_state:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                    else:
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
                        
            if found:
                break
                
        if found:
            break
            
    edges = edges_from_masks(best_n, best_adj)
    return {'n': best_n, 'edges': edges}
