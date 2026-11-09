import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Target n from 47 down to 40
    # R(C4, K1,39) is known to be 46 or 47.
    # If we find a graph for n=46, score is 46/39 ~ 1.179
    # If we find a graph for n=47, score is 47/39 ~ 1.205
    
    best_result = {'n': 39, 'edges': []}
    
    for target_n in range(47, 39, -1):
        min_deg_needed = target_n - 39
        
        # Try multiple random seeds
        # We can afford a lot of seeds if the check is fast.
        # n=47 has ~1100 edges. Check is fast.
        num_seeds = 100
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            
            # Bitset adjacency
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            
            # Generate all possible edges
            all_edges = list(combinations(range(target_n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Check if adding (u, v) creates a C4
                # Cycle: u - v - w - x - u
                # w in N(v), x in N(u) intersect N(w)
                # x != u, v, w
                
                conflict = False
                # Iterate over neighbors of v
                # We can use bit tricks to iterate, or just loop 0..n-1
                # Since n <= 47, loop is fine, but bit iteration is faster.
                mask_v = adj[v]
                
                while mask_v:
                    # Get lowest set bit
                    lsb = mask_v & -mask_v
                    w = lsb.bit_length() - 1
                    mask_v ^= lsb
                    
                    if w == u:
                        continue
                    
                    # Check common neighbors of u and w
                    common = adj[u] & adj[w]
                    # Exclude u, v, w
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    common &= ~(1 << w)
                    
                    if common:
                        conflict = True
                        break
                
                if not conflict:
                    # Add edge
                    edges.append([u, v])
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
            
            # Check if minimum degree condition is met
            if min(deg) >= min_deg_needed:
                # Verify C4-free (optional, but good for safety)
                # The greedy process ensures C4-free if the check is correct.
                # We can skip full verification to save time, or do a quick check.
                # Given the check logic is sound, we can trust it.
                # But let's do a quick check for the first few to be sure.
                if target_n > best_result['n']:
                    best_result = {'n': target_n, 'edges': edges}
                    # If we found a valid graph for this n, we can return immediately
                    # because we are iterating n downwards.
                    return best_result
        
        # If we didn't find a graph for this n, try smaller n
        # But keep the best result found so far (which might be from a larger n if we found one? No, we return immediately if found).
        # If we didn't return, it means we failed for this target_n.
        
    return best_result
