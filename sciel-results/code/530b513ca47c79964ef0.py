import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin for 90s limit
    
    P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}
    
    n = P['n']
    edges = P['edges']
    
    # Build neighbor masks
    # mask[v] is an integer with bit u set if (v,u) is an edge
    masks = [0] * n
    for u, v in edges:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        
    best_score = 0
    best_subset = list(range(8)) # Fallback, though we will find better
    
    # Iterate k from 17 down to 8
    # This allows pruning: if best_score >= k, we can skip this k? 
    # No, max distinct degrees is k. If best_score is 10, and k=9, max is 9 < 10. Skip.
    # So if best_score >= k, we can break if we are going down?
    # If we iterate 17, 16, ..., 8.
    # If at k=17 we find score 12.
    # Next k=16. Max possible 16. 16 > 12, so must check.
    # If at k=12 we find score 12.
    # Next k=11. Max possible 11. 11 < 12. Skip.
    # So we can break the loop if k <= best_score.
    
    for k in range(n, 7, -1):
        if k <= best_score:
            break
            
        if time.time() > deadline:
            break
            
        # Generate all subsets of size k
        # Using itertools.combinations
        for subset in itertools.combinations(range(n), k):
            if time.time() > deadline:
                break
                
            # Convert subset to bitmask
            mask = 0
            for v in subset:
                mask |= (1 << v)
                
            # Calculate degrees
            degrees = []
            for v in subset:
                # Count neighbors in mask
                # popcount of (mask & masks[v])
                # In Python 3.10+, int.bit_count() is available
                deg = (mask & masks[v]).bit_count()
                degrees.append(deg)
                
            distinct_degrees = len(set(degrees))
            
            if distinct_degrees > best_score:
                best_score = distinct_degrees
                best_subset = list(subset)
                
                # Early termination if we hit theoretical max for this k?
                # If best_score == k, we can't do better for this k.
                # But we might do better for larger k? No, we are going down.
                # So if best_score == k, we can break the inner loop for this k?
                # Yes, because we can't exceed k for size k.
                # But we still need to check smaller k?
                # If best_score == k, then for next k' < k, max is k' < best_score.
                # So we can break the outer loop too.
                if best_score == k:
                    break
                    
        # Check deadline again
        if time.time() > deadline:
            break

    return best_subset
