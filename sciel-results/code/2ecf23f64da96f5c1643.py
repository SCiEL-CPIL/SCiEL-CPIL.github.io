import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    total_points = q ** n
    points = np.array(list(itertools.product(range(q), repeat=n)))
    
    # Compute distances
    # Using broadcasting to create a (N, N, n) boolean array
    diff = points[:, np.newaxis, :] != points[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    in_ball = dists <= radius
    
    # Precompute ball indices and masks
    # ball_indices[i] is a list of indices of points within distance radius of i
    # ball_masks[i] is an integer bitmask of the same
    ball_indices = []
    ball_masks = []
    for i in range(total_points):
        idx = np.nonzero(in_ball[i])[0]
        ball_indices.append(idx)
        mask = 0
        for j in idx:
            mask |= (1 << j)
        ball_masks.append(mask)
        
    def greedy_cover(randomize_ties=False):
        uncovered = (1 << total_points) - 1
        selected = []
        selected_set = set()
        
        while uncovered:
            best_count = -1
            best_candidates = []
            
            for i in range(total_points):
                if i in selected_set:
                    continue
                # Count new coverage
                new_cov = (uncovered & ball_masks[i]).bit_count()
                if new_cov > best_count:
                    best_count = new_cov
                    best_candidates = [i]
                elif new_cov == best_count and new_cov > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
                
            if randomize_ties and len(best_candidates) > 1:
                best_i = random.choice(best_candidates)
            else:
                best_i = best_candidates[0]
                
            selected.append(best_i)
            selected_set.add(best_i)
            uncovered &= ~ball_masks[best_i]
            
        return selected

    def local_search_remove(selected):
        # Use cover counts for efficiency
        cover_counts = np.zeros(total_points, dtype=np.int32)
        
        # Initialize
        for c in selected:
            for j in ball_indices[c]:
                cover_counts[j] += 1
                
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is redundant
                # Decrement counts for points in c's ball
                redundant = True
                for j in ball_indices[c]:
                    cover_counts[j] -= 1
                    if cover_counts[j] == 0:
                        redundant = False
                        break
                
                if redundant:
                    changed = True
                    # Do not add c
                else:
                    # Restore counts
                    for j in ball_indices[c]:
                        cover_counts[j] += 1
                    new_selected.append(c)
                    
            selected = new_selected
        return selected

    best_code = None
    start_time = time.time()
    deadline = 145.0
    
    # Run initial deterministic greedy
    code = greedy_cover(randomize_ties=False)
    code = local_search_remove(code)
    if best_code is None or len(code) < len(best_code):
        best_code = code
        
    # Run randomized restarts
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        code = greedy_cover(randomize_ties=True)
        code = local_search_remove(code)
        if len(code) < len(best_code):
            best_code = code
            if len(best_code) == 73: # Known optimum for this instance
                break

    # Convert indices back to codewords
    return [[int(x) for x in points[i]] for i in best_code]
