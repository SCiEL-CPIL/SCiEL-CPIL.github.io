import itertools
import numpy as np
import random
import time
import sys

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as numpy arrays for fast operations
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(np.array(list(neigh_set), dtype=int))
    
    # Precompute neighbor sets as Python sets for fast intersection
    neighbor_sets = [set(neigh.tolist()) for neigh in neighbors]
    
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    def reduce_code(code, counts):
        """
        Reduces code by removing redundant words.
        Returns new_code and new_counts.
        """
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    # Remove c
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    # 1. Fast Greedy Initialization with weight matrix for O(1) gain queries
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0
    
    # Build weight matrix: weight[i] = number of uncovered points in neighbors[i]
    # Using sparse representation: for each point j, track which codewords cover it
    # Actually, let's use a different approach: maintain a "gain" array
    # gain[i] = number of currently-uncovered points that i would cover
    
    # Precompute: for each point j, which codewords cover it (inverse neighbors)
    # This is expensive, so instead we'll use a smarter greedy:
    # Maintain uncovered as boolean array, and for each candidate i, 
    # compute gain = sum(uncovered[neighbors[i]])
    # But this is O(N*17) per iteration which is too slow.
    
    # Alternative: use the fact that neighbors are small (size 17)
    # For each position in neighbors[i], we can quickly check if uncovered
    # Let's try a block-based approach
    
    # Actually, let's use a simpler optimization:
    # Process in batches and use numpy vectorization more effectively
    # Precompute neighbor arrays as a 2D structure if possible
    
    # For now, use the existing loop but with early termination and better ordering
    # Key insight: after selecting a codeword, many other candidates become less valuable
    # We can skip candidates whose max possible gain is already <= best_count
    
    # Let's use a priority-queue-like approach with numpy
    # Initialize gains
    gains = np.array([np.sum(uncovered[neighbors[i]]) for i in range(N)], dtype=int)
    
    while np.any(uncovered):
        if time.time() - start_time > time_limit:
            break
        
        best_idx = int(np.argmax(gains))
        best_count = gains[best_idx]
        if best_count == 0:
            break
        
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
        # Update gains for all candidates
        # A candidate i's gain decreases by the number of points in neighbors[i] 
        # that were just covered by best_idx
        # = |neighbors[i] ∩ neighbors[best_idx]|
        # Since neighbors are small, we can compute this efficiently
        
        # For each candidate i, new_gain[i] = old_gain[i] - |neighbors[i] ∩ neighbors[best_idx]|
        # But we only need to update candidates that actually lost uncovered points
        # The points lost are neighbors[best_idx] (the ones that were uncovered)
        # Actually, all points in neighbors[best_idx] that were uncovered are now covered
        # So for any candidate i, its gain decreases by |neighbors[i] ∩ neighbors[best_idx] ∩ {was_uncovered}|
        # Since we just marked neighbors[best_idx] as covered, the intersection with was_uncovered
        # is exactly neighbors[i] ∩ neighbors[best_idx] (all of which were uncovered before this step)
        # Wait, no: some of neighbors[best_idx] might have already been covered by previous selections
        # So we need: for each i, gain[i] -= number of points in neighbors[i] that are in neighbors[best_idx] AND were uncovered before this step
        
        # This is getting complex. Let's just recompute gains periodically or use a simpler update.
        # For now, recompute gains every few iterations for correctness, or use a conservative update.
        
        # Conservative update: for each i, if neighbors[i] shares any point with neighbors[best_idx],
        # then gain[i] might decrease. But computing the exact decrease requires knowing
        # which specific points were uncovered.
        
        # Simpler: just recompute gains from scratch every K iterations, or use the fact that
        # most candidates are unaffected. Let's recompute when needed.
        
        # Actually, the original approach of scanning all candidates each time is O(N*17) = ~111k ops
        # With ~400-500 iterations, that's ~50M ops, which should be fine in numpy.
        # Let's just recompute gains each iteration for correctness.
        
        gains = np.array([np.sum(uncovered[neighbors[i]]) for i in range(N)], dtype=int)
            
    # 2. Local Search
    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Local Search Loop
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        # 2. Swap Pass
        if len(C) == old_size:
            # No reduction happened. Try to find a move.
            improved = False
            # Randomize the order of C to avoid bias
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                c = C[idx]
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    # Should have been removed, but maybe logic missed it?
                    # Force removal
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                # If too many unique points, unlikely to find a single replacement
                if len(unique_points) > 3:
                    continue
                
                # Find candidates c' that cover all unique_points
                # Intersection of neighbors[u] for u in unique_points
                if len(unique_points) == 1:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                else:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                    for u in unique_points[1:]:
                        candidates_set &= neighbor_sets[u]
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                
                # Filter candidates: must not be in C
                candidates_set.difference_update(C)
                
                if not candidates_set:
                    continue
                    
                # Pick a candidate
                c_prime = random.choice(list(candidates_set))
                
                # Apply swap
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                # Check validity
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue
                
                # Try to reduce
                new_C, new_counts = reduce_code(C, counts)
                
                if len(new_C) < len(old_C):
                    # Improvement!
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    improved = True
                    break # Restart loop
                    
                else:
                    # No reduction.
                    # Keep the swap if it's a "neutral" move that changes the structure.
                    improved = True
                    break # Restart loop to see if this new state allows reduction
            
            if not improved:
                # No move found in this pass.
                # Try a few random kicks.
                for _ in range(10):
                    if time.time() - start_time > time_limit: break
                    if not C: break
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand)-1)]
                    if c_prime in C: continue
                    
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(c_prime)
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                        continue
                        
                    new_C, new_counts = reduce_code(C, counts)
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        break
                    else:
                        C = new_C
                        counts = new_counts

    # Final Reduction with aggressive polish
    best_C, best_counts = reduce_code(best_C, get_counts(best_C))
    
    # Additional polish: try multiple rounds of reduction + targeted swaps
    polish_start = time.time()
    polish_deadline = start_time + time_limit
    
    for polish_round in range(20):
        if time.time() > polish_deadline - 5:  # Leave 5s safety margin
            break
        
        # Try reduction
        old_size = len(best_C)
        best_C, best_counts = reduce_code(best_C, best_counts)
        
        if len(best_C) < old_size:
            continue  # Keep polishing if we made progress
        
        # If no reduction, try targeted swaps to enable future reductions
        improved = False
        for _ in range(50):
            if time.time() > polish_deadline - 5:
                break
            if not best_C:
                break
            
            c_idx = random.randint(0, len(best_C) - 1)
            c = best_C[c_idx]
            
            # Find points uniquely covered by c
            u_indices = neighbors[c]
            unique_mask = best_counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                # Should have been reduced, but try removing anyway
                best_C.remove(c)
                best_counts[neighbors[c]] -= 1
                improved = True
                break
            
            if len(unique_points) > 4:
                continue
            
            # Find candidates that cover all unique points
            if len(unique_points) == 1:
                candidates_set = set(neighbor_sets[unique_points[0]])
            else:
                candidates_set = set(neighbor_sets[unique_points[0]])
                for u in unique_points[1:]:
                    candidates_set &= neighbor_sets[u]
                    if not candidates_set:
                        break
            
            if not candidates_set:
                continue
            
            candidates_set.difference_update(best_C)
            if not candidates_set:
                continue
            
            c_prime = random.choice(list(candidates_set))
            
            # Try the swap
            old_C = best_C.copy()
            old_counts = best_counts.copy()
            
            best_C.remove(c)
            best_C.append(c_prime)
            best_counts[neighbors[c]] -= 1
            best_counts[neighbors[c_prime]] += 1
            
            if np.any(best_counts < 1):
                best_C = old_C
                best_counts = old_counts
                continue
            
            # Try to reduce after swap
            new_C, new_counts = reduce_code(best_C, best_counts)
            
            if len(new_C) < len(old_C):
                best_C = new_C
                best_counts = new_counts
                improved = True
                break
            else:
                # Accept neutral move if it changes structure
                best_C = new_C
                best_counts = new_counts
                improved = True
                break
        
        if not improved:
            break
    
    # Final reduction pass
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
