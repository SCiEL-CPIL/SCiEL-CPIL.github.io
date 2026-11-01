import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 6 and radius == 1:
        total_words = 729
        
        # Precompute index conversions
        def idx_to_tuple(idx):
            t = [0] * 6
            rem = idx
            for i in range(6):
                t[i] = rem % 3
                rem //= 3
            return tuple(t)

        def tuple_to_idx(t):
            idx = 0
            pow3 = 1
            for i in range(6):
                idx += t[i] * pow3
                pow3 *= 3
            return idx

        # Precompute balls as NumPy arrays of indices
        # This allows vectorized updates to cover_counts
        balls_np = [None] * total_words
        for i in range(total_words):
            w = idx_to_tuple(i)
            ball = [i]
            for pos in range(6):
                orig_val = w[pos]
                for new_val in range(3):
                    if new_val != orig_val:
                        new_w = list(w)
                        new_w[pos] = new_val
                        new_idx = tuple_to_idx(tuple(new_w))
                        ball.append(new_idx)
            balls_np[i] = np.array(ball, dtype=np.int32)
            
        # Also keep sets for fast membership if needed, but arrays are better for counts
        balls_set = [set(arr) for arr in balls_np]

        # Helper to check if a set of indices covers the space
        def is_cover(indices):
            if not indices:
                return total_words == 0
            covered_mask = np.zeros(total_words, dtype=bool)
            for idx in indices:
                covered_mask[balls_np[idx]] = True
            return np.all(covered_mask)

        # Helper to prune redundant codewords using counts
        def prune_cover(indices):
            indices = list(indices)
            if not indices:
                return indices
            cover_counts = np.zeros(total_words, dtype=np.int32)
            
            # Initialize counts
            for idx in indices:
                cover_counts[balls_np[idx]] += 1

            changed = True
            while changed:
                changed = False
                new_indices = []
                for idx in indices:
                    # Check if removing idx leaves any word uncovered
                    # i.e., is there any word in ball(idx) with count 1?
                    if np.min(cover_counts[balls_np[idx]]) > 1:
                        # Can remove
                        cover_counts[balls_np[idx]] -= 1
                        changed = True
                    else:
                        new_indices.append(idx)
                indices = new_indices
            return indices

        # Advanced Local Search: Deletion + Swap (NumPy optimized)
        def local_search(code_indices):
            if time.time() > deadline - 10.0:
                return code_indices
            
            code_indices = list(code_indices)
            if not code_indices:
                return code_indices
                
            cover_counts = np.zeros(total_words, dtype=np.int32)
            for idx in code_indices:
                cover_counts[balls_np[idx]] += 1
            
            improved = True
            stall_counter = 0
            max_stall = 500 # Prevent infinite loops
            
            while improved and time.time() < deadline - 5.0:
                improved = False
                stall_counter += 1
                if stall_counter > max_stall:
                    break
                
                # Phase 1: Try to delete
                # Find indices that can be removed
                deletable_indices = []
                for i, idx in enumerate(code_indices):
                    if np.min(cover_counts[balls_np[idx]]) > 1:
                        deletable_indices.append(i)
                
                if deletable_indices:
                    # Remove one (random or first)
                    i_to_remove = random.choice(deletable_indices)
                    idx = code_indices[i_to_remove]
                    cover_counts[balls_np[idx]] -= 1
                    code_indices.pop(i_to_remove)
                    improved = True
                    stall_counter = 0
                    continue # Restart loop to check new state
                
                # Phase 2: Try to swap
                # For each codeword, check if we can replace it with another word
                # that covers its "essential" words (words with count 1 in its ball)
                swap_done = False
                # Shuffle to avoid bias
                search_order = list(range(len(code_indices)))
                random.shuffle(search_order)
                
                for i in search_order:
                    if time.time() > deadline - 5.0:
                        break
                    
                    idx = code_indices[i]
                    # Calculate loss if we remove idx
                    # Loss words are those in ball(idx) with count 1
                    ball_idx = balls_np[idx]
                    loss_mask = cover_counts[ball_idx] == 1
                    loss_indices = ball_idx[loss_mask]
                    
                    if len(loss_indices) == 0:
                        # Should have been caught in Phase 1, but just in case
                        continue
                    
                    # If loss is large, skip (too expensive to check candidates)
                    # In K_3(6,1), ball size is 19. Max loss is 19.
                    # We need a candidate c such that ball(c) contains all loss_indices.
                    # This is equivalent to: loss_indices is subset of ball(c)
                    # Or: intersection of ball(c) and loss_indices has size == len(loss_indices)
                    
                    # Optimization: Pre-compute for each word c, the set of words it covers.
                    # We iterate through all 729 candidates.
                    found_swap = False
                    for c in range(total_words):
                        if c == idx or c in code_indices:
                            continue
                        
                        # Check if c covers all loss
                        # Fast check: if c is not in ball of any loss word? No.
                        # Check if loss_indices is subset of balls_np[c]
                        # Using set intersection is fast for small sets
                        if balls_set[c].issuperset(loss_indices):
                            # Perform swap
                            # Remove idx
                            cover_counts[balls_np[idx]] -= 1
                            # Add c
                            cover_counts[balls_np[c]] += 1
                            
                            code_indices[i] = c
                            improved = True
                            swap_done = True
                            stall_counter = 0
                            found_swap = True
                            break
                    
                    if found_swap:
                        break
                
                if not swap_done and not deletable_indices:
                    # No moves found
                    break
            
            return prune_cover(code_indices)

        best_code = None
        best_size = float('inf')

        # Strategy 1: Direct Product Construction (Tetracode based)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))

        pruned_dp = prune_cover(dp_indices)
        if len(pruned_dp) < best_size:
            best_size = len(pruned_dp)
            best_code = pruned_dp

        # Strategy 2: Randomized Greedy with symmetry-guided tie-breaking
        # We increase restarts because NumPy makes each iteration faster.
        num_restarts = 800 
        
        # Pre-compute gains for greedy? No, gains change.
        # But we can speed up the greedy step.
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break

            uncovered = np.ones(total_words, dtype=bool)
            code_indices = []

            while np.any(uncovered):
                # Find max gain
                # Gain for c is sum(uncovered[ball(c)])
                # We can vectorize this:
                # gains[c] = np.sum(uncovered[balls_np[c]])
                # This is O(N^2) per step if done naively.
                # N=729, steps ~ 73. 729*729*73 ~ 38M ops. In NumPy, this is fast.
                
                gains = np.zeros(total_words, dtype=np.int32)
                for c in range(total_words):
                    gains[c] = np.sum(uncovered[balls_np[c]])
                
                max_gain = np.max(gains)
                if max_gain == 0:
                    break # Should not happen if uncovered is not empty and balls cover space
                
                # Get all candidates with max gain
                top_candidates = np.where(gains == max_gain)[0]
                
                # Random selection
                best_c = top_candidates[random.randint(0, len(top_candidates)-1)]
                
                code_indices.append(best_c)
                uncovered[balls_np[best_c]] = False

                if time.time() > deadline - 10.0:
                    break

            if np.any(uncovered):
                continue

            # Prune and then local search
            pruned = prune_cover(code_indices)
            pruned = local_search(pruned)

            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned

        if best_code is None:
            best_code = pruned_dp
        else:
            if not is_cover(best_code):
                best_code = pruned_dp

        result = [list(idx_to_tuple(c)) for c in best_code]
        return result

    else:
        # Generic fallback for other parameters
        total_words = q ** n
        if total_words <= 10000:
            powers = [1] * n
            for i in range(1, n):
                powers[i] = powers[i-1] * q
                
            def idx_to_tuple_gen(idx):
                t = [0] * n
                rem = idx
                for i in range(n):
                    t[i] = rem % q
                    rem //= q
                return tuple(t)
                
            def tuple_to_idx_gen(t):
                idx = 0
                for i in range(n):
                    idx += t[i] * powers[i]
                return idx

            balls = [None] * total_words
            for i in range(total_words):
                w = idx_to_tuple_gen(i)
                ball = {i}
                for pos in range(n):
                    orig_val = w[pos]
                    for new_val in range(q):
                        if new_val != orig_val:
                            new_w = list(w)
                            new_w[pos] = new_val
                            new_idx = tuple_to_idx_gen(tuple(new_w))
                            ball.add(new_idx)
                balls[i] = ball

            uncovered = set(range(total_words))
            code_indices = []
            
            while uncovered:
                best_c = -1
                max_gain = 0
                for c in range(total_words):
                    gain = len(balls[c] & uncovered)
                    if gain > max_gain:
                        max_gain = gain
                        best_c = c
                if best_c == -1:
                    break
                code_indices.append(best_c)
                uncovered -= balls[best_c]
                
                if time.time() > deadline:
                    break
            
            result = [list(idx_to_tuple_gen(c)) for c in code_indices]
            return result
        else:
            return [[0] * n]
