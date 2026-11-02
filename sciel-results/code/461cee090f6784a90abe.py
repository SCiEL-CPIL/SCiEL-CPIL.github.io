"""Football pool K_3(7,1): minimize codewords covering all of {0,1,2}^7 within Hamming distance 1."""
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

    if q != 3 or n != 7 or radius != 1:
        return _direct_sum(q, n, radius)

    N = 3 ** 7  # 2187
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute balls as lists of indices
    balls = []
    for idx, w in enumerate(all_words):
        neighbors = [idx]
        for pos in range(7):
            orig = w[pos]
            for val in range(3):
                if val == orig:
                    continue
                neighbor = list(w)
                neighbor[pos] = val
                nidx = word_to_idx[tuple(neighbor)]
                neighbors.append(nidx)
        balls.append(np.array(neighbors, dtype=np.int32))

    # Precompute balls_matrix for vectorized greedy
    # balls_matrix[i, j] is True if j is in ball(i)
    balls_matrix = np.zeros((N, N), dtype=bool)
    for i in range(N):
        balls_matrix[i, balls[i]] = True

    # Phase 1: Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        # Calculate gain for each candidate
        # gains[i] = number of currently uncovered elements in ball(i)
        gains = balls_matrix @ uncovered
        # Pick the one with max gain
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break # Should not happen if uncovered is not empty and balls cover everything
        selected.append(best_idx)
        uncovered[balls[best_idx]] = False

    # Phase 2: Local Search with Count-based updates
    # counts[j] = number of selected words that cover j
    counts = np.zeros(N, dtype=int)
    for idx in selected:
        counts[balls[idx]] += 1

    best_selected = list(selected)
    best_count = len(selected)

    # Helper to check if a list of indices is a valid cover
    def is_valid(sel_list):
        c = np.zeros(N, dtype=int)
        for idx in sel_list:
            c[balls[idx]] += 1
        return np.all(c > 0)

    # Local Search Loop
    random.seed(42)
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # 1. Try to remove redundant words
        removed_any = False
        for i in range(len(selected) - 1, -1, -1):
            idx = selected[i]
            # Check if removing idx leaves any element with count 0
            # temp_counts = counts - balls_matrix[idx]
            # if np.any(temp_counts == 0): continue
            # else: remove
            
            # Vectorized check:
            # We need to know if counts[j] > 1 for all j in balls[idx]
            # If yes, idx is redundant.
            if np.all(counts[balls[idx]] > 1):
                counts[balls[idx]] -= 1
                selected.pop(i)
                removed_any = True
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_selected = list(selected)
        
        if not removed_any:
            # 2. Try a swap
            if len(selected) == 0:
                break
            
            # Pick a random word to remove
            i = random.randint(0, len(selected) - 1)
            idx_to_remove = selected[i]
            
            # Calculate deficit if we remove it
            # deficit[j] = 1 if counts[j] == 1 and j is in balls[idx_to_remove]
            # Actually, if we remove idx, new_counts = counts - balls_matrix[idx]
            # We need to find v such that new_counts + balls_matrix[v] > 0 everywhere.
            # This is equivalent to balls_matrix[v] covering all j where new_counts[j] == 0.
            
            new_counts = counts - balls_matrix[idx_to_remove]
            missing_indices = np.where(new_counts == 0)[0]
            
            if len(missing_indices) == 0:
                # We can just remove it (should have been caught by redundancy check, 
                # but maybe counts changed? No, counts is consistent.)
                # This case is handled above.
                pass
            else:
                # Find v that covers all missing_indices
                # covers_deficit = balls_matrix[:, missing_indices]
                # valid_v = np.all(covers_deficit, axis=1)
                
                # To speed up, we can check candidates.
                # But checking all N is 2187 * len(missing) ops.
                # len(missing) is small (size of ball of idx, max 19).
                # So 2187 * 19 ~ 40k ops. Very fast.
                
                # We want to pick a v not in selected.
                # And ideally one that minimizes "waste" or allows further removal.
                
                # Let's just find any valid v.
                # valid_v_mask = np.all(balls_matrix[:, missing_indices], axis=1)
                # valid_v_mask[selected] = False # Exclude already selected
                
                # For efficiency, we can just pick a random subset of candidates or all.
                # Let's compute valid_v_mask
                valid_v_mask = np.all(balls_matrix[:, missing_indices], axis=1)
                
                # Exclude selected
                for sel_idx in selected:
                    valid_v_mask[sel_idx] = False
                
                valid_indices = np.where(valid_v_mask)[0]
                
                if len(valid_indices) > 0:
                    # Pick one randomly
                    new_idx = valid_indices[random.randint(0, len(valid_indices) - 1)]
                    
                    # Perform swap
                    counts[balls[idx_to_remove]] -= 1
                    counts[balls[new_idx]] += 1
                    selected[i] = new_idx
                    
                    # After swap, try to remove redundancies again immediately
                    # (Loop back to step 1)
                    continue # Restart the redundancy check phase

        # If we get here, no removal and no valid swap found in this iteration?
        # We can try a random restart or perturbation if stuck, but for now, 
        # the swap logic above attempts to find a valid swap.
        # If no valid swap exists, we are at a local minimum for swap moves.
        
    # Final verification
    final_counts = np.zeros(N, dtype=int)
    for idx in best_selected:
        final_counts[balls[idx]] += 1
    if not np.all(final_counts > 0):
        # Fallback to initial greedy or direct sum if something went wrong
        best_selected = list(range(len(all_words))) # Worst case, but valid
        
    return [list(all_words[idx]) for idx in sorted(best_selected)]


def _direct_sum(q, n, radius):
    """Fallback for non-standard parameters."""
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    return [list(w) for w in itertools.product(range(q), repeat=n)]
