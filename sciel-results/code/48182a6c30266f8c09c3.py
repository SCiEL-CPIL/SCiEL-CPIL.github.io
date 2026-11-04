"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def construct(q, n, radius):
    """
    Greedy set-covering with simulated annealing refinement for ternary covering code.
    """
    start_time = time.time()
    time_limit = 115  # leave some buffer before 120s timeout
    
    # Total number of words in the space
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = [tuple(word) for word in np.ndindex(*([q] * n))]
    all_words_set = set(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls as sets of indices for faster coverage updates
    # ball_indices[i] is the set of indices of words covered by center i
    ball_indices = [None] * total_words
    for i, w in enumerate(all_words):
        ball = set()
        # Generate ball of radius r
        # For q=3, n=6, radius=1, size is 1 + 6*2 = 13
        # We can iterate over all words and check distance, but that's 729*729 ~ 500k ops, which is fine.
        # Or generate neighbors directly.
        # Direct generation is faster.
        ball.add(i)
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                new_word = list(w)
                new_word[pos] = new_val
                new_tuple = tuple(new_word)
                ball.add(word_to_idx[new_tuple])
        ball_indices[i] = ball

    # 1. Greedy Set Cover Initialization
    uncovered = set(range(total_words))
    centers = []
    
    # To allow for some randomness in initialization, we can shuffle the order we consider centers
    # or just do standard greedy. Standard greedy is strong.
    # Let's do standard greedy first.
    while uncovered:
        if time.time() - start_time > time_limit * 0.3:
            break
        
        best_center_idx = -1
        best_coverage = -1
        best_newly_covered = None
        
        # We only need to check centers that cover at least one uncovered word
        # In a full scan:
        for c_idx in range(total_words):
            if not uncovered:
                break
            newly_covered = ball_indices[c_idx] & uncovered
            cov = len(newly_covered)
            if cov > best_coverage:
                best_coverage = cov
                best_center_idx = c_idx
                best_newly_covered = newly_covered
        
        if best_center_idx == -1 or best_coverage <= 0:
            # Should not happen if uncovered is not empty
            break
            
        centers.append(best_center_idx)
        uncovered -= best_newly_covered
    
    # If greedy didn't finish (timeout or logic error), force cover
    if uncovered:
        for w_idx in uncovered:
            centers.append(w_idx)
            uncovered -= ball_indices[w_idx]
    
    # Convert centers to list of indices
    current_centers = list(centers)
    best_centers = list(current_centers)
    best_count = len(current_centers)
    
    # Track coverage count for each word
    # coverage_count[i] = number of centers covering word i
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for c_idx in current_centers:
        for w_idx in ball_indices[c_idx]:
            coverage_count[w_idx] += 1
    
    # 2. Simulated Annealing
    # Cost function: count + penalty * uncovered
    # We want to minimize count.
    # Move types:
    # - Remove a center: If it doesn't uncover any words, accept (count decreases).
    #   If it uncovers k words, cost increases by -1 + k*penalty.
    # - Swap a center: Replace c with w. Count same. Uncovered changes.
    
    penalty = 100  # Large penalty for uncovered words to ensure validity is prioritized in acceptance
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    random.seed(42)
    
    while T > T_min and time.time() - start_time < time_limit:
        if not current_centers:
            break
            
        move_type = random.random()
        
        if move_type < 0.5 and len(current_centers) > 1:
            # Try Remove
            idx = random.randint(0, len(current_centers) - 1)
            c_idx = current_centers[idx]
            
            # Check uncovered if we remove c_idx
            # Words w where coverage_count[w] == 1 and w in ball_indices[c_idx]
            newly_uncovered = 0
            for w_idx in ball_indices[c_idx]:
                if coverage_count[w_idx] == 1:
                    newly_uncovered += 1
            
            # Delta cost: -1 (count) + newly_uncovered * penalty
            delta_cost = -1 + newly_uncovered * penalty
            
            if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                # Update coverage
                for w_idx in ball_indices[c_idx]:
                    coverage_count[w_idx] -= 1
                current_centers.pop(idx)
                
                if newly_uncovered == 0:
                    # Valid solution, update best
                    if len(current_centers) < best_count:
                        best_count = len(current_centers)
                        best_centers = list(current_centers)
                        
        elif move_type < 0.9:
            # Try Swap
            if len(current_centers) < 1:
                continue
            idx = random.randint(0, len(current_centers) - 1)
            old_c_idx = current_centers[idx]
            new_c_idx = random.randint(0, total_words - 1)
            
            if old_c_idx == new_c_idx:
                continue
                
            # Calculate change in uncovered
            # Words that were covered ONLY by old_c_idx and NOT by new_c_idx become uncovered
            # Words that were uncovered and covered by new_c_idx become covered
            
            # This is slightly complex to compute exactly without iterating.
            # Let's iterate over the union of balls?
            # Ball size is small (~13).
            
            old_ball = ball_indices[old_c_idx]
            new_ball = ball_indices[new_c_idx]
            
            # Words in old_ball but not new_ball: lose coverage from old.
            # If coverage_count[w] == 1, it becomes uncovered.
            # Words in new_ball but not old_ball: gain coverage from new.
            # If coverage_count[w] == 0, it becomes covered.
            
            delta_uncovered = 0
            # Check words losing coverage
            for w_idx in old_ball:
                if w_idx not in new_ball:
                    if coverage_count[w_idx] == 1:
                        delta_uncovered += 1 # Becomes uncovered
            # Check words gaining coverage
            for w_idx in new_ball:
                if w_idx not in old_ball:
                    if coverage_count[w_idx] == 0:
                        delta_uncovered -= 1 # Becomes covered (good)
            
            # Delta cost: 0 (count same) + delta_uncovered * penalty
            delta_cost = delta_uncovered * penalty
            
            if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                # Update coverage
                for w_idx in old_ball:
                    if w_idx not in new_ball:
                        coverage_count[w_idx] -= 1
                for w_idx in new_ball:
                    if w_idx not in old_ball:
                        coverage_count[w_idx] += 1
                current_centers[idx] = new_c_idx
                
                # Check if valid and better
                # We can quickly check if any uncovered exists
                # Instead of full check, we can track a global uncovered count?
                # Let's just do a quick check for best update only if delta_cost <= 0 (improvement or neutral)
                # Actually, if delta_uncovered == 0, count is same, so no improvement in count.
                # Swap only helps if it enables future removals.
                # So we don't update best_centers on swap unless we also remove something.
                
        else:
            # Try Add (rarely useful alone, but might help escape)
            # Or just do nothing / small probability
            pass
            
        T *= cooling_rate
        
    # Final verification and return
    # Ensure best_centers is valid
    # Re-verify best_centers
    covered_set = set()
    for c_idx in best_centers:
        covered_set |= ball_indices[c_idx]
        
    if len(covered_set) == total_words:
        final_indices = best_centers
    else:
        # Fallback to the greedy result which was valid
        final_indices = [c for c in centers]
        
    # Convert indices back to lists of integers
    return [list(all_words[i]) for i in final_indices]
# EVOLVE-BLOCK-END
