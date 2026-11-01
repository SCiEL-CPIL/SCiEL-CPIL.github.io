import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        # Direct sum with tetracode (Radius 2 for n>4, but valid format)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (invalid for score but valid format)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Weights for positions
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Precompute balls for all words
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy instances with different seeds
    seed = 0
    while time.time() < deadline - 5.0:
        code, size = _run_greedy_and_local_search(balls, total, seed, deadline - 2.0)
        
        if size < best_size:
            best_size = size
            best_code = code
            
            if size == 486:
                break
                
        seed += 1
        
    if best_code is None:
        return [list(c) for c in itertools.product(range(q), repeat=n)]
        
    # Convert indices to digit lists
    result = []
    for w in best_code:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result

def _run_greedy_and_local_search(balls, total, seed, sub_deadline):
    rng = np.random.default_rng(seed)
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    # 1. Greedy Phase
    while np.any(current_uncovered):
        if time.time() > sub_deadline:
            break
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[rng.integers(len(candidates))]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
        
    # 2. Pruning Phase
    # Remove any codeword that is redundant
    changed = True
    while changed and time.time() < sub_deadline:
        changed = False
        if not selected:
            break
        sel_arr = np.array(selected, dtype=np.int32)
        sel_balls = balls[sel_arr]
        counts_in_balls = cover_count[sel_balls]
        min_counts = np.min(counts_in_balls, axis=1)
        redundant_mask = min_counts > 1
        
        if np.any(redundant_mask):
            redundant_indices = sel_arr[redundant_mask]
            cover_count[balls[redundant_indices]] -= 1
            keep_mask = ~redundant_mask
            selected = sel_arr[keep_mask].tolist()
            changed = True
            
    # 3. Local Search Phase
    # We try to improve the solution by:
    # a) Deleting a codeword if possible.
    # b) Swapping a codeword with another if it reduces the "slack" (sum of cover counts).
    
    # Precompute set of selected for fast lookup
    selected_set = set(selected)
    # Calculate initial slack
    current_slack = np.sum(cover_count)
    
    # We will run local search until no improvement or time runs out
    while time.time() < sub_deadline:
        improved = False
        
        # Try to delete
        if selected:
            # Sample a few random candidates for deletion to save time, or iterate all if small
            # For ~490 codewords, iterating all is fast enough (490 * 17 ops)
            for i in range(len(selected)):
                c_out = selected[i]
                # Check if c_out can be removed
                # min(cover_count[balls[c_out]]) > 1
                if np.min(cover_count[balls[c_out]]) > 1:
                    # Remove it
                    cover_count[balls[c_out]] -= 1
                    selected.pop(i)
                    selected_set.remove(c_out)
                    current_slack -= 17 # Approx, actually sum of counts in ball
                    # Recalculate slack accurately? 
                    # Slack change is -sum(cover_count[balls[c_out]]) before decrement?
                    # Let's just recalculate slack periodically or use a proxy.
                    # Actually, slack = sum(cover_count).
                    # Decrementing cover_count[x] by 1 for x in ball reduces slack by 1 for each x.
                    # So slack reduces by ball_size? No, slack is sum of ALL cover counts.
                    # Removing c reduces cover_count for all w in ball(c) by 1.
                    # So total slack reduces by ball_size.
                    current_slack -= ball_size
                    improved = True
                    # After deletion, we can continue checking other deletions or break to restart loop
                    break
        
        if improved:
            continue
            
        # Try to swap
        # Pick a random c_out from selected
        # Pick a random c_in from NOT selected
        # Check if swap is valid and improves slack
        
        if selected:
            c_out_idx = rng.integers(len(selected))
            c_out = selected[c_out_idx]
            
            # Generate a random candidate c_in
            # We can pick a random word from the whole space
            # If it's in selected, skip (or pick another)
            # To be more effective, we might want to pick c_in that covers words with low count?
            # For now, random sampling.
            
            # We can also try to be smarter:
            # Words that are covered by only 1 codeword (count=1) are "tight".
            # We want to avoid breaking them.
            # Swapping should ideally maintain coverage of tight words.
            
            # Let's try a few random swaps
            for _ in range(50):
                c_in = rng.integers(total)
                if c_in in selected_set:
                    continue
                
                # Check validity of swap c_out -> c_in
                # Valid if: for all w in ball(c_out) \ ball(c_in), cover_count[w] >= 2
                # We need to check the difference set.
                
                # Efficient check:
                # 1. Mark words in ball(c_out)
                # 2. Unmark words in ball(c_in)
                # 3. Check remaining marked words have count >= 2
                
                # Using numpy boolean masks is fast
                # But creating masks every time might be slow.
                # Let's use a temporary array or set logic.
                
                # Since ball_size is small (17), we can just iterate.
                # But we need to check intersection.
                
                # Let's construct the set of words to check
                # words_to_check = set(balls[c_out]) - set(balls[c_in])
                # This is slow in Python.
                
                # Alternative:
                # Count how many words in ball(c_out) have count 1.
                # If we remove c_out, those words become uncovered UNLESS they are in ball(c_in).
                # So we need: (words in ball(c_out) with count 1) subset of ball(c_in).
                
                # Let's find words in ball(c_out) with count 1
                ball_out = balls[c_out]
                ball_in = balls[c_in]
                
                # Create a set of ball_in for O(1) lookup
                # Or use numpy isin
                # isin(ball_out, ball_in)
                
                mask_in = np.isin(ball_out, ball_in)
                # Words in ball_out NOT in ball_in
                mask_not_in = ~mask_in
                critical_words = ball_out[mask_not_in]
                
                if len(critical_words) > 0:
                    # Check if any of these have count 1
                    if np.min(cover_count[critical_words]) < 2:
                        continue # Invalid swap
                
                # Swap is valid.
                # Does it improve slack?
                # New slack = Old slack - sum(cover_count[ball_out]) + sum(cover_count[ball_in])
                # Note: cover_count is current state.
                # When we swap, we decrement cover_count for ball_out and increment for ball_in.
                # Delta slack = -sum(cover_count[ball_out]) + sum(cover_count[ball_in])
                
                sum_out = np.sum(cover_count[ball_out])
                sum_in = np.sum(cover_count[ball_in])
                
                if sum_in < sum_out:
                    # Perform swap
                    cover_count[ball_out] -= 1
                    cover_count[ball_in] += 1
                    
                    selected[c_out_idx] = c_in
                    selected_set.remove(c_out)
                    selected_set.add(c_in)
                    
                    current_slack += sum_in - sum_out
                    improved = True
                    break
            else:
                # No improvement found in this batch of 50
                pass
        else:
            break
            
        if not improved:
            # If we didn't improve in a full pass, we might be stuck in a local minimum.
            # We could try a more aggressive perturbation, but for now, let's just break 
            # and let the outer loop restart with a new seed.
            # Or we can try a few more random swaps without slack improvement to escape?
            # Let's just break to save time for the next seed.
            break

    return selected, len(selected)
