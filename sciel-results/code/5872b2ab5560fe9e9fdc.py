import time
import numpy as np
import random
import itertools
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters (not expected in this task, but good practice)
    if q == 3 and n >= 4:
        # Simple tetracode extension, not optimal but valid
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    N = q ** n  # 6561
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls (neighborhoods of radius 1)
    # balls[w] is a list of 17 indices: w itself and its 16 neighbors
    # We store as a matrix for vectorized operations
    balls = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    for w in range(N):
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
                
    # Helper to convert index to list of digits
    def idx_to_list(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]

    best_size = N
    best_set = None
    
    # Precompute ball membership for fast candidate filtering in local search
    # ball_membership[w, v] = True if v is in balls[w]
    # This is an N x N boolean matrix (6561 x 6561 ~ 43M booleans ~ 43MB), acceptable.
    ball_membership = np.zeros((N, N), dtype=bool)
    for w in range(N):
        ball_membership[w, balls[w]] = True
    
    # Main Loop: Multiple restarts of Greedy + Local Search
    while time.time() < deadline - 10.0:
        
        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)
        
        # Greedy Loop
        while np.any(uncovered_mask):
            if time.time() > deadline - 15.0:
                break # Time out for this iteration to allow finishing a valid set if possible, 
                      # but actually if we break here, the set is invalid. 
                      # So we should only break if we have no time to finish.
                      # If we break, we discard this solution.
                
            # Count coverage for each candidate among uncovered words
            # counts[c] = number of uncovered words in balls[c]
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break # Should not happen if uncovered_mask has True
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best to encourage diversity
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts
            cover_count[balls[best_idx]] += 1
            
            # Update uncovered mask
            # A word is uncovered if its cover count is 0
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid solution due to timeout in greedy
            
        # 2. Local Search Phase
        # We alternate between pruning and targeted swaps
        
        # Phase 2a: Initial Pruning
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is removable: all words in balls[c] are covered by >= 2 codewords
                # i.e., min(cover_count[balls[c]]) > 1
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Phase 2b: Targeted Swaps and 2-Exchanges
        # We will run this until no improvement is found for a while or time runs out
        # To avoid getting stuck in a cycle, we can keep track of recent sizes or just run for a fixed number of steps per restart
        # Given the time budget, we run a loop that tries to improve.
        
        # Pre-allocate a mask for candidate filtering
        candidate_mask = np.ones(N, dtype=bool)
        
        # We will iterate through the list of selected codewords multiple times
        # Each pass, we try to improve each codeword.
        
        improved_in_pass = True
        passes = 0
        max_passes = 50 # Limit passes to ensure we return within time
        
        while improved_in_pass and passes < max_passes and time.time() < deadline - 5.0:
            passes += 1
            improved_in_pass = False
            new_selected = []
            
            # Shuffle to avoid order bias? Or keep order? 
            # Keeping order is fine, but shuffling might help escape local minima related to order.
            # Let's shuffle indices of selected for processing order.
            indices = list(range(len(selected)))
            np.random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5.0:
                    break
                    
                c = selected[idx]
                
                # Check if c is removable
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved_in_pass = True
                    # Do not add to new_selected
                    continue
                
                # Compute critical set for c
                # Critical words are those in balls[c] with cover_count == 1
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    # Should have been removed, but just in case
                    cover_count[balls[c]] -= 1
                    improved_in_pass = True
                    continue

                # Find candidates c' that cover all critical words
                # c' must be in balls[cw] for all cw in critical_words
                # i.e., ball_membership[c', cw] is True for all cw
                candidate_mask.fill(True)
                for cw in critical_words:
                    candidate_mask &= ball_membership[:, cw]
                
                # Exclude c itself
                candidate_mask[c] = False
                
                valid_candidates = np.where(candidate_mask)[0]
                
                if len(valid_candidates) > 0:
                    # Pick a random candidate
                    c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                    
                    # Perform swap
                    # Safety check:
                    # Removing c: critical words are still covered by c_prime. 
                    # Non-critical words in B(c) were covered by others (count >= 2 including c, so >= 1 after removal).
                    # Adding c_prime: covers its ball.
                    # So the swap is safe.
                    
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved_in_pass = True
                else:
                    # No valid swap found for this c in this pass
                    new_selected.append(c)
            
            selected = new_selected
            
            # If we improved, we might have made some codewords removable again.
            # The next pass will handle pruning.
            
            # Additional 2-Exchange attempt
            # Try to find two codewords c1, c2 that can be replaced by one c3
            # This is expensive O(K^2 * N). We only do it if K is large enough and time permits.
            if len(selected) > 450 and time.time() < deadline - 8.0:
                # Sample a few pairs
                num_pairs_to_try = 50
                for _ in range(num_pairs_to_try):
                    if time.time() > deadline - 8.0:
                        break
                    if len(selected) < 2:
                        break
                    i, j = np.random.choice(len(selected), 2, replace=False)
                    c1 = selected[i]
                    c2 = selected[j]
                    
                    # Words that would be uncovered if we remove c1 and c2
                    # These are words in (B(c1) U B(c2)) where cover_count == 1 (and the cover is c1 or c2)
                    # Actually, if we remove both, a word w is uncovered if:
                    # 1. w is in B(c1) or B(c2)
                    # 2. The only codeword covering w is c1 or c2.
                    
                    # Let's identify the "risk" set: words covered by c1 or c2 but not by any other.
                    # Risk = (B(c1) U B(c2)) \ (Union of B(c) for c not in {c1, c2})
                    
                    # Simpler: 
                    # For each w in B(c1) U B(c2):
                    # If cover_count[w] == 1, it must be covered by c1 or c2.
                    # If we remove both, it is uncovered.
                    # So the risk set is { w in B(c1) U B(c2) | cover_count[w] == 1 }
                    
                    union_ball = np.union1d(balls[c1], balls[c2])
                    risk_mask = (cover_count[union_ball] == 1)
                    risk_words = union_ball[risk_mask]
                    
                    if len(risk_words) == 0:
                        # We can remove both and add nothing? No, we must add something to maintain coverage of the rest?
                        # If risk set is empty, it means every word in the union was covered by at least 2 codewords.
                        # So we can just remove c1 and c2.
                        # But wait, if we remove them, do we lose coverage of words ONLY covered by c1 or c2?
                        # No, cover_count[w] >= 2 for all w in union. So removing c1 and c2 leaves cover_count[w] >= 0?
                        # If cover_count[w] == 2, and one is c1, one is c2, then removing both makes it 0.
                        # So the condition for safe removal of c1, c2 is that for all w in B(c1) U B(c2), cover_count[w] >= 2.
                        # And if cover_count[w] == 2, it cannot be that both c1 and c2 cover w? 
                        # If c1 and c2 both cover w, and no one else does, then cover_count[w] = 2. Removing both makes it 0.
                        # So the condition is: for all w in B(c1) U B(c2), cover_count[w] >= 2 AND (if cover_count[w]==2, not (c1 covers w AND c2 covers w AND no one else covers w)).
                        # This is getting complicated. 
                        # The standard 2-exchange is: remove c1, c2, add c3.
                        # Condition: All words in (B(c1) U B(c2)) \ B(c3) must be covered by other codewords (count >= 1 after removing c1, c2).
                        # And c3 must cover the "risk" words (those with count 1 after removing c1, c2).
                        
                        # Let's simplify: 
                        # Let S = B(c1) U B(c2).
                        # Let R = { w in S | cover_count[w] == 1 }.
                        # If we remove c1, c2, words in R become uncovered.
                        # So c3 must cover R.
                        # Also, words in S \ R have cover_count >= 2. If we remove c1, c2, they have cover_count >= 0.
                        # If cover_count[w] == 2, and c1, c2 are the only ones covering w, then w becomes 0.
                        # So we must check: for w in S, if cover_count[w] == 2, it is NOT the case that {c1, c2} is the set of codewords covering w.
                        
                        # This check is expensive. 
                        # Let's just use the "Targeted Swap" logic which is safer and simpler.
                        # 2-exchange is less critical if 1-swap is good.
                        pass # Skip complex 2-exchange for now to keep code robust and fast.
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
        # If we reached a very good size, we can try to polish with pure random swaps
        # But the targeted swap is already quite good.
        
    if best_set is None:
        # Fallback
        return [idx_to_list(w) for w in range(N)]
        
    return [idx_to_list(w) for w in best_set]
