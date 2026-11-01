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
                
    # Precompute adjacency: for each codeword, which other codewords share at least one covered word
    # This helps with swap-based local search
    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy instances with different seeds
    seed = 0
    while time.time() < deadline - 10.0:
        code, size = _run_greedy_and_prune_v2(balls, total, seed, deadline - 5.0)
        
        if size < best_size:
            best_size = size
            best_code = code
            if size <= 490:
                break
                
        seed += 1
        
    if best_code is None:
        return [list(c) for c in itertools.product(range(q), repeat=n)]
        
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

def _run_greedy_and_prune_v2(balls, total, seed, sub_deadline):
    rng = np.random.default_rng(seed)
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    # Greedy Phase with degree-based selection
    while np.any(current_uncovered):
        if time.time() > sub_deadline:
            break
            
        # Count uncovered neighbors for each word
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        
        if max_count == 0:
            break
        
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[rng.integers(len(candidates))]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
        
    # Aggressive Pruning Phase with multiple passes
    selected_set = set(selected)
    for _pass in range(50):
        if time.time() > sub_deadline:
            break
        if not selected:
            break
            
        sel_arr = np.array(selected, dtype=np.int32)
        sel_balls = balls[sel_arr]
        counts_in_balls = cover_count[sel_balls]
        min_counts = np.min(counts_in_balls, axis=1)
        
        redundant_mask = min_counts > 1
        
        if not np.any(redundant_mask):
            break
        
        redundant_indices = sel_arr[redundant_mask]
        cover_count[balls[redundant_indices]] -= 1
        keep_mask = ~redundant_mask
        selected = sel_arr[keep_mask].tolist()
    
    # Swap-based local search: try to replace two codewords with one
    # A swap is valid if the union of their balls is covered by the remaining codewords
    # plus the new single codeword
    improved = True
    while improved and time.time() < sub_deadline - 2.0:
        improved = False
        if len(selected) < 2:
            break
            
        sel_arr = np.array(selected, dtype=np.int32)
        m = len(sel_arr)
        
        # For efficiency, only try swaps involving pairs that share coverage
        # Precompute for each codeword, which other codewords' balls overlap with its ball
        # A word is in ball[i] and ball[j] if it's within distance 1 of both codewords
        
        # Try random pairs
        for _attempt in range(200):
            if time.time() > sub_deadline - 2.0:
                break
                
            i = rng.integers(m)
            j = rng.integers(m)
            if i == j:
                continue
                
            # Check if removing both i and j leaves some words uncovered
            # that could be covered by a single new codeword
            ball_i = balls[sel_arr[i]]
            ball_j = balls[sel_arr[j]]
            
            # Words covered by i or j
            # A word w is covered by i or j if w is in ball_i or ball_j
            # After removing i and j, w is still covered if cover_count[w] > 1 (since we're removing 2)
            # Wait, cover_count[w] counts how many selected codewords cover w
            # After removing i and j, cover_count_new[w] = cover_count[w] - (1 if w in ball_i else 0) - (1 if w in ball_j else 0)
            # We need cover_count_new[w] >= 1 for all w, OR w is in ball of some new codeword k
            
            # Simplified: find words that would become uncovered (cover_count becomes 0)
            # after removing i and j
            temp_cover = cover_count.copy()
            temp_cover[ball_i] -= 1
            temp_cover[ball_j] -= 1
            
            uncovered_after_removal = np.where(temp_cover == 0)[0]
            
            if len(uncovered_after_removal) == 0:
                # We can remove both i and j without adding anything
                cover_count[ball_i] -= 1
                cover_count[ball_j] -= 1
                selected = [sel_arr[k] for k in range(m) if k != i and k != j]
                improved = True
                break
            elif len(uncovered_after_removal) <= 17:
                # Maybe a single codeword can cover all uncovered words
                # The new codeword must be within distance 1 of all uncovered words
                # This means the new codeword must be in the intersection of balls of all uncovered words
                # For efficiency, just try a few candidates
                
                # Find candidate codewords that cover all uncovered words
                # A codeword k covers word w if w is in balls[k]
                # We need k such that all uncovered words are in balls[k]
                
                # For small sets, brute force
                if len(uncovered_after_removal) <= 5:
                    # Find codewords that cover all uncovered words
                    # For each candidate k, check if all uncovered words are in balls[k]
                    # This is expensive, so limit candidates
                    # Heuristic: try codewords that are close to the uncovered words
                    # Use the first uncovered word to generate candidates
                    first_uncov = uncovered_after_removal[0]
                    # Candidates are words within distance 1 of first_uncov
                    candidates = balls[first_uncov]
                    
                    found = False
                    for k in candidates:
                        # Check if k covers all uncovered words
                        ball_k = balls[k]
                        # Check if all uncovered words are in ball_k
                        # ball_k is an array of 17 words
                        # We need to check if each uncovered word is in ball_k
                        all_covered = True
                        for uw in uncovered_after_removal:
                            if uw not in ball_k:
                                all_covered = False
                                break
                        if all_covered:
                            # Valid swap: remove i and j, add k
                            cover_count[ball_i] -= 1
                            cover_count[ball_j] -= 1
                            cover_count[ball_k] += 1
                            new_selected = [sel_arr[t] for t in range(m) if t != i and t != j]
                            if k not in new_selected:
                                new_selected.append(k)
                            selected = new_selected
                            improved = True
                            found = True
                            break
                    
                    if found:
                        break
    
    return selected, len(selected)
