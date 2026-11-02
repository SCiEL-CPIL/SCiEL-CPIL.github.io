import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls (neighborhoods of radius 1)
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
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
                
    # Precompute ball membership as a boolean matrix for fast subset checks
    # ball_membership[w, v] = True if v is in balls[w]
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True
        
    # Helper: Convert index to list of digits
    def idx_to_digits(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        return digits

    # Helper: Greedy construction
    def greedy_construct():
        sel = []
        cc = np.zeros(total, dtype=np.int32)
        unc = np.ones(total, dtype=bool)
        
        while np.any(unc):
            # Find word that covers most uncovered points
            # counts[v] = number of uncovered points in ball[v]
            counts = unc[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            # Randomly pick one to break symmetry
            best_idx = candidates[np.random.randint(len(candidates))]
            sel.append(best_idx)
            cc[balls[best_idx]] += 1
            unc = (cc == 0)
            
        return sel

    # Helper: Local Search (Hill Climbing)
    def local_search(selected, cover_count):
        # Returns (new_selected, new_cover_count, improved)
        improved = True
        passes = 0
        while improved and passes < 50 and time.time() < deadline - 5.0:
            improved = False
            passes += 1
            new_selected = []
            
            for c in selected:
                # Check if c is redundant
                if np.min(cover_count[balls[c]]) > 1:
                    # Remove c
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                # Try to find a better swap
                # Critical words: covered by exactly 1 codeword (c)
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    # c is redundant (all its words are covered by others)
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                # Find candidates that cover all critical words
                # candidate_mask[v] = True if v covers all critical_words
                candidate_mask = np.ones(total, dtype=bool)
                for cw in critical_words:
                    candidate_mask &= ball_membership[:, cw]
                
                valid_candidates = np.where(candidate_mask)[0]
                valid_candidates = valid_candidates[valid_candidates != c]
                
                if len(valid_candidates) > 0:
                    # Choose candidate that minimizes "variance" or maximizes potential removal
                    # Simple heuristic: pick random valid candidate
                    # Better heuristic: pick candidate that covers most words already covered by c (overlap)
                    # Or pick candidate that has lowest max cover count in its ball
                    best_c_prime = None
                    best_score = -1
                    
                    # Sample a few candidates if list is large
                    sample_size = min(10, len(valid_candidates))
                    indices = np.random.choice(len(valid_candidates), sample_size, replace=False)
                    for i in indices:
                        c_prime = valid_candidates[i]
                        # Score: -max(cover_count[balls[c_prime]]) + overlap
                        # We want to reduce peak load
                        max_load = np.max(cover_count[balls[c_prime]])
                        # Overlap with c: how many words in ball[c] are in ball[c_prime]
                        # This is expensive. Let's just use max_load.
                        score = -max_load
                        if score > best_score:
                            best_score = score
                            best_c_prime = c_prime
                            
                    if best_c_prime is not None:
                        # Perform swap
                        cover_count[balls[c]] -= 1
                        cover_count[balls[best_c_prime]] += 1
                        new_selected.append(best_c_prime)
                        improved = True
                    else:
                        new_selected.append(c)
                else:
                    new_selected.append(c)
                    
            selected = new_selected
            
        return selected, cover_count, improved

    # Helper: Kick (Perturbation)
    def kick(selected, cover_count):
        # Randomly swap 2 codewords or remove/add 1
        if len(selected) < 2:
            return selected, cover_count
            
        move_type = np.random.random()
        if move_type < 0.5:
            # Swap 2 random codewords
            i, j = np.random.choice(len(selected), 2, replace=False)
            c_i = selected[i]
            c_j = selected[j]
            
            # Ensure they are distinct
            if c_i == c_j:
                return selected, cover_count
                
            # Update counts
            cover_count[balls[c_i]] -= 1
            cover_count[balls[c_j]] -= 1
            cover_count[balls[c_j]] += 1
            cover_count[balls[c_i]] += 1
            
            selected[i] = c_j
            selected[j] = c_i
        else:
            # Remove 1 random codeword (even if it breaks coverage temporarily)
            # And add 1 random new codeword
            i = np.random.randint(len(selected))
            c_old = selected[i]
            cover_count[balls[c_old]] -= 1
            selected.pop(i)
            
            # Add a random codeword from the space
            c_new = np.random.randint(total)
            # Ensure not duplicate
            while c_new in selected:
                c_new = np.random.randint(total)
                
            cover_count[balls[c_new]] += 1
            selected.append(c_new)
            
        return selected, cover_count

    # Main Loop: Iterated Local Search
    best_size = total
    best_set = None
    
    # Initial Greedy
    selected = greedy_construct()
    cover_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        cover_count[balls[c]] += 1
        
    if len(selected) < best_size:
        best_size = len(selected)
        best_set = selected.copy()
        
    stall_count = 0
    
    while time.time() < deadline - 2.0:
        # Local Search
        selected, cover_count, improved = local_search(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            stall_count = 0
            if best_size <= 403:
                break
        else:
            stall_count += 1
            
        # If stuck, kick
        if stall_count > 5:
            selected, cover_count = kick(selected, cover_count)
            stall_count = 0
            
    # Final Polish
    if best_set is not None:
        selected = best_set.copy()
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        local_search(selected, cover_count)
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

    # Convert best_set to output format
    if best_set is None:
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
