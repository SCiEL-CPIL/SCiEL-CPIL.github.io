import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    import itertools
    if q == 3 and n >= 4:
        # Tetracode based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    
    # Precompute neighbors (balls) for all 6561 words
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    balls = []
    for w in range(total):
        ball = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                ball.append(new_w)
        balls.append(ball)
    
    balls_np = np.array(balls, dtype=np.int32)
    
    best_size = total
    best_set = None

    # Helper: Remove redundant codewords efficiently
    def remove_redundant(codewords):
        if not codewords:
            return []
        
        current_set = set(codewords)
        cover_count = np.zeros(total, dtype=np.int32)
        
        # Initial count
        for c in current_set:
            cover_count[balls_np[c]] += 1
            
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            # Find a codeword c such that all words in its ball have count > 1
            # i.e., no word in ball(c) has count == 1.
            # We can check this for each c in current_set.
            # To speed up, we can just iterate.
            
            # Optimization: Check if there are any words with count == 1 globally.
            # If not, we can remove any codeword?
            # No, if no word has count 1, then every word has count >= 2.
            # Removing any c will keep all words covered (count >= 1).
            # So we can just remove one and restart?
            # Actually, if min(cover_count) >= 2, we can remove ANY codeword.
            
            min_count = np.min(cover_count)
            if min_count >= 2:
                # Remove an arbitrary codeword
                c_to_remove = next(iter(current_set))
                current_set.remove(c_to_remove)
                cover_count[balls_np[c_to_remove]] -= 1
                changed = True
                continue
            
            # If min_count == 1, we need to find a codeword c that does NOT cover any word with count 1.
            # Let U be the set of words with count 1.
            # We need c such that ball(c) is disjoint from U.
            # i.e., ball(c) subset of (complement of U).
            
            unique_words_mask = (cover_count == 1)
            
            # Iterate through current_set to find a removable codeword
            # We can randomize the order to avoid bias
            candidates = list(current_set)
            random.shuffle(candidates)
            
            for c in candidates:
                # Check if ball(c) intersects U
                # If no intersection, c is redundant
                # np.any(balls_np[c] is in U)
                # Efficient: count how many words in ball(c) have cover_count == 1
                num_unique_covered = np.sum(cover_count[balls_np[c]] == 1)
                
                if num_unique_covered == 0:
                    # c is redundant
                    current_set.remove(c)
                    cover_count[balls_np[c]] -= 1
                    changed = True
                    break
                    
        return list(current_set)

    # Helper: Local Search with Swap
    def local_search(codewords):
        current_set = set(remove_redundant(codewords))
        if not current_set:
            return []
        
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            cover_count[balls_np[c]] += 1
            
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # Try to find a codeword to swap or remove
            candidates = list(current_set)
            random.shuffle(candidates)
            
            for c in candidates:
                if time.time() > deadline - 1.0:
                    break
                
                # Find words covered ONLY by c
                # U_c = { w | w in ball(c) and cover_count[w] == 1 }
                # Since cover_count is global, cover_count[w] == 1 implies c is the only one covering w (if w in ball(c)).
                # Wait, if cover_count[w] == 1 and w in ball(c), then c MUST be the one covering it?
                # Yes, because cover_count counts how many codewords cover w. If it's 1, only one codeword covers it.
                # If c covers w, then c is that codeword.
                
                # Compute U_c
                c_ball = balls_np[c]
                U_c = c_ball[cover_count[c_ball] == 1]
                
                if len(U_c) == 0:
                    # c is redundant
                    current_set.remove(c)
                    cover_count[c_ball] -= 1
                    improved = True
                    break # Restart loop with new set
                
                # If U_c is not empty, we must cover U_c if we remove c.
                # We look for a candidate c' not in current_set such that U_c subset of ball(c').
                
                # Strategy: 
                # If |U_c| == 1, we can pick any c' in ball(U_c[0]) \ current_set.
                # If |U_c| > 1, we need c' in intersection of ball(w) for w in U_c.
                
                found_swap = False
                U_c_list = list(U_c)
                
                # Heuristic: Try to find c' that covers U_c.
                # We can iterate through a sample of non-selected codewords or all.
                # Given 6561 size, iterating all is fast if the check is vectorized.
                
                # To speed up, we can prioritize c' that are "close" to c? 
                # Or just randomize.
                
                # Let's iterate through all non-selected codewords.
                # To avoid generating a list of 6000 every time, we can just loop 0..total and skip if in current_set.
                # Checking `c_prime in current_set` is O(1) for set.
                
                for c_prime in range(total):
                    if c_prime in current_set:
                        continue
                    
                    # Check if c_prime covers all words in U_c
                    # U_c is small.
                    # Check: all(w in balls_np[c_prime] for w in U_c_list)
                    # Vectorized check:
                    # Create a boolean mask for U_c in balls_np[c_prime]?
                    # Simpler:
                    valid = True
                    for w in U_c_list:
                        # Check if w is in balls_np[c_prime]
                        # Since balls_np[c_prime] is sorted? No, not necessarily.
                        # But we can use np.isin?
                        # Or just check if (w - c_prime) is a valid step?
                        # Hamming distance check is O(n).
                        # Let's use the precomputed balls.
                        # Is w in balls_np[c_prime]?
                        # We can precompute a lookup table? 
                        # Or use the fact that balls are small.
                        # Let's stick to a simple loop for U_c_list, which is small.
                        if w not in balls_np[c_prime]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        current_set.remove(c)
                        cover_count[c_ball] -= 1
                        
                        current_set.add(c_prime)
                        cover_count[balls_np[c_prime]] += 1
                        
                        improved = True
                        found_swap = True
                        break # Restart outer loop
                    
                if found_swap:
                    break # Restart while loop with new set
                    
        return list(current_set)

    # Run multiple restarts
    num_restarts = 50 # Increased restarts due to faster search
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Greedy Initialization
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 5.0:
                break
            
            # Vectorized count of uncovered words covered by each candidate
            # counts[c] = number of uncovered words in ball(c)
            # We can compute this by summing uncovered_mask over balls_np
            # This is 6561 x 17 sum. Fast enough?
            # 6561 * 17 ~ 100k operations. Yes.
            
            counts = uncovered_mask[balls_np].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls_np[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue
            
        # Local Search
        refined_set = local_search(selected)
        
        if len(refined_set) < best_size:
            best_size = len(refined_set)
            best_set = refined_set
            
    # Fallback
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
        
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
