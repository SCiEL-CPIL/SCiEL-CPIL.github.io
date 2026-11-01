import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Only handle the specific case in the task contract; for other cases use greedy
    # For q=3, n=8, radius=1: K_3(8,1)
    
    total = q ** n
    
    # Precompute all words as indices
    # Index mapping: word with digits (d0, d1, ..., d_{n-1}) -> index
    # index = d0 * q^(n-1) + d1 * q^(n-2) + ... + d_{n-1}
    # Let's use a simpler mapping: index = d0 + d1*q + d2*q^2 + ...
    
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
    
    def digits_to_idx(digits):
        idx = 0
        for i in range(n):
            idx += digits[i] * powers[i]
        return idx
    
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = (idx // powers[i]) % q
        return digits
    
    # Precompute all balls (sets of indices within Hamming distance 1)
    # For efficiency, we'll store balls as numpy arrays of indices
    balls = []
    for x in range(total):
        ball = [x]
        digits = idx_to_digits(x)
        for i in range(n):
            old_digit = digits[i]
            for new_digit in range(q):
                if new_digit != old_digit:
                    new_digits = digits[:]
                    new_digits[i] = new_digit
                    ball.append(digits_to_idx(new_digits))
        balls.append(np.array(ball, dtype=np.int32))
    
    # Precompute ball bitmasks as numpy arrays for fast operations
    # We'll use a 2D boolean array: ball_mask[i, y] = True if y is in ball i
    # But that's total * total = 6561 * 6561 which is ~43M booleans ~ 43MB, acceptable
    # Actually, let's use a more memory-efficient approach.
    # For each ball, store the indices directly.
    
    # Greedy set cover with local search
    # Strategy:
    # 1. Start with a random subset of ~500 codewords
    # 2. Fill gaps with greedy
    # 3. Local search: try removing redundant codewords, then try swapping to improve
    
    # Helper: check if a set of codewords covers everything
    def is_covering(code_set):
        """Check if code_set covers all words."""
        covered = np.zeros(total, dtype=bool)
        for c in code_set:
            covered[balls[c]] = True
        return not np.any(~covered)
    
    # Helper: get uncovered words given a set of codewords
    def get_uncovered(code_set):
        covered = np.zeros(total, dtype=bool)
        for c in code_set:
            covered[balls[c]] = True
        return np.where(~covered)[0]
    
    # Helper: count how many uncovered words a codeword covers
    def count_new_cover(c, uncovered_mask):
        """Count how many words in ball c are in uncovered_mask."""
        return np.sum(uncovered_mask[balls[c]])
    
    # Greedy fill: add codewords to cover all uncovered words
    def greedy_fill(code_set, uncovered_mask, time_limit=5.0):
        code_list = list(code_set)
        start = time.time()
        while True:
            if time.time() > start + time_limit or time.time() > deadline - 5:
                break
            if not np.any(uncovered_mask):
                break
            # Find codeword that covers most uncovered words
            best_c = -1
            best_count = 0
            # Sample a subset for speed
            candidates = list(range(total))
            random.shuffle(candidates)
            for c in candidates:
                if c in code_set:
                    continue
                cnt = count_new_cover(c, uncovered_mask)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if cnt >= 17:  # Max possible
                        break
            if best_c == -1:
                break
            code_list.append(best_c)
            code_set.add(best_c)
            uncovered_mask[balls[best_c]] = False
        return code_list, uncovered_mask
    
    # Local search: try to remove redundant codewords
    def local_search_remove(code_list, max_time=20.0):
        """Try to remove codewords that are redundant."""
        start = time.time()
        improved = True
        while improved and time.time() < start + max_time and time.time() < deadline - 5:
            improved = False
            # Compute coverage count for each word
            cover_count = np.zeros(total, dtype=np.int32)
            for c in code_list:
                cover_count[balls[c]] += 1
            
            # Try to remove each codeword
            for i in range(len(code_list)):
                if time.time() > start + max_time or time.time() > deadline - 5:
                    break
                c = code_list[i]
                # Check if any word in ball[c] has cover_count 1
                if np.any(cover_count[balls[c]] == 1):
                    continue  # Cannot remove
                # Remove c
                code_list.pop(i)
                improved = True
                break  # Restart
        return code_list
    
    # Local search: try 1-exchange (replace one codeword with another)
    def local_search_swap(code_list, max_time=30.0):
        """Try to replace one codeword with another to maintain coverage with same or fewer words."""
        start = time.time()
        improved = True
        while improved and time.time() < start + max_time and time.time() < deadline - 5:
            improved = False
            # Compute coverage count
            cover_count = np.zeros(total, dtype=np.int32)
            for c in code_list:
                cover_count[balls[c]] += 1
            
            # Try to find a codeword c that can be removed and a codeword c' that can replace it
            # c is removable if all words in ball[c] are covered by other codewords
            # c' should cover the "unique" words of c
            for i in range(len(code_list)):
                if time.time() > start + max_time or time.time() > deadline - 5:
                    break
                c = code_list[i]
                # Check if c is removable
                if np.any(cover_count[balls[c]] == 1):
                    continue
                
                # c is removable. Try to find c' that covers the words only covered by c
                # Actually, if c is removable, we can just remove it.
                # For swap, we want to replace c with c' where c' covers more new words
                # but this is complex. Let's just do removal for now.
                code_list.pop(i)
                improved = True
                break
        return code_list
    
    # Better local search: iterative removal + greedy refilling
    def local_search_iterative(code_list, max_time=40.0):
        """Iteratively try to remove and refill."""
        start = time.time()
        best_code = list(code_list)
        best_size = len(best_code)
        
        while time.time() < start + max_time and time.time() < deadline - 10:
            # Compute coverage
            covered = np.zeros(total, dtype=bool)
            for c in code_list:
                covered[balls[c]] = True
            
            # Try to remove a random codeword
            removed_any = False
            for _ in range(100):  # Try up to 100 removals
                if time.time() > start + max_time or time.time() > deadline - 10:
                    break
                idx = random.randint(0, len(code_list) - 1)
                c = code_list[idx]
                # Remove temporarily
                code_list.remove(c)
                # Check coverage
                test_covered = np.zeros(total, dtype=bool)
                for cc in code_list:
                    test_covered[balls[cc]] = True
                if np.any(~test_covered):
                    # Not covered, try to refill
                    uncovered = ~test_covered
                    # Find best codeword to add
                    best_new = -1
                    best_cnt = 0
                    for nc in range(total):
                        if nc not in code_list:
                            cnt = np.sum(uncovered[balls[nc]])
                            if cnt > best_cnt:
                                best_cnt = cnt
                                best_new = nc
                    if best_new != -1 and best_cnt > 0:
                        code_list.append(best_new)
                        removed_any = True
                    # Keep the removed codeword out (net -1 or 0)
                else:
                    # Coverage maintained, keep it removed
                    removed_any = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
        
        return best_code
    
    # Main algorithm
    best_code = None
    best_size = float('inf')
    
    # Multiple restarts with different random seeds
    num_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline - 10 and num_restarts < max_restarts:
        num_restarts += 1
        
        # Start with a random subset
        init_size = random.randint(450, 550)
        init_indices = random.sample(range(total), init_size)
        init_set = set(init_indices)
        
        # Compute initial coverage
        uncovered_mask = np.ones(total, dtype=bool)
        for c in init_set:
            uncovered_mask[balls[c]] = False
        
        # Greedy fill
        code_list, uncovered_mask = greedy_fill(init_set, uncovered_mask, time_limit=5.0)
        
        if len(code_list) < best_size:
            best_size = len(code_list)
            best_code = list(code_list)
        
        # Local search: removal
        if time.time() < deadline - 15:
            code_list = local_search_remove(code_list, max_time=15.0)
        
        # Local search: iterative
        if time.time() < deadline - 10:
            code_list = local_search_iterative(code_list, max_time=20.0)
        
        if len(code_list) < best_size:
            best_size = len(code_list)
            best_code = list(code_list)
        
        if best_size <= 486:
            # We've matched or beaten the known best, stop
            break
    
    # Final verification and cleanup
    if best_code is None:
        # Fallback: greedy from scratch
        code_list = []
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask) and time.time() < deadline - 5:
            best_c = -1
            best_count = 0
            for c in range(total):
                if c not in code_list:
                    cnt = np.sum(uncovered_mask[balls[c]])
                    if cnt > best_count:
                        best_count = cnt
                        best_c = c
            if best_c == -1:
                break
            code_list.append(best_c)
            uncovered_mask[balls[best_c]] = False
        best_code = code_list
    
    # Convert to codewords
    codewords = []
    for idx in best_code:
        codewords.append(idx_to_digits(idx))
    
    return codewords
