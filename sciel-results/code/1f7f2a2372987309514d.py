import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # The task is specific to q=3, n=6, radius=1
    # But the interface is general. We should handle the specific case efficiently.
    # If q, n, radius don't match, we might need a fallback, but the prompt implies 
    # the evaluator passes this task's parameters.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: simple greedy for other cases (not optimized)
        return _general_greedy(q, n, radius)
    
    total_words = 3 ** n
    # Precompute balls for all possible words in {0,1,2}^n
    # A ball of radius 1 around a word w contains w and all words at distance 1.
    # Size is 1 + n*(q-1) = 1 + 6*2 = 13.
    
    # Map word (tuple) to int
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
        
    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return tuple(w)
        
    # Precompute balls as list of lists of ints
    balls = []
    for i in range(total_words):
        w = int_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                ball.append(word_to_int(w))
            w[pos] = orig
        balls.append(ball)
        
    # 1. Initial Code: Tetracode Sum
    # Tetracode: 9 words of length 4.
    # tetracode = [[a, b, (a+b)%3, (a+2*b)%3] for a in range(3) for b in range(3)]
    # Code = [h + tail for h in tetracode for tail in product(0..2, repeat=2)]
    
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append((a, b, (a + b) % 3, (a + 2 * b) % 3))
            
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + tail
            initial_code.append(word_to_int(w))
            
    best_code = list(initial_code)
    
    def evaluate_code(code):
        """
        Applies greedy removal and swap to reduce the code size.
        Returns the minimized code.
        """
        code = list(code)
        cover_count = np.zeros(total_words, dtype=np.int32)
        
        # Add all initial codewords
        for c in code:
            cover_count[balls[c]] += 1
            
        # Helper to check if min cover count is > 0 (validity)
        # We maintain cover_count, so validity is min(cover_count) >= 1
        
        # Helper to try removals
        def try_remove(code, cover_count):
            removed = True
            while removed:
                removed = False
                # Iterate through a copy of the list to allow modification
                # We iterate in reverse to make removal easier, or just track indices
                for i in range(len(code) - 1, -1, -1):
                    c = code[i]
                    # Check if removing c leaves all covered
                    # min(cover_count[balls[c]]) > 1
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        code.pop(i)
                        removed = True
                        if time.time() > deadline:
                            return code
            return code

        # Helper to try swaps
        def try_swaps(code, cover_count):
            # Try all pairs (c_out, c_in) where c_out in code, c_in not in code
            # To speed up, we can just iterate
            code_set = set(code)
            all_words = range(total_words)
            
            best_swap = None
            best_metric = float('inf')
            
            # We want to find a swap that results in a valid code
            # and ideally reduces the number of "critical" words (cover_count == 1)
            
            for i, c_out in enumerate(code):
                # Temporarily remove c_out
                cover_count[balls[c_out]] -= 1
                
                # Check if code is still valid without c_out? 
                # If it is, we would have removed it in try_remove. 
                # So min(cover_count) must be 0 somewhere? 
                # No, try_remove ensures that no single removal is valid.
                # So removing c_out creates at least one uncovered word.
                
                # We need to add a c_in to cover those.
                # Iterate all potential c_in
                for c_in in all_words:
                    if c_in in code_set:
                        continue
                        
                    # Add c_in
                    cover_count[balls[c_in]] += 1
                    
                    # Check validity
                    if np.min(cover_count) >= 1:
                        # Valid swap
                        # Metric: number of words with cover_count == 1
                        # Lower is better (more robust)
                        metric = np.sum(cover_count == 1)
                        
                        if metric < best_metric:
                            best_metric = metric
                            best_swap = (i, c_in)
                            
                            # Optional: Early stop if metric is very low?
                            # But we want the best one to avoid getting stuck.
                        
                    # Undo c_in
                    cover_count[balls[c_in]] -= 1
                
                # Undo c_out
                cover_count[balls[c_out]] += 1
                
                if time.time() > deadline:
                    break
                    
            if best_swap is not None:
                i, c_in = best_swap
                c_out = code[i]
                code[i] = c_in
                code_set.remove(c_out)
                code_set.add(c_in)
                return True
            return False

        # Main Loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # 1. Remove
            len_before = len(code)
            code = try_remove(code, cover_count)
            if len(code) < len_before:
                improved = True
            
            # 2. Swap
            if len(code) > 73: # Only try to swap if we haven't hit target
                if try_swaps(code, cover_count):
                    improved = True
                    # After a swap, we should restart removals
                    # The loop continues, so try_remove will run again
        
        return code

    # Run the search
    candidate = evaluate_code(best_code)
    
    # If we didn't get 73, try a greedy construction from scratch
    if len(candidate) > 73 and time.time() < deadline - 5.0:
        # Greedy Construction
        greedy_code = []
        cover_count = np.zeros(total_words, dtype=np.int32)
        uncovered = set(range(total_words))
        
        while uncovered and time.time() < deadline - 5.0:
            # Pick a random uncovered word to break symmetry?
            # Or just pick the first one?
            # To make it deterministic and fast, pick the one that is covered by the fewest potential codewords?
            # Simpler: Pick the first uncovered word.
            target_word = next(iter(uncovered))
            
            best_c = -1
            best_new_cover = -1
            
            # Find codeword that covers target_word and maximizes new coverage
            # Candidates are all words in the ball of target_word?
            # No, the codeword c must be such that target_word is in ball(c).
            # This means c is in ball(target_word).
            # So we only check 13 candidates!
            
            for c in balls[target_word]:
                # Count how many currently uncovered words are in ball(c)
                new_cover = 0
                for w in balls[c]:
                    if w in uncovered:
                        new_cover += 1
                if new_cover > best_new_cover:
                    best_new_cover = new_cover
                    best_c = c
            
            if best_c != -1:
                greedy_code.append(best_c)
                # Update cover_count and uncovered
                for w in balls[best_c]:
                    if w in uncovered:
                        uncovered.remove(w)
                # Note: we don't strictly need cover_count for the greedy phase 
                # if we just track 'uncovered', but we need it for the local search phase.
                # Let's rebuild cover_count at the end.
                
        # Rebuild cover_count for local search
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in greedy_code:
            cover_count[balls[c]] += 1
            
        # Run local search on greedy code
        # We need a version of evaluate_code that takes cover_count
        # Let's refactor evaluate_code to accept initial state or just rebuild it.
        
        # Re-using the logic, but we need to pass the cover_count.
        # Let's create a helper that runs the local search given a code and its cover_count.
        
        def local_search(code, cover_count):
            code = list(code)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                len_before = len(code)
                
                # Remove
                for i in range(len(code) - 1, -1, -1):
                    c = code[i]
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        code.pop(i)
                        improved = True
                        if time.time() > deadline:
                            return code
                
                if len(code) < len_before:
                    continue # Restart loop for removals
                
                # Swap
                if len(code) > 73:
                    code_set = set(code)
                    best_swap = None
                    best_metric = float('inf')
                    
                    for i, c_out in enumerate(code):
                        cover_count[balls[c_out]] -= 1
                        for c_in in range(total_words):
                            if c_in in code_set:
                                continue
                            cover_count[balls[c_in]] += 1
                            if np.min(cover_count) >= 1:
                                metric = np.sum(cover_count == 1)
                                if metric < best_metric:
                                    best_metric = metric
                                    best_swap = (i, c_in)
                            cover_count[balls[c_in]] -= 1
                        cover_count[balls[c_out]] += 1
                        if time.time() > deadline:
                            break
                    
                    if best_swap:
                        i, c_in = best_swap
                        c_out = code[i]
                        code[i] = c_in
                        code_set.remove(c_out)
                        code_set.add(c_in)
                        improved = True
            
            return code

        greedy_candidate = local_search(greedy_code, cover_count)
        
        if len(greedy_candidate) < len(candidate):
            candidate = greedy_candidate
            
    return [list(int_to_word(c)) for c in candidate]

def _general_greedy(q, n, radius):
    # Fallback for non-specific cases
    # Simple greedy set cover
    total_words = q ** n
    words = [list(map(int, str(i).zfill(n))) if False else None for i in range(total_words)]
    # Generate all words properly
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    def dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    covered = [False] * total_words
    code = []
    
    while any(not c for c in covered):
        best_idx = -1
        best_count = -1
        for i, w in enumerate(words):
            if covered[i]:
                continue
            count = 0
            for j, w2 in enumerate(words):
                if dist(w, w2) <= radius:
                    if not covered[j]:
                        count += 1
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1:
            break
        code.append(words[best_idx])
        for j, w2 in enumerate(words):
            if dist(words[best_idx], w2) <= radius:
                covered[j] = True
                
    return code
