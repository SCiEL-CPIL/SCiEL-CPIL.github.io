import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    # 1. Generate initial code: Direct sum of ternary tetracode
    # Tetracode: 9 words of length 4.
    # C = { (a, b, a+b, a+2b) mod 3 }
    tetracode = []
    for a in range(q):
        for b in range(q):
            tetracode.append([a, b, (a + b) % q, (a + 2 * b) % q])
    
    # Extend to length n by adding all possible tails
    # Tails are all words in {0..q-1}^(n-4)
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    initial_code = []
    for h in tetracode:
        for tail in tails:
            initial_code.append(h + list(tail))
            
    random.shuffle(initial_code)
    
    best_code = list(initial_code)
    best_size = len(initial_code)
    
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
        return w
        
    # Pre-compute the ball of radius 1 for each integer in the space
    size_space = q**n
    balls = {}
    for i in range(size_space):
        w = int_to_word(i)
        neighbors = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                w[pos] = new_val
                neighbors.append(word_to_int(w))
            w[pos] = original_val
        balls[i] = neighbors
        
    cov = [0] * size_space
    current_code = list(initial_code)
    
    # Initial coverage
    for c in current_code:
        ci = word_to_int(c)
        for nb in balls[ci]:
            cov[nb] += 1
            
    # Helper to check if a swap from c_old_idx to c_new_word is valid
    # c_old_idx: index in current_code
    # c_new_word: list of ints
    # Returns True if valid, False otherwise
    def is_valid_swap(c_old_idx, c_new_word):
        c_old = current_code[c_old_idx]
        ci_old = word_to_int(c_old)
        ci_new = word_to_int(c_new_word)
        
        # Check if new word is already in code (optional, but good for distinctness)
        # We will handle distinctness by checking set, but for speed, 
        # if c_new == c_old, it's a no-op.
        if ci_old == ci_new:
            return False # No change, or we can return True but it's useless
            
        # Check coverage constraints
        # For all w in ball(c_old), if cov[w] == 1, then w must be in ball(c_new)
        ball_old = balls[ci_old]
        ball_new = balls[ci_new]
        set_new = set(ball_new) # Small, 17 elements
        
        for w in ball_old:
            if cov[w] == 1:
                if w not in set_new:
                    return False
        return True

    # Phase 1: Deletion
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current_code)):
            c = current_code[i]
            ci = word_to_int(c)
            redundant = True
            for nb in balls[ci]:
                if cov[nb] == 1:
                    redundant = False
                    break
            
            if redundant:
                current_code.pop(i)
                for nb in balls[ci]:
                    cov[nb] -= 1
                changed = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                break # Restart scan

    # Phase 2: Swapping (Local Search)
    # We try to find a critical word, identify its unique coverer, and swap it.
    # We do this in a loop, trying random critical words.
    
    # To speed up finding critical words, we can maintain a list, but 
    # scanning 6561 ints is fast (~0.001s).
    
    while time.time() < deadline:
        # Find a critical word
        critical_words = []
        for i in range(size_space):
            if cov[i] == 1:
                critical_words.append(i)
        
        if not critical_words:
            break # No critical words, might be optimal or stuck in local min where all cov >= 2?
            # If all cov >= 2, we can't delete. We might need to move codewords to reduce size?
            # No, size is fixed. We just want to ensure validity.
            # If no critical words, the code is "2-covered" everywhere.
            # We can still try swaps to see if we can enable deletions?
            # Swapping doesn't change size. It only changes structure.
            # If we are stuck with size S, and no critical words, we are in a stable state.
            # We should probably break or try random perturbations.
            # Given time constraints, breaking is safer.
            pass

        # Pick a random critical word
        w_c = random.choice(critical_words)
        
        # Find the unique codeword covering w_c
        c_old_idx = -1
        for idx, c in enumerate(current_code):
            ci = word_to_int(c)
            if w_c in balls[ci]:
                c_old_idx = idx
                break
        
        if c_old_idx == -1:
            # Should not happen if cov[w_c] == 1
            continue
            
        c_old = current_code[c_old_idx]
        ci_old = word_to_int(c_old)
        
        # Generate candidates for c_new
        # Candidates are words at distance <= 1 from w_c
        # There are 17 such words.
        # We can just iterate through them.
        # Or generate them explicitly.
        w_c_list = int_to_word(w_c)
        candidates = []
        # The word itself
        candidates.append(list(w_c_list))
        # Neighbors
        for pos in range(n):
            orig = w_c_list[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                cand = list(w_c_list)
                cand[pos] = new_val
                candidates.append(cand)
        
        random.shuffle(candidates)
        
        swapped = False
        for c_new in candidates:
            ci_new = word_to_int(c_new)
            
            # Skip if same as old
            if ci_new == ci_old:
                continue
                
            # Check if c_new is already in current_code
            # This is O(N) check. N~500. 17 candidates * 500 = 8500 checks.
            # We can optimize by keeping a set of ints.
            # But updating the set on swap is O(1).
            # Let's maintain a set of current code ints for O(1) lookup.
            
            # Actually, let's just do the is_valid_swap check first, it's cheaper.
            if is_valid_swap(c_old_idx, c_new):
                # Check distinctness
                # If c_new is already in code, we would have a duplicate.
                # We need to ensure distinctness.
                # If c_new is in code, we can't add it.
                # We can check: if ci_new in [word_to_int(c) for c in current_code]
                # To speed this up, let's maintain a set.
                
                # Let's add a set to the outer scope?
                # No, let's just check.
                # Since N is small, linear search is fine.
                is_dup = False
                for c in current_code:
                    if word_to_int(c) == ci_new:
                        is_dup = True
                        break
                if is_dup:
                    continue
                    
                # Perform swap
                # Update cov
                for nb in balls[ci_old]:
                    cov[nb] -= 1
                for nb in balls[ci_new]:
                    cov[nb] += 1
                    
                # Update code list
                current_code[c_old_idx] = c_new
                
                swapped = True
                break
        
        if not swapped:
            # If we couldn't swap for this critical word, try another one
            # Or break if we've tried many?
            # Let's just loop and pick a different critical word next time.
            # To avoid infinite loop on same failure, we can shuffle critical_words list externally?
            # We pick random each time, so it should vary.
            pass

    # Final check: is the code valid?
    # The logic ensures validity during swaps.
    # But let's verify best_code is valid?
    # best_code was updated during deletion.
    # The current_code might be better?
    # Swaps don't change size. Deletion does.
    # So best_size is the minimum size found.
    # If we found a smaller size during deletion, best_code holds it.
    # If we didn't delete anything, best_code is the initial 729.
    # If we swapped, size is still 729 (or whatever it was after deletion).
    # So we should return the code with the smallest size.
    
    # If current_code has the same size as best_code, it might be "better" in terms of structure?
    # No, the score is just the number of codewords.
    # So we return best_code.
    
    return best_code
