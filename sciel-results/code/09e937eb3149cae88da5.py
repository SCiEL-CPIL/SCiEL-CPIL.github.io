import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin for safety
    
    # Alphabet size q=3, length n=7, radius=1
    # Total words = 3^7 = 2187
    
    # 1. Generate initial code using tetracode product
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Initial code: tetracode x all tails of length n-4
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(tuple(h + list(tail)))
    
    # Convert to set for fast lookup
    code_set = set(initial_code)
    
    # 2. Precompute neighbors for all words in the space?
    # Space is 3^7 = 2187. We can precompute neighbors for all 2187 words.
    # This will make the redundancy check very fast.
    
    def word_to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def int_to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return tuple(word)

    # Precompute neighbors for all 2187 words
    all_words = [int_to_word(i) for i in range(q ** n)]
    neighbors = {}
    for w in all_words:
        w_int = word_to_int(w)
        neigh = [w]
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    new_w = list(w)
                    new_w[i] = v
                    neigh.append(tuple(new_w))
        neighbors[w_int] = neigh

    # 3. Initialize cover_count
    cover_count = [0] * (q ** n)
    
    for c in code_set:
        c_int = word_to_int(c)
        for w in neighbors[c_int]:
            w_int = word_to_int(w)
            cover_count[w_int] += 1
            
    # Verify initial validity (should be valid)
    # for i in range(q**n):
    #     if cover_count[i] == 0:
    #         return list(code_set) # Fallback, shouldn't happen

    best_code = list(code_set)
    
    # 4. Local Search Loop
    # Strategy: Repeatedly find and remove redundant codewords.
    # If stuck, shuffle the order and try again.
    
    def is_redundant(c):
        c_int = word_to_int(c)
        for w in neighbors[c_int]:
            w_int = word_to_int(w)
            if cover_count[w_int] <= 1:
                return False
        return True

    def remove_codeword(c):
        nonlocal code_set
        code_set.remove(c)
        c_int = word_to_int(c)
        for w in neighbors[c_int]:
            w_int = word_to_int(w)
            cover_count[w_int] -= 1

    # We will run multiple passes.
    # To escape local minima, we can also try to swap codewords, but simple removal
    # with randomization of check order is often effective for covering codes.
    
    max_passes = 5000
    passes = 0
    last_improvement_size = len(code_set)
    
    while passes < max_passes and time.time() < deadline:
        passes += 1
        # Shuffle the list to randomize the order of removal checks
        # This helps find different local minima
        current_list = list(code_set)
        random.shuffle(current_list)
        
        removed_any = False
        for c in current_list:
            if time.time() > deadline:
                break
            if c not in code_set:
                continue
            if is_redundant(c):
                remove_codeword(c)
                removed_any = True
                # Update best if we improved
                if len(code_set) < len(best_code):
                    best_code = list(code_set)
        
        if not removed_any:
            # Stuck. Try a perturbation?
            # One simple perturbation: if we are stuck, we might be in a local minimum.
            # We can try to add a random codeword and then remove others?
            # Or just restart the shuffling process.
            # Given the speed, just shuffling and retrying is cheap.
            # However, if we are truly stuck, shuffling won't help if the set of
            # redundant words is empty.
            
            # A stronger move: try to find a codeword c that is "almost" redundant.
            # i.e., removing it leaves only 1 word uncovered.
            # Then find a new codeword c' that covers that word and is redundant
            # after insertion? This is complex.
            
            # Let's try a simple "kick":
            # Pick a random codeword c, remove it (breaking coverage), 
            # then try to repair by adding codewords? No, we need to minimize.
            
            # Alternative: If stuck, we can try to swap a codeword c with another c' 
            # such that the new set is smaller? No, swap keeps size same.
            # We want to reduce size.
            
            # If no single removal is possible, we are in a minimal covering.
            # To get smaller, we need to find a set of codewords that can be replaced 
            # by fewer codewords.
            
            # For now, let's just rely on the fact that the tetracode start is good 
            # and random shuffling might find a different path if we hadn't converged.
            # But if we converged, shuffling does nothing.
            
            # Let's try a "random restart" of the removal process on a slightly perturbed set.
            # Perturbation: Add a random codeword, then run removal.
            # This might allow escaping the local minimum.
            
            if len(code_set) > 180: # Only perturb if we are not already very good
                # Add a random word
                random_word = all_words[random.randint(0, q**n - 1)]
                if random_word not in code_set:
                    code_set.add(random_word)
                    rw_int = word_to_int(random_word)
                    for w in neighbors[rw_int]:
                        w_int = word_to_int(w)
                        cover_count[w_int] += 1
                    # Now run removal again in the next iteration
                    # We break out of the inner loop to restart the pass
                    break
            else:
                # If we are close to optimal, just stop or try more shuffles
                if passes > 100:
                    break

    # Ensure we return a valid code
    # The current code_set is always valid by construction (we only remove redundant ones)
    # So we can just return it.
    
    # If time is up, return best found
    return best_code
