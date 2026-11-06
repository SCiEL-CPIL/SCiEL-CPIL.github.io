import time
import random

def construct():
    N = 17
    MIN_DIST = 6
    DEPTH = 17
    
    # Deadline guard: 85 seconds to leave buffer
    deadline = time.time() + 85.0
    
    def hamming_dist(a, b):
        return (a ^ b).bit_count()
    
    def is_valid(word, code_list):
        for c in code_list:
            if hamming_dist(word, c) < MIN_DIST:
                return False
        return True

    best_code = []
    
    # Helper to run one search instance
    def run_search(init_code, max_time):
        current_code = list(init_code)
        start_time = time.time()
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 2.0 and time.time() - start_time < max_time:
            # 1. Try to add a random word
            if random.random() < 0.3: # 30% chance to try add
                w = random.getrandbits(N)
                if is_valid(w, current_code):
                    current_code.append(w)
                    no_improve = 0
                    continue
            
            # 2. Try to swap a word
            # Pick a random index
            i = random.randint(0, len(current_code) - 1)
            old_w = current_code[i]
            others = current_code[:i] + current_code[i+1:]
            
            # Try to find a replacement
            found = False
            # Try a few random replacements
            for _ in range(10):
                new_w = random.getrandbits(N)
                if new_w == old_w:
                    continue
                if is_valid(new_w, others):
                    current_code[i] = new_w
                    no_improve = 0
                    found = True
                    break
            
            if not found:
                no_improve += 1
                
            # 3. Perturbation if stuck
            if no_improve > max_no_improve:
                # Remove a random word to create space
                if len(current_code) > 1:
                    remove_idx = random.randint(0, len(current_code) - 1)
                    current_code.pop(remove_idx)
                    no_improve = 0
                else:
                    # Reset if empty
                    break

        return current_code

    # Strategy: Run multiple searches
    # 1. Start from the seed's 32 words
    # 2. Start from empty
    # 3. Start from a small random greedy set
    
    # Generate seed words (same as current program)
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            seed_words.append(word)
    
    # Run 1: Start with seed
    if time.time() < deadline - 10:
        res = run_search(seed_words, 10.0)
        if len(res) > len(best_code):
            best_code = res

    # Run 2-10: Start with empty or small random sets
    for i in range(10):
        if time.time() > deadline - 5:
            break
        
        # Random initialization: Greedy add for a short time
        init = []
        init_start = time.time()
        while time.time() - init_start < 0.5: # 0.5s to init
            w = random.getrandbits(N)
            if is_valid(w, init):
                init.append(w)
                if len(init) > 100: # Don't let init get too big to waste time
                    break
        
        # Run local search
        res = run_search(init, 5.0) # 5s per search
        if len(res) > len(best_code):
            best_code = res

    return best_code
