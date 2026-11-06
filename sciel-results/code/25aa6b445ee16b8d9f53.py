import random
import time

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 85.0
    
    def can_add(w, code):
        for c in code:
            if (w ^ c).bit_count() < MIN_DIST:
                return False
        return True
    
    def make_reed_muller():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    def run_once(seed, time_budget):
        rng = random.Random(seed)
        code = make_reed_muller()
        deadline = time.time() + time_budget
        
        # Greedy add
        greedy_deadline = time.time() + time_budget * 0.3
        while time.time() < greedy_deadline:
            w = rng.getrandbits(N)
            if can_add(w, code):
                code.append(w)
        
        # Local search with swaps
        while time.time() < deadline:
            idx = rng.randrange(len(code))
            old_w = code[idx]
            code.pop(idx)
            found = False
            for _ in range(25):
                w = rng.getrandbits(N)
                if can_add(w, code):
                    code.append(w)
                    found = True
                    break
            if not found:
                code.insert(idx, old_w)
        
        return code
    
    best_words = make_reed_muller()
    best_size = len(best_words)
    
    seed = 42
    while time.time() - start_time < time_limit - 3.0:
        remaining = time_limit - (time.time() - start_time) - 2.0
        budget = min(15.0, remaining)
        if budget < 5.0:
            break
        code = run_once(seed, budget)
        if len(code) > best_size:
            best_size = len(code)
            best_words = list(code)
        seed += 1
    
    # Final polish: try to add more words to best
    if time.time() - start_time < time_limit - 0.5:
        rng = random.Random(seed)
        final = list(best_words)
        while time.time() < time_limit - 0.2:
            w = rng.getrandbits(N)
            if can_add(w, final):
                final.append(w)
        if len(final) > best_size:
            best_words = final
    
    return best_words
