import random
import time

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 55.0
    
    # Start with Reed-Muller code (32 words, distance 8)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    rng = random.Random(42)
    best_size = len(words)
    best_words = list(words)
    
    def can_add(w, code):
        for c in code:
            if (w ^ c).bit_count() < MIN_DIST:
                return False
        return True

    def find_conflicts(w, code):
        conflicts = []
        for i, c in enumerate(code):
            if (w ^ c).bit_count() < MIN_DIST:
                conflicts.append(i)
        return conflicts

    def try_add_smart(code, rng, attempts=50):
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                return False
            w = rng.getrandbits(N)
            conflicts = find_conflicts(w, code)
            if not conflicts:
                code.append(w)
                return True
            elif len(conflicts) == 1:
                idx = conflicts[0]
                code.pop(idx)
                code.append(w)
                return True
            elif len(conflicts) == 2 and rng.random() < 0.05:
                idx = rng.choice(conflicts)
                code.pop(idx)
                code.append(w)
                return True
        return False

    def try_flip(code, rng):
        idx = rng.randrange(len(code))
        bit = rng.randrange(N)
        new_w = code[idx] ^ (1 << bit)
        for j, c in enumerate(code):
            if j != idx and (new_w ^ c).bit_count() < MIN_DIST:
                return False
        code[idx] = new_w
        return True

    def kick(code, rng, k=3):
        if time.time() - start_time > time_limit:
            return
        removed = []
        for _ in range(min(k, len(code))):
            idx = rng.randrange(len(code))
            removed.append(code.pop(idx))
        for w in removed:
            if can_add(w, code):
                code.append(w)
        try_add_smart(code, rng)

    # Phase 1: Greedy add
    while time.time() - start_time < time_limit * 0.2:
        if not try_add_smart(words, rng):
            break
    
    if len(words) > best_size:
        best_size = len(words)
        best_words = list(words)
    
    # Phase 2: Local search
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit:
            break
        move = rng.random()
        if move < 0.4:
            try_add_smart(words, rng)
        elif move < 0.7:
            try_flip(words, rng)
        else:
            kick(words, rng, k=rng.randint(2, 5))
        
        if len(words) > best_size:
            best_size = len(words)
            best_words = list(words)
    
    return best_words
