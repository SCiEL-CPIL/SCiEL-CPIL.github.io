import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin

    if q != 3 or n != 6 or radius != 1:
        # Fallback
        all_words = list(itertools.product(range(q), repeat=n))
        covered = set()
        code = []
        balls = []
        for w in all_words:
            ball = {w}
            for i in range(n):
                for d in range(q):
                    if d == w[i]: continue
                    nw = list(w)
                    nw[i] = d
                    ball.add(tuple(nw))
            balls.append(list(ball))
        
        uncovered = set(all_words)
        while uncovered and time.time() < deadline:
            best_word = None
            best_count = -1
            # Sample if too large? No, small enough
            for i, w in enumerate(all_words):
                if w in covered: continue
                count = len(balls[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = w
            if best_word is None: break
            code.append(list(best_word))
            idx = all_words.index(best_word)
            covered.update(balls[idx])
            uncovered.difference_update(balls[idx])
        return code

    # Specific optimization for q=3, n=6, r=1
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    def word_to_idx(w):
        idx = 0
        for i in range(6):
            idx = idx * 3 + w[i]
        return idx
        
    def idx_to_word(idx):
        w = [0]*6
        for i in range(5, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    all_words_idx = list(range(729))
    balls = {}
    for idx in all_words_idx:
        w = idx_to_word(idx)
        ball = [idx]
        for i in range(6):
            for d in range(3):
                if d == w[i]: continue
                nw = w.copy()
                nw[i] = d
                ball.append(word_to_idx(nw))
        balls[idx] = ball

    best_code_size = len(initial_code)
    best_active = set(word_to_idx(w) for w in initial_code)
    
    # Try multiple random removal orders
    for attempt in range(10):
        if time.time() > deadline: break
        
        active = set(best_active) # Start from best found so far? Or initial?
        # Starting from initial is safer to find global structure, but starting from best is faster.
        # Let's start from initial each time to avoid bias?
        # Actually, peeling from initial is deterministic if order is fixed.
        # Randomizing order helps.
        
        active = set(word_to_idx(w) for w in initial_code)
        counts = np.zeros(729, dtype=np.int32)
        for ci in active:
            for neighbor in balls[ci]:
                counts[neighbor] += 1
                
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Randomize order
            items = list(active)
            random.shuffle(items)
            for ci in items:
                if ci not in active: continue
                redundant = True
                for neighbor in balls[ci]:
                    if counts[neighbor] <= 1:
                        redundant = False
                        break
                if redundant:
                    active.remove(ci)
                    for neighbor in balls[ci]:
                        counts[neighbor] -= 1
                    changed = True
                    
        if len(active) < best_code_size:
            best_code_size = len(active)
            best_active = active.copy()
            
    result = [idx_to_word(ci) for ci in best_active]
    return result
