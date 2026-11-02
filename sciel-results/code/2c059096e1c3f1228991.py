import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0

    total_words = q ** n

    def int_to_word(x):
        w = [0] * n
        for i in range(n):
            w[i] = x % q
            x //= q
        return w

    def word_to_int(w):
        x = 0
        for i in range(n):
            x = x * q + w[i]
        return x

    # Precompute balls
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

    # Specific optimization for the known hard case
    if q == 3 and n == 6 and radius == 1:
        # Seed: Direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                seed_codewords.append(h + list(tail))
        
        seed_ints = [word_to_int(w) for w in seed_codewords]
        
        # Verify coverage (should be true)
        covered = np.zeros(total_words, dtype=bool)
        for c in seed_ints:
            covered[balls[c]] = True
            
        if covered.all():
            current_codewords = list(seed_ints)
            
            # Pruning Loop
            changed = True
            while changed and time.time() < deadline:
                changed = False
                new_codewords = []
                for c in current_codewords:
                    # Check if c is redundant
                    # c is redundant if ball(c) is subset of union(ball(c') for c' in current \ {c})
                    redundant = True
                    for w in balls[c]:
                        found_cover = False
                        for c2 in current_codewords:
                            if c2 == c:
                                continue
                            # Check if w is in ball(c2)
                            # Since R is small, linear scan is ok
                            if w in balls[c2]:
                                found_cover = True
                                break
                        if not found_cover:
                            redundant = False
                            break
                    
                    if redundant:
                        changed = True
                    else:
                        new_codewords.append(c)
                
                current_codewords = new_codewords
            
            return [int_to_word(c) for c in current_codewords]

    # General Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    selected = []
    active_candidates = list(range(total_words))
    
    while not covered.all() and time.time() < deadline:
        best_candidate = -1
        best_gain = -1
        
        for c in active_candidates:
            gain = 0
            for w in balls[c]:
                if not covered[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
                if gain == len(balls[c]):
                    break
        
        if best_gain <= 0:
            break
        
        selected.append(best_candidate)
        covered[balls[best_candidate]] = True
        active_candidates.remove(best_candidate)
        
    return [int_to_word(c) for c in selected]
