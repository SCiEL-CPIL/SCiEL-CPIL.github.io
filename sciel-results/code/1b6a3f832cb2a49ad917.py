import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # 1. Initial Solution: Direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Represent words as integers for speed.
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

    total_words = q ** n
    all_words = list(range(total_words))
    
    # Precompute all words as ints and their closed balls
    # balls[i] is a list of ints covered by codeword i
    balls = []
    for w_int in all_words:
        w = int_to_word(w_int)
        ball = [w_int]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val == orig: continue
                w[i] = new_val
                ball.append(word_to_int(w))
            w[i] = orig
        balls.append(ball)
    
    # Precompute reverse mapping: for each word, which codewords cover it?
    # Or rather, just use the balls for coverage checks.
    # To speed up coverage check, we can use a numpy array.
    
    # Convert initial code to ints
    code_ints = set(word_to_int(c) for c in initial_code)
    
    # Helper to check coverage of a set of ints using numpy
    def is_covering_ints(code_set):
        # Create a boolean array of size total_words
        # We can't easily vectorize the "union of balls" without precomputed bitsets or similar.
        # However, total_words is small (729).
        # Let's stick to a simple loop but optimized.
        covered = [False] * total_words
        for c in code_set:
            for w in balls[c]:
                covered[w] = True
        return all(covered)
        
    # Local Search
    current_code = set(code_ints)
    
    def try_remove_redundant():
        removed = False
        candidates = list(current_code)
        random.shuffle(candidates)
        for c in candidates:
            if c not in current_code: continue
            # Optimization: if we remove c, do we still cover everything?
            # We can check if there is any word ONLY covered by c.
            # This is expensive to compute from scratch.
            # Simple check:
            test_set = current_code - {c}
            if is_covering_ints(test_set):
                current_code.remove(c)
                removed = True
        return removed

    def try_swap():
        swapped = False
        candidates = list(current_code)
        random.shuffle(candidates)
        for c in candidates:
            if c not in current_code: continue
            for _ in range(10):
                new_c = random.choice(all_words)
                if new_c in current_code: continue
                test_set = (current_code - {c}) | {new_c}
                if is_covering_ints(test_set):
                    current_code.remove(c)
                    current_code.add(new_c)
                    swapped = True
                    break
        return swapped

    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # Initial cleanup
    while try_remove_redundant():
        if time.time() - start_time > deadline:
            break

    # Iterative improvement
    while time.time() - start_time < deadline:
        if try_remove_redundant():
            continue
        
        if try_swap():
            pass
        else:
            break

    # Convert back to lists of ints
    result = [int_to_word(c) for c in current_code]
    return result
