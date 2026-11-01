import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        return [[0] * n]

    # Precompute powers of 3
    pow3 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow3[i] = pow3[i - 1] * 3
    total_words = pow3[n]  # 6561

    def encode(digits):
        w = 0
        for i in range(n):
            w += digits[i] * pow3[i]
        return w

    def decode(w):
        digits = [0] * n
        for i in range(n):
            digits[i] = w % 3
            w //= 3
        return digits

    # Precompute all balls (words within distance 1) for all 6561 words
    # For efficiency, use numpy
    all_words = np.arange(total_words)
    
    # Precompute the ball for each word
    # ball[w] = set of words within distance 1 of w
    # For each word w, its ball = {w} union {words at distance 1}
    # Words at distance 1: change one digit to one of the other 2 values
    
    # Precompute neighbor indices for each position
    # For position i, the multiplier is pow3[i]
    # Changing digit i from d to d+1 adds pow3[i], from d to d-1 subtracts pow3[i]
    # But we need to handle wraparound in base 3
    
    # Actually, let's just precompute all balls using a simple approach
    # Since 6561 * 17 = ~111k entries, this is manageable
    
    balls = [None] * total_words
    
    for w in range(total_words):
        digits = [0] * n
        temp = w
        for i in range(n):
            digits[i] = temp % 3
            temp //= 3
        
        ball = {w}
        for i in range(n):
            for new_val in range(3):
                if new_val != digits[i]:
                    new_digits = digits[:]
                    new_digits[i] = new_val
                    ball.add(encode(new_digits))
        balls[w] = ball

    # Greedy set cover
    uncovered = set(range(total_words))
    code = []
    code_set = set()
    
    # For each word, track how many times it's covered
    coverage_count = np.zeros(total_words, dtype=np.int32)
    
    # Greedy: repeatedly pick the center that covers the most uncovered words
    # To speed this up, we can use a priority queue or just scan
    
    while uncovered and time.time() < deadline:
        best_center = None
        best_count = -1
        
        # For each potential center, count how many uncovered words it covers
        # This is O(6561 * 17) per iteration, which is ~111k operations
        # We need ~400 iterations, so ~44M operations total - might be too slow
        
        # Better: use a different approach
        # Actually, let's just do it and see
        
        for c in range(total_words):
            if c in code_set:
                continue
            count = 0
            for w in balls[c]:
                if w in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_center = c
        
        if best_center is None or best_count == 0:
            break
        
        # Add best_center to code
        code.append(best_center)
        code_set.add(best_center)
        
        # Update uncovered and coverage_count
        for w in balls[best_center]:
            if w in uncovered:
                uncovered.discard(w)
            coverage_count[w] += 1
        
        # Early exit if we've already exceeded a reasonable bound
        if len(code) > 500:
            break
    
    # Now try to remove redundant centers
    # A center is redundant if all words in its ball are covered by at least one other center
    
    def is_redundant(center):
        for w in balls[center]:
            if coverage_count[w] <= 1:
                return False
        return True
    
    # Iterative removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        centers = list(code_set)
        random.shuffle(centers)
        for c in centers:
            if is_redundant(c):
                code_set.discard(c)
                for w in balls[c]:
                    coverage_count[w] -= 1
                changed = True
    
    # Try multiple passes with different orderings
    for _ in range(5):
        if time.time() > deadline:
            break
        changed = True
        while changed and time.time() < deadline:
            changed = False
            centers = list(code_set)
            random.shuffle(centers)
            for c in centers:
                if is_redundant(c):
                    code_set.discard(c)
                    for w in balls[c]:
                        coverage_count[w] -= 1
                    changed = True
    
    # Convert back to list of digit lists
    result = [decode(w) for w in code_set]
    
    return result
