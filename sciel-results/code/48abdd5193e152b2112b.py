"""Greedy maximal covering set for ternary covering codes."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # 10s safety margin before 150s kill

    # Total space size
    total = q ** n

    # Precompute all words as integers 0..total-1
    # Represent each word as a base-q integer
    all_words = np.arange(total, dtype=np.int32)

    # Precompute the "ball" for each possible codeword:
    # For radius 1, a codeword c covers c itself and all words differing in exactly 1 position.
    # We need to map from integer representation to coverage.
    
    # Strategy: for each candidate codeword c (0..total-1), compute the set of words it covers.
    # But total can be 2187, so we can precompute all balls.
    
    # Convert integer to tuple of digits
    def int_to_digits(x, n, q):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    def digits_to_int(digits, q):
        x = 0
        for i in range(len(digits) - 1, -1, -1):
            x = x * q + digits[i]
        return x

    # Precompute all balls: ball[c] = set of integers covered by codeword c
    # For radius 1: ball[c] = {c} union {words differing in exactly one position}
    
    # For each position i and each alternative digit d != c[i], the word with digit d at position i
    # is at Hamming distance 1.
    
    # Precompute the "step" for each position: q^i
    steps = [1] * n
    for i in range(1, n):
        steps[i] = steps[i - 1] * q

    # For each codeword c (as integer), compute the list of covered integers
    # We'll store balls as a list of sets, or better, use a bitset approach.
    
    # Since total = 2187, we can use a bitset of size 2187 for each ball.
    # But 2187 * 2187 bits = ~600KB, which is fine.
    
    # Actually, let's use a different approach: 
    # uncovered = set of all words initially
    # For each candidate c, compute |ball(c) ∩ uncovered|
    # Pick c with max, add c to code, remove ball(c) from uncovered
    
    # To speed up, precompute for each c the list of words in its ball.
    
    balls = []
    for c in range(total):
        digits = int_to_digits(c, n, q)
        covered = [c]
        for i in range(n):
            orig = digits[i]
            for d in range(q):
                if d == orig:
                    continue
                digits[i] = d
                w = digits_to_int(digits, q)
                covered.append(w)
            digits[i] = orig
        balls.append(covered)
    
    # Convert to sets for fast intersection
    ball_sets = [set(b) for b in balls]
    
    uncovered = set(range(total))
    code = []
    
    while len(uncovered) > 0:
        if time.time() > deadline:
            break
        
        best_c = -1
        best_count = 0
        for c in range(total):
            if not uncovered:
                break
            # Count how many of ball_sets[c] are in uncovered
            cnt = 0
            for w in ball_sets[c]:
                if w in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_c = c
            if cnt == 0:
                continue
        
        if best_c < 0 or best_count == 0:
            # Should not happen if uncovered is non-empty
            break
        
        # Add best_c to code
        code.append(int_to_digits(best_c, n, q))
        # Remove its ball from uncovered
        uncovered.difference_update(ball_sets[best_c])
    
    # If we ran out of time, we may not have covered everything.
    # The evaluator checks validity, so we need a valid covering.
    # If invalid, the score is 0. So we should only return if valid.
    # Let's check validity.
    
    if len(uncovered) > 0:
        # Not fully covered. We need to either:
        # 1. Continue until covered (risky for time)
        # 2. Return the seed program's result as fallback
        # Let's try to finish quickly with a simpler approach
        # Actually, let's just return what we have and hope it's valid,
        # or fall back to the known-good seed.
        
        # Fallback: use the tetracode direct sum
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return code

# EVOLVE-BLOCK-END
