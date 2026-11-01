"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _time_remaining(deadline, margin=5.0):
    return deadline - time.time() - margin > 0

def construct(q, n, radius):
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    if q == 3 and n == 8 and radius == 1:
        # Best known: 486 words. Try to find a covering with <= 486 words.
        # Strategy: greedy set cover with efficient numpy operations.
        
        # Total space size
        total = q ** n  # 6561
        
        # Represent words as integers 0..6560
        # For each codeword, we need to mark all words within distance <= radius.
        
        # Precompute: for a given word (as int), what are all words within distance 1?
        # We can precompute the "ball" for each possible word, but that's 6561 * 25 entries.
        
        # Actually, let's use a different approach:
        # Greedy: at each step, pick the codeword that covers the most uncovered words.
        
        # Represent uncovered set as a boolean array of size 6561
        uncovered = np.ones(total, dtype=bool)
        
        codewords = []
        
        # Precompute digit representations
        # word_int to digits
        # For speed, precompute all balls (set of words within distance 1 of each word)
        # ball[i] = list of word indices within distance 1 of word i
        
        # Precompute balls
        # For word i, digits are [i // 3^7, ..., i % 3]
        # Words within distance 1: change one digit to one of 2 other values
        # So each ball has 1 + 8*2 = 17 words
        
        # Let's precompute the ball for each word as a numpy array
        # balls[i] = array of word indices in ball(i)
        
        # Precompute powers of 3
        pow3 = [1]
        for i in range(n):
            pow3.append(pow3[-1] * q)
        # pow3[i] = q^i, so digit j (0-indexed from left) has weight pow3[n-1-j]
        
        # Precompute digit values for each word
        # digits[word][j] = digit at position j
        # This is 6561 x 8 array of uint8
        digits = np.zeros((total, n), dtype=np.uint8)
        for w in range(total):
            for j in range(n):
                digits[w, j] = (w // pow3[n-1-j]) % q
        
        # Precompute balls: for each word, the list of words within distance 1
        # ball[w] = np.array of word indices
        # We can compute this efficiently:
        # For word w, for each position j, for each new digit d != digits[w,j]:
        #   new_w = w + (d - digits[w,j]) * pow3[n-1-j]
        
        # Store balls as list of arrays
        balls = []
        for w in range(total):
            ball = [w]
            for j in range(n):
                cur_d = digits[w, j]
                weight = pow3[n-1-j]
                for d in range(q):
                    if d != cur_d:
                        ball.append(w + (d - cur_d) * weight)
            balls.append(np.array(ball, dtype=np.int32))
        
        # Greedy set cover
        # At each step, find the word that maximizes |ball(w) ∩ uncovered|
        # This is expensive: 6561 * 17 operations per step, but we have 6561 steps worst case.
        # We need a smarter approach.
        
        # Alternative: use a priority queue or approximate method.
        # For now, let's do a simple greedy with numpy vectorization.
        
        # Actually, for 6561 words and 17 per ball, a simple greedy might be too slow in Python.
        # Let's try a different approach: use the known structure.
        
        # The tetracode construction gives 9 * 3^4 = 729 words.
        # We want to do better, ideally close to 486.
        
        # Let's try a randomized greedy with local search.
        
        # Approach: 
        # 1. Start with a random set of words
        # 2. Greedily add words that cover the most uncovered
        # 3. Try to remove redundant words
        # 4. Local search: swap words to reduce count
        
        # Let's try a simpler approach first: pure greedy with numpy acceleration.
        
        # For speed, let's use a different representation:
        # uncovered as a set (Python set) for membership testing
        # But numpy is faster for bulk operations.
        
        # Let's try: maintain uncovered as a boolean numpy array.
        # For each candidate word w, count uncovered words in ball(w).
        # This requires summing uncovered[ball[w]] for each w, which is 6561 * 17 = ~112k ops per step.
        # With ~500 steps, that's ~56M ops. In numpy, this might be feasible.
        
        # Let's precompute balls as a 2D array where possible, but balls have variable... no, all have size 17.
        # balls_2d = np.array(balls)  # shape (6561, 17)
        
        balls_2d = np.array(balls, dtype=np.int32)  # (6561, 17)
        
        # Greedy
        uncovered = np.ones(total, dtype=bool)
        codewords = []
        
        while uncovered.any() and _time_remaining(deadline):
            # For each word w, compute how many uncovered words are in ball(w)
            # counts[w] = uncovered[balls_2d[w]].sum()
            # This can be done as: counts = balls_2d @ uncovered (as float) ... no, that's not right.
            # Actually: counts[w] = sum of uncovered[balls_2d[w, j]] for j in range(17)
            # We can do: counts = np.array([uncovered[b].sum() for b in balls_2d]) but that's slow.
            # Better: use matrix multiplication? No, uncovered is boolean.
            # Let's convert uncovered to float and do: counts = balls_2d @ ... no.
            
            # Actually: if we think of it as: for each position in balls_2d, we look up uncovered.
            # counts = uncovered[balls_2d].reshape(total, 17).sum(axis=1)
            
            counts = uncovered[balls_2d].reshape(total, q + n*(q-1)).sum(axis=1)
            
            # Pick the word with max count
            best_w = np.argmax(counts)
            
            if counts[best_w] == 0:
                break
            
            codewords.append(best_w)
            # Mark ball(best_w) as covered
            uncovered[balls_2d[best_w]] = False
        
        # Convert to list of digit lists
        result = []
        for w in codewords:
            word = []
            for j in range(n):
                word.append((w // pow3[n-1-j]) % q)
            result.append(word)
        
        # If we ran out of time, return what we have (might not be a valid cover)
        # Check if valid
        if uncovered.any():
            # Not a valid cover. Fall back to the tetracode construction.
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        return result
    
    # Generic fallback: tetracode-style construction for q=3
    if q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # For n < 4, use all words
            return [list(t) for t in itertools.product(range(3), repeat=n)]
    
    # Very generic fallback
    return [list(t) for t in itertools.product(range(q), repeat=n)]

# EVOLVE-BLOCK-END
