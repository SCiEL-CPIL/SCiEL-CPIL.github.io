import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    # Calculate space size
    space_size = q ** n
    
    # If space is too large, fallback to product construction or similar
    # 10^7 is roughly the limit for fast python enumeration with this logic
    if space_size > 2000000:
        # Fallback: Direct sum of tetracode if q=3 and n>=4
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # This creates 9 words of length 4
            # Extend to length n by appending all combinations of length n-4
            # This is what the seed did.
            # But this is O(3^(n-4)) * 9.
            # If n is large, this is still huge.
            # However, for the specific task, n=8.
            # If n is large, we might just return a random set or a structured one.
            # Given the context, n is likely small.
            pass
        # For now, if space is too big, we can't run greedy.
        # We'll try a simpler heuristic: Random Greedy
        # Pick random words, check if they cover new stuff.
        # But we can't check "new stuff" without enumerating neighbors.
        # So we assume space is small enough for the primary algorithm.
        # If not, we return the seed result to be safe.
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: Random codewords? No, must be valid.
            # Return empty? No, must cover.
            # Return all words? Valid but huge score.
            # Let's just return the seed logic if possible, else all words (worst case).
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precompute balls
    # Index mapping: tuple of digits <-> integer
    # We can use integer representation for speed.
    # word = d_0 * q^(n-1) + ... + d_{n-1}
    
    # Generate all words as tuples for distance calculation?
    # Or just work with integers and decode when needed?
    # Decoding is expensive.
    # Let's precompute the balls using integer arithmetic.
    
    # Function to get neighbors of an integer index within radius R
    # This is O(|B|) per index.
    # Total precomputation: O(N * |B|).
    
    # To compute neighbors of index i:
    # 1. Decode i to digits.
    # 2. For each position pos in 0..n-1:
    #    For each digit d in 0..q-1:
    #       If d != digits[pos]:
    #          Create new digits, change pos to d.
    #          Encode to int.
    #          Add to ball.
    # 3. Add i itself.
    
    # This is O(n * q) per index.
    # N * n * q.
    # For 3^8=6561, n=8, q=3: 6561 * 24 ~ 150k ops. Very fast.
    
    balls = []
    for i in range(space_size):
        # Decode
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits = digits[::-1] # Most significant first? Doesn't matter as long as consistent.
        
        ball = [i]
        # Generate neighbors
        # To avoid duplicates (if q=1? no q>=2 usually), and to be fast:
        # Just iterate positions and digits.
        for pos in range(n):
            orig_d = digits[pos]
            for new_d in range(q):
                if new_d == orig_d:
                    continue
                # Create new word
                # Instead of creating list, we can compute the integer difference
                # diff = (new_d - orig_d) * q^(n-1-pos)
                # new_i = i + diff
                # This is much faster.
                power = q ** (n - 1 - pos)
                new_i = i + (new_d - orig_d) * power
                ball.append(new_i)
        
        # Note: This generates all words at distance 1.
        # If radius > 1, we need more.
        # The task specifies radius 1 in the description.
        # The interface takes `radius`.
        # If radius > 1, the ball size grows.
        # For radius 1, the above is correct.
        # Let's assume radius is small (1 or 2).
        # If radius is larger, this simple generation is insufficient.
        # However, the "Football pool" is radius 1.
        # I will implement for radius 1 specifically for speed, 
        # and fall back to seed for radius > 1 or large space.
        
        if radius == 1:
            balls.append(ball)
        else:
            # If radius > 1, the precomputation is harder.
            # We'll stick to radius 1 optimization.
            # For radius > 1, we might just use the seed if it works, 
            # or a slower neighbor generation.
            # Given the prompt focuses on the ternary 8-match (radius 1), 
            # I'll optimize for that.
            balls.append(ball) # Incorrect for R>1, but we'll guard against it below.

    if radius > 1:
        # Fallback for R > 1 to be safe, or implement BFS for balls.
        # BFS for each i is O(|B|).
        # Let's just use the seed if R > 1 to avoid bugs.
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Greedy Algorithm
    covered = [False] * space_size
    uncovered_count = [len(balls[i]) for i in range(space_size)]
    
    codewords = []
    
    # Optimization: Use a list of (count, index) and find max?
    # Or just loop.
    # N=6561. Looping 6561 times is fast.
    # Number of iterations of outer loop ~ N / |B| ~ 6561 / 17 ~ 386.
    # Inner loop: find max (N), update (|B| * |B|).
    # Total ~ 386 * (6561 + 289) ~ 386 * 6850 ~ 2.6M ops.
    # Very fast.
    
    for _ in range(space_size): # Max possible codewords
        if time.time() > deadline:
            break
            
        # Find index with max uncovered_count
        best_i = -1
        best_count = -1
        for i in range(space_size):
            if uncovered_count[i] > best_count:
                best_count = uncovered_count[i]
                best_i = i
                
        if best_count == 0:
            break
            
        codewords.append(best_i)
        
        # Update coverage
        for j in balls[best_i]:
            if not covered[j]:
                covered[j] = True
                # Update counts for all k that cover j
                # k covers j if j is in balls[k]
                # This is equivalent to k is in balls[j] (symmetry)
                for k in balls[j]:
                    uncovered_count[k] -= 1
                    
    # Convert indices back to lists of digits
    result = []
    for i in codewords:
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        result.append(digits)
        
    return result
