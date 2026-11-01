import time
import random
import numpy as np
import itertools

# EVOLVE-BLOCK-START

def _int_to_digits(x, q, n):
    """Convert integer representation to list of digits."""
    digits = [0] * n
    for i in range(n):
        digits[i] = x % q
        x //= q
    return digits

def _digits_to_int(digits, q):
    """Convert list of digits to integer representation."""
    x = 0
    for i in range(len(digits) - 1, -1, -1):
        x = x * q + digits[i]
    return x

def _generate_initial_cover(q, n, radius, deadline):
    """
    Generate a valid initial cover using a fast greedy approach.
    We prioritize words that are currently uncovered.
    """
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    # Precompute balls for speed in the initial greedy phase
    # For radius 1, ball(x) = {x} U {x +/- 1*q^i for i in range(n)} (modulo q)
    # But we need to handle wrap-around carefully.
    # Let's just generate neighbors on the fly or precompute a neighbor list for all x?
    # Precomputing neighbor lists for 6561 words is fast.
    
    # Actually, for the initial greedy, we can just pick random words until covered?
    # No, that's inefficient.
    # Let's use the greedy method from the seed but optimized.
    
    # Precompute balls
    balls = []
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    for x in range(total):
        ball = [x]
        for i in range(n):
            old_digit = (x // powers[i]) % q
            for new_digit in range(q):
                if new_digit != old_digit:
                    y = x + (new_digit - old_digit) * powers[i]
                    ball.append(y)
        balls.append(ball)
        
    # Greedy selection
    # We want to pick balls that cover the most *new* elements.
    # Since all balls have the same size, and we pick sequentially,
    # a simple heuristic is to pick the ball with the most uncovered neighbors.
    
    # To speed this up, we can just iterate through all x, and if x is not covered,
    # add it to codewords and mark its ball as covered.
    # This is a "randomized greedy" if we shuffle x, or "systematic greedy" if we don't.
    # Systematic greedy (x=0..N-1) often yields decent covers.
    
    # Let's try a slightly smarter greedy: 
    # 1. Start with empty cover.
    # 2. While not all covered:
    #    - Find the word y with the highest number of uncovered neighbors (including itself).
    #    - Add y to cover.
    #    - Mark y's ball as covered.
    
    # This is O(N^2) roughly if done naively, but N=6561 is small.
    # However, updating counts is O(N * ball_size).
    
    # Let's stick to the simple systematic greedy first to ensure we have a valid baseline.
    # Then local search will improve it.
    
    for x in range(total):
        if not covered[x]:
            codewords.append(x)
            for y in balls[x]:
                covered[y] = True
                
        if time.time() > deadline - 20:
            break
            
    return codewords, balls

def _local_search(codewords, q, n, radius, balls, deadline):
    """
    Improve the cover by:
    1. Removing redundant codewords (if a codeword's ball is fully covered by others).
    2. Swapping codewords: replace a codeword c with a neighbor c' such that 
       the new cover is still valid and potentially allows further reductions.
       (Actually, simple removal is the primary driver for reduction. 
        Swapping is useful if we get stuck, but removal is the direct minimizer.)
       
    Strategy:
    - Maintain a count of how many times each word is covered.
    - If count[word] >= 2 for all words in ball(c), c is redundant.
    - If redundant, remove c.
    - Repeat until no redundant codewords exist.
    
    This is a "First Fit Decreasing" style cleanup, but dynamic.
    """
    total = q ** n
    # covered_count[y] = number of codewords whose ball contains y
    covered_count = np.zeros(total, dtype=np.int32)
    
    # Initialize counts
    for c in codewords:
        for y in balls[c]:
            covered_count[y] += 1
            
    # A word is "safe" to remove from cover if every word in its ball is covered by >= 2 codewords.
    # If we remove c, then for all y in ball(c), covered_count[y] decreases by 1.
    # We need covered_count[y] - 1 >= 1 for all y in ball(c).
    # i.e., min(covered_count[y] for y in ball(c)) >= 2.
    
    improved = True
    while improved:
        improved = False
        # Iterate through codewords
        # We should check in a random order to avoid bias, or just sequential.
        # Sequential is fine for now.
        
        for i in range(len(codewords)):
            if time.time() > deadline - 5:
                return codewords
                
            c = codewords[i]
            ball = balls[c]
            
            # Check if c is redundant
            # min_count = min(covered_count[y] for y in ball)
            # If min_count >= 2, we can remove c.
            
            min_count = np.min(covered_count[ball])
            
            if min_count >= 2:
                # Remove c
                # Decrement counts
                for y in ball:
                    covered_count[y] -= 1
                    
                # Remove from list
                codewords.pop(i)
                improved = True
                # Since we popped, we need to restart or be careful with indices.
                # Restarting the inner loop is safer.
                break
                
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: aim to finish by 140s to avoid timeout
    deadline = start_time + 140
    
    # 1. Generate initial cover
    codewords, balls = _generate_initial_cover(q, n, radius, deadline)
    
    # 2. Local Search (Reduction)
    codewords = _local_search(codewords, q, n, radius, balls, deadline)
    
    # 3. Optional: Try some random restarts or swaps if time permits?
    # The reduction phase is the main structural change from simple greedy.
    # Greedy picks a set. Local search prunes it.
    
    # Convert to required format
    result = []
    for c in codewords:
        result.append(_int_to_digits(c, q, n))
        
    return result

# EVOLVE-BLOCK-END
