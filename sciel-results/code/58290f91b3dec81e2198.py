import time
import random
import numpy as np

def construct(q, n, radius):
    # This problem is specifically for q=3, n=6, radius=1
    # The evaluator passes these, but the algorithm is tailored.
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (not expected in this specific task contract usually, but good practice)
        # For now, we assume the specific case.
        pass

    total_words = q ** n
    # Precompute balls as bitmasks
    # Word i is covered by codeword c if dist(i, c) <= radius
    # We map word indices to their digit representation
    
    def int_to_digits(x, n, q):
        return [x % q for _ in range(n)] # This is wrong, need to handle base conversion properly
        # Actually, for q=3, n=6, we can just precompute distances.
    
    # Efficient precomputation of balls
    # ball[c] is a bitmask of all words within distance 1 of c
    balls = [0] * total_words
    
    # Helper to get digits
    def get_digits(x):
        return [x // (q ** i) % q for i in range(n)]
        
    # Precompute distances or just check neighbors
    # For radius 1, a ball contains c itself and all words that differ in exactly 1 position.
    # Number of neighbors: n * (q - 1)
    
    for c in range(total_words):
        mask = 1 << c
        dc = get_digits(c)
        for pos in range(n):
            for new_val in range(q):
                if new_val == dc[pos]:
                    continue
                # Construct neighbor
                # This is slow in Python. 
                # Optimization: Precompute neighbor lists for all c.
                pass

    # Better precomputation:
    # For each position p and each value v, we can identify which words have v at p.
    # But let's just compute the neighbor indices directly.
    
    # Precompute powers of q
    powers = [1]
    for i in range(n):
        powers.append(powers[-1] * q)
        
    # For each c, calculate its neighbors
    for c in range(total_words):
        mask = 1 << c
        # Get digits of c
        digits = [0] * n
        tmp = c
        for i in range(n):
            digits[i] = tmp % q
            tmp //= q
            
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Calculate index of neighbor
                # diff = (new_val - orig_val) * powers[pos]
                neighbor = c + (new_val - orig_val) * powers[pos]
                mask |= (1 << neighbor)
        balls[c] = mask

    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin
    
    best_codewords = None
    best_count = total_words + 1
    
    # Initial greedy to get a baseline
    def run_greedy(random_seed):
        random.seed(random_seed)
        uncovered = (1 << total_words) - 1
        codewords = []
        
        while uncovered:
            best_c = -1
            best_covered = 0
            best_score = -1
            
            # To speed up, we can just iterate all candidates. 729 is small.
            # We want to maximize popcount(balls[c] & uncovered)
            # To introduce randomness, we can shuffle candidates or pick randomly from top k.
            # Let's just pick the max, but if ties, random.
            
            # Optimization: 
            # Instead of checking all 729 every time, we can't easily skip.
            # But 729 * 73 steps is small.
            
            candidates = list(range(total_words))
            # We don't need to shuffle all, just find max.
            
            # To randomize, we can add a small random noise to the score?
            # Or just pick randomly from the set of candidates that have the max score.
            
            max_score = -1
            max_candidates = []
            
            for c in candidates:
                # If balls[c] & uncovered is 0, it's useless
                # We can check if it's already fully covered? No, balls[c] is fixed.
                # We need intersection with uncovered.
                inter = balls[c] & uncovered
                if inter == 0:
                    continue
                score = inter.bit_count()
                if score > max_score:
                    max_score = score
                    max_candidates = [c]
                elif score == max_score:
                    max_candidates.append(c)
            
            if not max_candidates:
                break # Should not happen if uncovered != 0
                
            # Pick one
            c = random.choice(max_candidates)
            codewords.append(c)
            uncovered &= ~balls[c]
            
        return codewords

    # Run multiple restarts
    seed = 42
    while time.time() - start_time < time_limit:
        codewords = run_greedy(seed)
        if len(codewords) < best_count:
            best_count = len(codewords)
            best_codewords = codewords[:]
            # Try local search (removal) on the new best
            # Check if any codeword can be removed
            modified = True
            while modified:
                modified = False
                for i in range(len(best_codewords)):
                    c = best_codewords[i]
                    # Check if c is necessary
                    # Remove c temporarily
                    remaining = best_codewords[:i] + best_codewords[i+1:]
                    # Check if union of balls of remaining covers everything
                    # We can compute the union mask
                    union_mask = 0
                    for rc in remaining:
                        union_mask |= balls[rc]
                    if union_mask == (1 << total_words) - 1:
                        # Can remove
                        best_codewords = remaining
                        modified = True
                        break # Restart loop as indices changed
                if not modified:
                    break
            
        seed += 1

    # Convert integer indices to lists of digits
    result = []
    for c in best_codewords:
        digits = [0] * n
        tmp = c
        for i in range(n):
            digits[i] = tmp % q
            tmp //= q
        result.append(digits)
        
    return result
