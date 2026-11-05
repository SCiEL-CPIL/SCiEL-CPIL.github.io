import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Precompute the space
    space_size = q ** n
    # Generate all words as integers 0..space_size-1
    # Convert integer to tuple of digits
    def int_to_word(val):
        return [val // (q ** (n - i - 1)) % q for i in range(n)]
    
    # Precompute the ball of radius 1 for each word
    # ball[i] is a list of indices in 0..space_size-1 that are within distance 1 of int_to_word(i)
    balls = []
    for i in range(space_size):
        w = int_to_word(i)
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                # Convert back to int
                new_val = 0
                for j in range(n):
                    new_val = new_val * q + new_w[j]
                neighbors.append(new_val)
        balls.append(neighbors)
        
    # Convert balls to a numpy array for fast summing?
    # Actually, we just need to count how many are already covered.
    # We can maintain a boolean array 'covered' of size space_size.
    # For each candidate c, count sum(covered[balls[c]]).
    
    best_code = None
    best_size = space_size + 1
    
    # Run multiple restarts
    num_restarts = 200 # Adjust based on speed
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        covered = np.zeros(space_size, dtype=bool)
        code = []
        
        # Greedy loop
        while True:
            if time.time() > deadline - 5:
                break
                
            # Find candidate with min covered count in its ball
            # We can vectorize this if we structure data well, 
            # but space_size is small (729), so a loop might be fast enough if optimized.
            
            min_covered = space_size + 1
            best_candidate = -1
            best_new_cover = 0
            
            # To speed up, we can sample or use a smarter heuristic?
            # With 729 candidates, checking all is 729 * avg_ball_size operations.
            # avg_ball_size = 1 + n*(q-1) = 1 + 6*2 = 13.
            # 729 * 13 ~ 9500 operations per step.
            # 73 steps * 9500 ~ 700k operations per restart.
            # 200 restarts * 700k = 140M operations.
            # Python loops are slow. We need NumPy.
            
            # Let's precompute a matrix M where M[i, j] = 1 if j is in ball(i)
            # Then covered_count[i] = M[i] @ covered
            # This is a matrix-vector multiplication.
            
            # Initialize M
            if restart == 0:
                M = np.zeros((space_size, space_size), dtype=np.int8)
                for i in range(space_size):
                    for j in balls[i]:
                        M[i, j] = 1
                
            covered_counts = M @ covered
            # We want to minimize covered_counts, which maximizes new coverage
            # Tie-breaking: randomize indices where covered_counts is minimal
            
            min_val = np.min(covered_counts)
            candidates = np.where(covered_counts == min_val)[0]
            
            if len(candidates) == 0:
                break
                
            # Pick one randomly
            idx = np.random.choice(candidates)
            
            code.append(idx)
            
            # Update covered
            covered[balls[idx]] = True
            
            if np.all(covered):
                break
                
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
        # Early stop if we found the theoretical lower bound or known best
        if best_size <= 73:
            break

    if best_code is None:
        # Fallback to a simple construction if something went wrong
        # Direct sum from seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert integer indices back to lists of digits
    return [int_to_word(c) for c in best_code]
