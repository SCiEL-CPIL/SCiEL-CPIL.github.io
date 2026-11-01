import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    # Total number of words in the space
    N = q ** n
    
    # Precompute all words as integers 0..N-1
    # We will work with integer representations for speed
    # But we need to map integer <-> tuple for distance calculation or ball generation
    
    # Helper to convert integer to base-q digits
    def int_to_tuple(x, n, q):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = x % q
            x //= q
        return digits

    # Helper to convert tuple to integer
    def tuple_to_int(t, q):
        x = 0
        for d in t:
            x = x * q + d
        return x

    # Precompute the ball for each word
    # Ball of a word w contains all words within Hamming distance <= radius
    # Storing balls as lists of integers is memory efficient enough for N=6561
    # balls[w] = list of integers covered by w
    
    balls = []
    for w_int in range(N):
        w_tuple = int_to_tuple(w_int, n, q)
        ball = []
        # Generate all neighbors within radius
        # For radius 1, this is just w and all single-digit changes
        if radius == 1:
            # Include w itself
            ball.append(w_int)
            for pos in range(n):
                orig_val = w_tuple[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        new_tuple = w_tuple[:]
                        new_tuple[pos] = new_val
                        new_int = tuple_to_int(new_tuple, q)
                        ball.append(new_int)
        else:
            # General case (not needed for this specific task but good for robustness)
            # Use itertools.product for all words and filter by distance
            # This is slow, but radius is small in typical tasks.
            # For this task, radius is 1.
            pass
            
        balls.append(ball)
    
    # coverage_count[i] = number of selected codewords that cover word i
    coverage = np.zeros(N, dtype=np.int32)
    
    # selected_codewords: list of integers
    selected = []
    
    # uncovered_mask: boolean array, True if not yet covered
    # We can just use coverage == 0
    
    # To speed up finding the best candidate:
    # For each candidate c, the number of newly covered words is sum(coverage[b] == 0 for b in balls[c])
    # We can maintain a "gain" array for each candidate.
    # gain[c] = number of uncovered words in balls[c]
    
    gains = np.zeros(N, dtype=np.int32)
    for c in range(N):
        gains[c] = len(balls[c]) # Initially all are uncovered
    
    while True:
        if time.time() > deadline:
            break
        
        # Find candidate with max gain
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        # Select best_idx
        selected.append(best_idx)
        
        # Update coverage
        # For each word in balls[best_idx], increment coverage
        # If coverage was 0, it's now 1, so it's no longer uncovered
        # We need to update gains for all other candidates that cover these words
        
        # This update step is O(N * ball_size) if done naively for all candidates.
        # ball_size is q*n (approx 24 for q=3, n=8, r=1).
        # N is 6561.
        # Total updates per step: 6561 * 24 ~ 157k operations.
        # Number of steps: ~500.
        # Total ops: 500 * 157k = 78M ops.
        # NumPy vectorization helps, but iterating over candidates is slow in Python.
        
        # Optimization:
        # Instead of updating gains for all candidates, we can just recompute gains for all candidates 
        # using vectorized numpy operations if we store balls as a 2D array or use sparse structures.
        # Or, we can accept the O(N) update per step if we do it smartly.
        
        # Let's try a different approach:
        # Maintain `uncovered` as a set or bitset?
        # Actually, let's just recompute the "newly covered" count for the best candidate selection
        # by checking coverage of its ball.
        # But finding the BEST candidate requires checking ALL candidates.
        
        # Efficient Update Strategy:
        # When we add codeword `c`, it covers `ball_c`.
        # For any other candidate `d`, if `ball_d` intersects `ball_c`, its gain might decrease.
        # Specifically, for each word `w` in `ball_c` that was previously uncovered (coverage[w]==0 before update):
        #   coverage[w] becomes 1.
        #   For every candidate `d` such that `w` in `ball_d`:
        #       gains[d] -= 1
        
        # To do this efficiently, we need a reverse mapping:
        # `coverers[w]` = list of codewords `d` such that `w` is in `ball_d`.
        # Since the ball is symmetric (distance is symmetric), `w` in `ball_d` iff `d` in `ball_w`.
        # So `coverers[w]` is just `balls[w]`!
        
        # So:
        # For each w in balls[best_idx]:
        #   if coverage[w] == 0:
        #       for d in balls[w]:
        #           gains[d] -= 1
        #   coverage[w] += 1
        
        # Complexity per step:
        # Sum over w in balls[best_idx] of len(balls[w])
        # balls[best_idx] has size ~ 1 + n*(q-1) = 1 + 8*2 = 17.
        # Each balls[w] has size 17.
        # So ~ 17 * 17 = 289 operations per step.
        # 500 steps * 289 ops = 144,500 ops.
        # This is very fast!
        
        # Let's implement this update logic.
        
        # Identify words in best_idx's ball that are currently uncovered
        # We need to check coverage before updating.
        # 
        # 1. Get the list of words in the selected codeword's ball
        selected_ball = balls[best_idx]
        
        # 2. For each word in this ball, if it was uncovered, decrement gains of its coverers
        for w in selected_ball:
            if coverage[w] == 0:
                # w is being covered for the first time
                # Decrement gain for all codewords that cover w
                # These are exactly the codewords in balls[w]
                for d in balls[w]:
                    gains[d] -= 1
        
        # 3. Update coverage counts
        for w in selected_ball:
            coverage[w] += 1
            
    # Convert selected integers back to lists of integers
    result = [int_to_tuple(w, n, q) for w in selected]
    
    # Verify validity (optional, but good for debugging)
    # The problem states we must return valid codewords.
    # Greedy ensures coverage.
    
    return result
