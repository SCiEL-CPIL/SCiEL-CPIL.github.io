import random
import time
import numpy as np

def construct(q, n, radius):
    # Task specific: q=3, n=7, radius=1
    # General fallback for other parameters if needed, but optimized for the specific case
    if q != 3 or n != 7 or radius != 1:
        # Fallback to the original direct sum for safety, though task implies specific case
        import itertools
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial covering: all words
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = 120.0 # 120s limit, keep margin
    
    total_words = 3**n
    # Map word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Pre-compute balls for all words
    # Ball of a word w: w itself + all words at distance 1
    # Distance 1 means changing one coordinate to one of the other 2 values
    balls = []
    for i in range(total_words):
        w = idx_to_word(i)
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    ball.append(word_to_idx(new_w))
        balls.append(ball)
    
    # Convert balls to numpy arrays for faster summing if needed, 
    # but list of lists is fine for N=2187.
    
    best_code = None
    best_len = float('inf')
    
    # Run multiple random restarts of the greedy algorithm
    # Greedy: pick word with max number of uncovered neighbors
    for attempt in range(20): # 20 restarts should be fast enough
        if time.time() > deadline - 5.0:
            break
            
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        
        while not covered.all():
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_idx = -1
            # To break ties randomly, we can shuffle the range or just pick first max
            # Shuffling 2187 items is cheap.
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for idx in candidates:
                # Count uncovered neighbors
                # balls[idx] is a list of indices
                # We need count of j in balls[idx] where covered[j] is False
                # Using numpy: (~covered[balls[idx]]).sum()
                gain = (~covered[balls[idx]]).sum()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_gain == 0:
                # Should not happen if not covered.all()
                break
                
            code_indices.append(best_idx)
            # Mark covered
            covered[balls[best_idx]] = True
            
        if covered.all():
            if len(code_indices) < best_len:
                best_len = len(code_indices)
                best_code = code_indices
                # If we hit the theoretical lower bound or known optimum, we can stop
                if best_len <= 186:
                    break

    if best_code is None:
        # Fallback if something went wrong or time ran out
        # Return the direct sum construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [idx_to_word(idx) for idx in best_code]
