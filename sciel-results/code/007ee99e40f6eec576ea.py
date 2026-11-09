import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Constructs a covering code using optimized NumPy-based Greedy Set Cover
    and Reduction.
    """
    start_time = time.time()
    timeout = 115  # Safety margin from 120s limit
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]
    
    total_words = 3 ** n
    n_words = total_words
    
    # Precompute digits
    digits = np.zeros((n_words, n), dtype=int)
    for i in range(n_words):
        temp = i
        for k in range(n):
            digits[i, k] = temp % q
            temp //= q
            
    # Precompute balls as indices
    # For q=3, n=7, radius=1, ball size is 1 + n*(q-1) = 15
    ball_size = 1 + n * (q - 1)
    balls = np.zeros((n_words, ball_size), dtype=int)
    
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    for i in range(n_words):
        balls[i, 0] = i
        idx = 1
        for pos in range(n):
            orig_digit = digits[i, pos]
            p = powers[pos]
            base_val = i - orig_digit * p
            for new_digit in range(q):
                if new_digit == orig_digit:
                    continue
                balls[i, idx] = base_val + new_digit * p
                idx += 1

    # Greedy Set Cover with NumPy
    uncovered = np.ones(n_words, dtype=bool)
    code = []
    
    # Precompute coverage masks for all words? 
    # That would be 2187 * 15 booleans, which is small.
    # But checking "how many uncovered" for all 2187 words at each step is O(N * BallSize).
    # Total steps ~ N / BallSize. Total ops ~ N^2 / BallSize.
    # 2187^2 / 15 ~ 300k ops. Very fast.
    
    # We can vectorize the count of uncovered neighbors for all candidates at once.
    # Create a matrix where M[i, j] is 1 if j is in ball[i].
    # Actually, we just need sum(uncovered[ball[i]]) for all i.
    
    # Let's use a sparse-like approach or just loop since N is small (2187).
    # But pure Python loop over 2187 items * ~150 iterations is 300k iterations.
    # Each iteration does a sum over 15 items. 4.5M operations.
    # NumPy vectorization is better.
    
    # Create a boolean matrix `covers` of shape (n_words, n_words) ? 
    # 2187 * 2187 * 8 bytes = ~38 MB. Acceptable.
    # covers[i, j] = 1 if j is in ball[i].
    covers = np.zeros((n_words, n_words), dtype=bool)
    for i in range(n_words):
        covers[i, balls[i]] = True
        
    # Now, at each step, we want argmax(sum(covers[:, j] * uncovered[j] for j in range(N)))
    # This is (covers @ uncovered).
    
    while np.any(uncovered):
        if time.time() - start_time > timeout:
            break
            
        # Count how many uncovered points each word covers
        # covers is (N, N) bool, uncovered is (N,) bool
        # Result is (N,) int
        counts = covers @ uncovered.astype(int)
        
        # Pick the word with max count
        # If multiple, pick first one (or random, but first is deterministic and fast)
        best_idx = np.argmax(counts)
        
        if counts[best_idx] == 0:
            break # Should not happen if uncovered is not empty and covers is valid
            
        code.append(best_idx)
        uncovered[balls[best_idx]] = False
        
    # Reduction Phase
    # Remove redundant words
    # A word c is redundant if for all j in ball[c], there exists another word c' in code (c' != c) such that j in ball[c'].
    # Equivalently, if we remove c, the union of balls of remaining words still covers ball[c].
    
    # We can iterate and remove.
    # To speed up, we can check if the "critical" points of c (points only covered by c) is empty.
    # Critical points of c = ball[c] \ (Union of ball[c'] for c' in code, c' != c)
    
    # We can maintain a count of how many times each point is covered.
    # count[j] = number of c in code such that j in ball[c].
    
    count_cover = np.zeros(n_words, dtype=int)
    for c in code:
        count_cover[balls[c]] += 1
        
    # Now, a word c is redundant if min(count_cover[balls[c]]) > 1.
    # If min is 1, it has a critical point and cannot be removed.
    
    changed = True
    while changed:
        if time.time() - start_time > timeout:
            break
        changed = False
        new_code = []
        for c in code:
            # Check if c is redundant
            # Get counts for points in ball[c]
            min_count = np.min(count_cover[balls[c]])
            if min_count > 1:
                # Redundant, remove it
                count_cover[balls[c]] -= 1
                changed = True
            else:
                new_code.append(c)
        code = new_code
        
    # Convert to list of lists
    result = [list(digits[c]) for c in code]
    return result
