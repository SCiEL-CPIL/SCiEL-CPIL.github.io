import numpy as np
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Generate all words and map to indices
    N = q ** n
    # Generate all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    # Map word to index (index is just the integer value in base q)
    # We can compute index directly: sum(w[i] * q^(n-1-i))
    # But for M construction, we need distance.
    
    # Precompute distances or balls
    # balls[i] is a list of indices j such that dist(word_i, word_j) <= radius
    # Since radius=1, ball size is 1 + n*(q-1)
    
    # To speed up M construction, we can compute indices directly.
    # Let's create an array `word_indices` where word_indices[i] is the base-q integer for word i.
    # Actually, iterating 0..N-1 and converting to base q is fast.
    
    # M[i, j] = 1 if dist(i, j) <= radius
    # We can compute this by checking digits.
    
    # Optimization: Precompute the "neighbors" for each word.
    # For each word i, generate all words at distance <= 1.
    # This is 1 + n*(q-1) words.
    
    # Let's create a list of lists `balls`
    balls = [None] * N
    for i in range(N):
        # Convert i to base q digits
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits = digits[::-1] # Most significant first
        
        # Generate all words within distance 1
        neighbors = set()
        # Distance 0
        neighbors.add(i)
        
        # Distance 1
        for pos in range(n):
            original_digit = digits[pos]
            for new_digit in range(q):
                if new_digit == original_digit:
                    continue
                new_digits = digits[:]
                new_digits[pos] = new_digit
                # Convert back to index
                new_idx = 0
                for d in new_digits:
                    new_idx = new_idx * q + d
                neighbors.add(new_idx)
        
        balls[i] = list(neighbors)
        
        if time.time() > deadline:
            break
            
    if time.time() > deadline:
        # Fallback to seed if precomputation takes too long (unlikely)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Construct M matrix (N x N) of int8
    M = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        M[i, balls[i]] = 1
        
    # Greedy Set Cover
    covered = np.zeros(N, dtype=np.int8)
    code = []
    
    while covered.sum() < N:
        if time.time() > deadline:
            break
            
        # scores[i] = number of currently uncovered words covered by candidate i
        # scores = M.dot(1 - covered)
        uncovered = 1 - covered
        scores = M.dot(uncovered)
        
        best = np.argmax(scores)
        if scores[best] == 0:
            break # Should not happen if coverage is possible
            
        code.append(best)
        covered[balls[best]] = 1
        
    # Pruning Phase
    # cover_counts[w] = number of codewords in `code` that cover word w
    cover_counts = np.zeros(N, dtype=np.int16)
    for c in code:
        cover_counts[balls[c]] += 1
        
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        new_code = []
        for c in code:
            # Check if removing c leaves any word with count 0
            # If min count in ball(c) is > 1, we can remove c
            if cover_counts[balls[c]].min() > 1:
                cover_counts[balls[c]] -= 1
                changed = True
            else:
                new_code.append(c)
        code = new_code
        
    # Convert indices back to words
    result = []
    for idx in code:
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        result.append(digits)
        
    return result
