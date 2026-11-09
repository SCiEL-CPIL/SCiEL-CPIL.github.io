import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    # Generate all words
    total = q ** n
    words = np.array(list(np.ndindex(*([q] * n))), dtype=np.int8)
    
    # Precompute balls for each word
    # ball[i] is a list of indices of words within distance 1 of words[i]
    balls = []
    for i in range(total):
        ball = [i]
        w = words[i]
        for j in range(n):
            for delta in range(1, q):
                new_w = w.copy()
                new_w[j] = (new_w[j] + delta) % q
                # Find the index of new_w
                # This is slow, so we need a mapping
                # Let's create a mapping from tuple to index
                pass
        balls.append(ball)
    
    # This is too slow. Let's use a different approach.
    # Map each word to an integer index
    # Use a dictionary or a direct formula
    # Index = sum(w[i] * q**(n-1-i))
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    # Precompute balls
    balls = []
    for i in range(total):
        w = words[i]
        ball = {i}
        for j in range(n):
            for delta in range(1, q):
                new_w = w.copy()
                new_w[j] = (new_w[j] + delta) % q
                idx = word_to_idx(new_w)
                ball.add(idx)
        balls.append(list(ball))
    
    # Greedy set cover
    covered = np.zeros(total, dtype=bool)
    code = []
    
    while not covered.all():
        # Find the candidate with max uncovered coverage
        best_idx = -1
        best_count = 0
        for i in range(total):
            if covered[i]:
                continue
            count = np.sum(~covered[balls[i]])
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break
        
        code.append(list(words[best_idx]))
        covered[balls[best_idx]] = True
        
        # Time check
        if time.time() - start_time > deadline:
            break
    
    return code
