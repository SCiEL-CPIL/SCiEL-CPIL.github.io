import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    
    total_words = q ** n
    
    # Fallback for large spaces or non-matching parameters
    if total_words > 5000 or radius != 1:
        # Use the seed's direct sum logic if applicable
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Trivial valid code: all words
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Optimized path for small spaces (e.g., 3^6 = 729)
    
    def idx_to_word(idx):
        w = [0] * n
        for j in range(n):
            w[j] = idx % q
            idx //= q
        return w
    
    def word_to_idx(w):
        idx = 0
        mult = 1
        for j in range(n):
            idx += w[j] * mult
            mult *= q
        return idx

    # Precompute balls
    balls = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        w = idx_to_word(i)
        balls[i, i] = True
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w[pos] = new_val
                    j = word_to_idx(w)
                    balls[i, j] = True
            w[pos] = orig_val

    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy iterations with random noise
    num_iterations = 2000
    
    for _ in range(num_iterations):
        if time.time() > deadline:
            break
            
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        
        while not covered.all():
            uncovered = 1.0 - covered.astype(float)
            scores = balls.dot(uncovered)
            
            # Add noise to break ties and explore
            noise = np.random.uniform(-0.1, 0.1, total_words)
            scores += noise
            
            best_idx = np.argmax(scores)
            
            code_indices.append(best_idx)
            covered[balls[best_idx]] = True
            
        # Reduction Phase
        code_set = list(code_indices)
        coverage_count = np.zeros(total_words, dtype=int)
        for idx in code_set:
            coverage_count[balls[idx]] += 1
            
        while True:
            removed = False
            for k in range(len(code_set)):
                idx = code_set[k]
                # Check if idx is essential
                # Essential if it is the only cover for any word
                if not np.any((coverage_count == 1) & balls[idx]):
                    # Redundant
                    coverage_count[balls[idx]] -= 1
                    code_set.pop(k)
                    removed = True
                    break
            if not removed:
                break
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = [idx_to_word(i) for i in code_set]
            
    if best_code is None:
        return [idx_to_word(i) for i in range(total_words)]
        
    return best_code
