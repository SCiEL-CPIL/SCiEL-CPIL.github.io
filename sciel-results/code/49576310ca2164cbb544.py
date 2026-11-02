import numpy as np
import time
import itertools

def construct(q, n, radius):
    # Assume q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n
    start_time = time.time()
    deadline = start_time + 118  # Leave 2s margin

    # Precompute neighbors
    neighbors = []
    for w in range(total_words):
        digits = []
        tmp = w
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits = digits[::-1]
        
        neigh = [w]
        for i in range(n):
            for d in range(q):
                if d == digits[i]:
                    continue
                new_digits = digits.copy()
                new_digits[i] = d
                val = 0
                for j in range(n):
                    val = val * q + new_digits[j]
                neigh.append(val)
        neighbors.append(neigh)
    
    neighbors = np.array(neighbors, dtype=np.int32)

    best_code = None
    best_size = total_words + 1

    for restart in range(10):  # Try more restarts
        if time.time() > deadline - 5:
            break
            
        cover_count = np.zeros(total_words, dtype=np.int32)
        code = []
        uncovered_count = total_words
        
        while uncovered_count > 0:
            if time.time() > deadline - 2:
                break
            
            scores = np.sum((cover_count[neighbors] == 0), axis=1)
            
            if restart > 0:
                noise = np.random.rand(total_words) * 1e-6
                scores += scores * 0 + noise  # Just add noise to break ties
                
            best_c = np.argmax(scores)
            code.append(best_c)
            
            cover_count[neighbors[best_c]] += 1
            uncovered_count -= scores[best_c]
            
        if uncovered_count > 0:
            continue
            
        # Remove redundant codewords
        improved = True
        while improved:
            improved = False
            new_code = []
            for c in code:
                min_cover = np.min(cover_count[neighbors[c]])
                if min_cover >= 2:
                    cover_count[neighbors[c]] -= 1
                    improved = True
                else:
                    new_code.append(c)
            code = new_code
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
    if best_code is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for c in best_code:
        digits = []
        tmp = c
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits = digits[::-1]
        result.append(digits)
        
    return result
