import numpy as np
import itertools
import random

def construct(q, n, radius):
    total_words = q ** n
    
    if total_words > 50000:
        # Fallback (kept from original, though unlikely to be hit for K_3(7,1))
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                return [list(random.choices(range(q), k=n)) for _ in range(100)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Optimized path for small spaces
    
    # 1. Generate all words and map them to indices
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    
    # 2. Precompute balls. 
    # For radius=1, we can be very efficient.
    # General radius fallback if needed, but task is radius=1.
    balls = np.zeros((total_words, total_words), dtype=bool)
    
    if radius == 1:
        for i, w in enumerate(all_words):
            balls[i, i] = True
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        neighbor = w.copy()
                        neighbor[pos] = new_val
                        j = word_to_idx[tuple(neighbor)]
                        balls[i, j] = True
    else:
        # General case (slower)
        for i, w in enumerate(all_words):
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    balls[i, j] = True

    # 3. Greedy Set Cover with Removal
    best_code = None
    best_size = total_words + 1
    num_restarts = 10
    
    for _ in range(num_restarts):
        uncovered = np.ones(total_words, dtype=bool)
        coverage_counts = np.zeros(total_words, dtype=np.int32)
        code_indices = []
        
        # Greedy Phase
        while uncovered.any():
            # Count how many uncovered words each candidate covers
            # balls @ uncovered gives the count of True values in the intersection
            counts = balls @ uncovered
            
            # Add small random noise to break ties
            noise = np.random.rand(total_words) * 0.1
            scores = counts + noise
            
            best_idx = np.argmax(scores)
            
            # Add to code
            code_indices.append(best_idx)
            
            # Update coverage counts and uncovered mask
            coverage_counts[balls[best_idx]] += 1
            uncovered = (coverage_counts == 0)
            
        # Removal Phase
        # Try to remove redundant codewords
        changed = True
        while changed:
            changed = False
            # Iterate backwards to handle index shifts easily if we were popping,
            # but we are checking validity first.
            for i in range(len(code_indices) - 1, -1, -1):
                idx = code_indices[i]
                # Tentatively remove
                coverage_counts[balls[idx]] -= 1
                
                # Check if any word is now uncovered
                if np.any(coverage_counts == 0):
                    # Restore
                    coverage_counts[balls[idx]] += 1
                else:
                    # Removal is valid
                    code_indices.pop(i)
                    changed = True
                    # Break to restart the scan? 
                    # Or continue? If we continue, indices before i are still valid.
                    # But the set changed. It's safer to restart the while loop 
                    # or just break the for loop and let the while loop restart.
                    break
        
        current_code = [idx_to_word[i] for i in code_indices]
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code
            
    return best_code
