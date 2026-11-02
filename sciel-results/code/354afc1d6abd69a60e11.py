import numpy as np
import time
import sys
import random

def construct(q, n, radius):
    # For this specific task, q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (not expected)
        # Simple greedy
        pass
        
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precomputation
    total_words = q ** n
    
    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[n-1-i] = idx % q
            idx //= q
        return w

    # Precompute balls for all indices
    # Ball of idx: idx itself, and idx with one digit changed to other q-1 values
    # Size: 1 + n*(q-1) = 1 + 8*2 = 17
    ball_size = 1 + n * (q - 1)
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    
    # Generate balls efficiently
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    for i in range(total_words):
        balls[i, 0] = i
        col = 1
        temp = i
        digits = [0] * n
        for pos in range(n):
            digits[n-1-pos] = temp % q
            temp //= q
            
        for pos in range(n):
            original = digits[pos]
            power = powers[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                new_idx = i + (new_val - original) * power
                balls[i, col] = new_idx
                col += 1

    def solve():
        # 1. Greedy Set Cover with Randomization
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        
        while not np.all(covered):
            if time.time() > deadline - 5:
                break
                
            # Calculate new coverage for all candidates
            new_covers = np.sum(~covered[balls], axis=1)
            
            max_cover = np.max(new_covers)
            if max_cover == 0:
                break
                
            # Add small random noise to break ties and allow diversity
            noisy_covers = new_covers + np.random.random(total_words) * 0.5
            best_idx = np.argmax(noisy_covers)
            
            code_indices.append(best_idx)
            covered[balls[best_idx]] = True
            
        # 2. Local Search (Removal)
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_indices:
            counts[balls[idx]] += 1
            
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5:
                break
                
            # Try to remove codewords
            for i in range(len(code_indices) - 1, -1, -1):
                idx = code_indices[i]
                # Tentatively remove
                counts[balls[idx]] -= 1
                if np.any(counts == 0):
                    counts[balls[idx]] += 1
                else:
                    code_indices.pop(i)
                    improved = True
                    
        return code_indices

    best_code = None
    best_len = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline - 2:
        code = solve()
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            # If we hit 486, we can stop? Or keep trying for 485?
            # 486 is known best. Lower bound 403.
            # If best_len == 486, we might break to save time, but maybe 485 exists?
            # Unlikely.
            if best_len <= 486:
                # Check if we have time for more
                if time.time() > deadline - 10:
                    break

    if best_code is None:
        # Fallback: run one final greedy without time check if nothing found?
        # Should not happen.
        code = solve()
        best_code = code

    # Convert indices back to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result

construct(3, 8, 1)
