import time
import itertools
import random
import numpy as np

_DEADLINE = 140.0  # seconds

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    total = q ** n
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    # Precompute all words
    all_words = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        all_words.append(tuple(digits))
        
    # Precompute cover matrix using NumPy
    # cover_matrix[i, j] is 1 if word j is within radius of word i
    cover_matrix = np.zeros((total, total), dtype=np.uint8)
    
    # Diagonal is 1
    np.fill_diagonal(cover_matrix, 1)
    
    if radius >= 1:
        for i in range(total):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    power = powers[pos]
                    j = i + (val - orig_val) * power
                    cover_matrix[i, j] = 1
    
    best_code = None
    best_size = total # Upper bound
    
    # Number of restarts
    # Estimate:
    # Greedy: ~150 steps * (2187x2187 dot product ~ 1ms) = 150ms
    # Reduction: ~150 checks * (2187 ops) = negligible
    # Total per restart: ~160ms
    # 140s / 0.16s = ~875 restarts.
    # Let's aim for as many as possible until deadline.
    
    for restart in range(2000):
        if time.time() > deadline - 2:
            break
            
        # Greedy Set Cover
        uncovered = np.ones(total, dtype=bool)
        uncovered_uint8 = uncovered.astype(np.uint8)
        selected = []
        
        while uncovered.any():
            # Calculate gain for each codeword
            # gains[i] = number of uncovered words covered by i
            gains = cover_matrix @ uncovered_uint8
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Should not happen if space is connected, but safety
                
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random choice among best to diversify
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            best_idx_int = int(best_idx)
            
            selected.append(best_idx_int)
            
            # Update uncovered
            # uncovered[cover_matrix[best_idx_int] == 1] = False
            # Faster:
            mask = cover_matrix[best_idx_int]
            uncovered_uint8[mask == 1] = 0
            uncovered = uncovered_uint8.astype(bool) # Keep bool for .any()
            
        if not uncovered.any():
            # Local Search: Remove redundant codewords
            # cover_count[w] = number of codewords covering word w
            cover_count = np.zeros(total, dtype=np.int32)
            
            for idx in selected:
                cover_count += cover_matrix[idx]
                
            improved = True
            while improved:
                improved = False
                # Shuffle to randomize removal order
                random.shuffle(selected)
                n_cw = len(selected)
                
                for i in range(n_cw):
                    if time.time() > deadline - 2:
                        break
                    c = selected[i]
                    mask = cover_matrix[c]
                    
                    # Check if c covers any word uniquely
                    # unique_words = mask & (cover_count == 1)
                    # If no unique words, c is redundant.
                    unique_mask = (cover_count == 1) & (mask == 1)
                    
                    if not np.any(unique_mask):
                        # Remove c
                        selected.pop(i)
                        cover_count -= mask
                        improved = True
                        break # Restart the while loop with new list size
                    
            if len(selected) < best_size:
                best_size = len(selected)
                best_code = selected[:]
                
    # Convert best_code to list of lists
    if best_code is None:
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for idx in best_code:
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        result.append(digits)
        
    return result
