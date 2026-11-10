import numpy as np
import time
import random

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # But keep it general enough if possible, though the task is specific.
    # The evaluator passes q, n, radius.
    
    N = q ** n
    if N > 100000:
        # Fallback for larger spaces, though this task is small
        # For now, assume small N as per task description
        pass
        
    # Precompute neighbors for all words
    # A word is covered if it is in the code or within 'radius' of a code word.
    # Here radius=1.
    
    # Function to convert integer index to base-q digits
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits
        
    # Function to convert base-q digits to integer index
    def digits_to_idx(digits):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + digits[i]
        return idx

    # Precompute neighbors for radius=1
    # For general radius, we'd need to generate all words within that distance.
    # Since the task is specifically K_3(7,1), we optimize for radius=1.
    # If radius != 1, we might need a different approach, but the contract says "this task's".
    
    neighbors = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
    
    for i in range(N):
        d = idx_to_digits(i)
        nb_list = [i]
        for pos in range(n):
            original_val = d[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                d[pos] = new_val
                nb_idx = digits_to_idx(d)
                nb_list.append(nb_idx)
            d[pos] = original_val
        neighbors[i] = nb_list

    best_code = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        covered = np.zeros(N, dtype=bool)
        code = []
        
        # Optional: Start with a random word or a structured one?
        # Greedy usually starts empty.
        
        while not np.all(covered):
            # Calculate gains: number of uncovered neighbors for each candidate
            # neighbors shape: (N, ball_size)
            # covered[neighbors] shape: (N, ball_size) boolean
            # ~covered[neighbors] shape: (N, ball_size) boolean
            # sum along axis 1
            gains = (~covered[neighbors]).sum(axis=1)
            
            max_gain = np.max(gains)
            
            # If max_gain is 0, we are stuck (shouldn't happen if space is connected)
            if max_gain == 0:
                break
                
            # Find all candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly to diversify search
            best_c = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(int(best_c))
            
            # Update covered
            covered[neighbors[best_c]] = True
            
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            # If we found the theoretical lower bound or a very good result, we might stop early?
            # Lower bound is 156. Best known 186.
            if best_len <= 186:
                # Keep searching for a bit to see if we can do better, but don't spend too much time
                # If we hit 156, we can stop.
                if best_len == 156:
                    break

    # Convert best_code indices back to lists of digits
    if best_code is None:
        # Fallback: return a trivial covering (all words)
        return [idx_to_digits(i) for i in range(N)]
        
    result = []
    for c in best_code:
        result.append(idx_to_digits(c))
        
    return result
