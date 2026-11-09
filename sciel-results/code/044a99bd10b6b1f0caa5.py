import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin

    def evaluate(seq):
        # seq is a numpy array of -1/1
        max_disc = 0
        # d ranges from 1 to n//2
        # If d > n//2, the APs have length 1 or 2?
        # "arithmetic progression contained in [1,256]"
        # Length 1 is always sum +/- 1.
        # Length 2: i, i+d. Sum is s_i + s_{i+d}.
        # Max abs sum for length 1 is 1.
        # We need to check all lengths.
        # The prefix sum range method covers all lengths >= 1.
        # For d > n//2, the subsequence length is at most 2 (indices r, r+d).
        # If length 1: range is 1.
        # If length 2: values 0, s_r, s_r+s_{r+d}. Range is max(0, s_r, s_r+s_{r+d}) - min(0, s_r, s_r+s_{r+d}).
        # Since s in {-1, 1}, possible values for s_r+s_{r+d} are -2, 0, 2.
        # Max range is 2.
        # So for d > n//2, max_disc is at least 2 (if there is a pair with same sign) or 1.
        # Actually, if all adjacent (step d) pairs have opposite signs, sum is 0.
        # But we can just loop d up to n.
        # However, for d > n/2, the number of elements in subsequence is <= 2.
        # We can handle d up to n.
        for d in range(1, n + 1):
            for r in range(d):
                if r >= n:
                    break
                sub = seq[r::d]
                if len(sub) == 0:
                    continue
                psum = np.concatenate(([0], np.cumsum(sub)))
                diff = psum.max() - psum.min()
                if diff > max_disc:
                    max_disc = diff
        return max_disc

    # Initialize with Thue-Morse
    s = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        if i % 2 == 0:
            s[i] = s[i // 2]
        else:
            s[i] = -s[i // 2]
            
    current_s = s
    current_disc = evaluate(current_s)
    
    best_s = current_s.copy()
    best_disc = current_disc
    
    rng = random.Random(42)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        i = rng.randint(0, n - 1)
        new_s = current_s.copy()
        new_s[i] *= -1
        
        new_disc = evaluate(new_s)
        
        delta = new_disc - current_disc
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_s = new_s
            current_disc = new_disc
            
            if current_disc < best_disc:
                best_s = current_s.copy()
                best_disc = current_disc
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart if stuck
            if no_improve_count > max_no_improve:
                # Perturb best solution
                current_s = best_s.copy()
                current_disc = best_disc
                # Flip a few random bits
                for _ in range(5):
                    idx = rng.randint(0, n - 1)
                    current_s[idx] *= -1
                current_disc = evaluate(current_s)
                no_improve_count = 0
                T = 1.0 # Reheat
                
    return best_s.tolist()
