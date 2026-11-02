import numpy as np
import time

N = 400

def _autoconv_max(s):
    """Compute max |autoconvolution| for a ±1 step function, excluding lag 0."""
    max_abs = 0.0
    for d in range(1, N):
        val = np.dot(s[:N-d], s[d:])
        av = abs(val)
        if av > max_abs:
            max_abs = av
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_s = None
    best_val = float('inf')
    
    for restart in range(8):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(1000 + restart * 7)
        s = np.random.choice([-1.0, 1.0], size=N)
        cur_val = _autoconv_max(s)
        
        if cur_val < best_val:
            best_val = cur_val
            best_s = s.copy()
        
        # Simulated annealing with block flips
        T = 10.0
        T_min = 0.05
        cooling = 0.9999
        max_steps = 40000
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            L = np.random.randint(4, 33)  # block size 4-32
            start = np.random.randint(0, N - L + 1)
            
            old_block = s[start:start+L].copy()
            s[start:start+L] = -s[start:start+L]
            new_val = _autoconv_max(s)
            
            delta = new_val - cur_val
            if delta < 0:
                cur_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_s = s.copy()
            else:
                # Accept with probability exp(-delta/T)
                if T > 1e-10 and np.random.random() < np.exp(-delta / T):
                    cur_val = new_val
                else:
                    s[start:start+L] = old_block
            
            T = max(T_min, T * cooling)
        
        # Polish with single-bit flips
        if best_s is not None:
            s = best_s.copy()
            cur_val = best_val
            polish_steps = 15000
            for _ in range(polish_steps):
                if time.time() > deadline - 2:
                    break
                i = np.random.randint(0, N)
                old = s[i]
                s[i] = -s[i]
                new_val = _autoconv_max(s)
                if new_val < cur_val:
                    cur_val = new_val
                    if new_val < best_val:
                        best_val = new_val
                        best_s = s.copy()
                else:
                    s[i] = old
    
    if best_s is None:
        return np.ones(N)
    return best_s
