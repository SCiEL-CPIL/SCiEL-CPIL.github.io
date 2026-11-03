"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def init_autoconv(f):
        conv = np.zeros(N)
        for k in range(N):
            if k == 0:
                conv[k] = np.sum(f * f)
            else:
                conv[k] = np.dot(f[:N-k], f[k:])
        return conv
    
    def score_from_conv(conv, S):
        if abs(S) < 1e-10:
            return 1e10
        max_abs = np.max(np.abs(conv))
        return max_abs / (S * S)
    
    np.random.seed(42)
    best_f = np.ones(N, dtype=np.float64)
    best_score = 1e10
    
    def local_search(f, S, conv, cur_score, max_iters, temp_start):
        temp = temp_start
        iters = 0
        best_local_score = cur_score
        best_local_f = f.copy()
        
        while iters < max_iters and time.time() < deadline - 3.0:
            i = np.random.randint(N)
            old_val = f[i]
            delta = -2.0 * old_val
            new_val = -old_val
            new_S = S + delta
            
            delta_conv = np.zeros(N)
            if N - i > 0:
                delta_conv[:N-i] += delta * f[i:]
            if i >= 0:
                delta_conv[:i+1] += delta * f[i::-1]
            
            new_conv = conv + delta_conv
            new_score = score_from_conv(new_conv, new_S)
            
            if new_score < best_local_score:
                best_local_score = new_score
                best_local_f = f.copy()
            
            # Metropolis acceptance
            if new_score < cur_score:
                accept = True
            else:
                if cur_score > 1e9 or new_score > 1e9:
                    accept = False
                else:
                    delta_e = new_score - cur_score
                    accept = np.random.random() < np.exp(-delta_e / max(temp, 1e-10))
            
            if accept:
                f[i] = new_val
                S = new_S
                conv = new_conv
                cur_score = new_score
            
            iters += 1
            if iters % 2000 == 0:
                temp *= 0.97
        
        return best_local_f, best_local_score
    
    # Initial: all ones
    f = np.ones(N, dtype=np.float64)
    S = float(N)
    conv = init_autoconv(f)
    cur_score = score_from_conv(conv, S)
    best_f, best_score = local_search(f, S, conv, cur_score, 100000, 0.5)
    
    # Random restarts
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break
        
        f = np.random.choice([-1.0, 1.0], size=N)
        S = float(np.sum(f))
        conv = init_autoconv(f)
        cur_score = score_from_conv(conv, S)
        
        f2, s2 = local_search(f, S, conv, cur_score, 50000, 0.3)
        if s2 < best_score:
            best_score = s2
            best_f = f2
    
    return best_f

# EVOLVE-BLOCK-END
