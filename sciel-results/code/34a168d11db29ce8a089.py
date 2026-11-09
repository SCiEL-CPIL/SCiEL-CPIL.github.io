"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    def autoconv(x):
        n_fft = 2 * N
        X = np.fft.fft(x, n=n_fft)
        return np.fft.ifft(X * np.conj(X)).real[:N]
    
    def compute_score(x):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return float('inf')
        c = autoconv(x)
        return np.max(np.abs(c)) / (s * s)
    
    # Start with all ones
    best_x = np.ones(N)
    best_score = compute_score(best_x)
    
    # Phase 1: Simulated annealing on ±1 sequences
    rng = np.random.default_rng(42)
    x = rng.choice([-1.0, 1.0], size=N)
    current_score = compute_score(x)
    
    t = 0.1
    cooling = 0.9999
    min_t = 0.005
    
    while time.time() < deadline - 30:
        idx = rng.integers(0, N)
        x[idx] *= -1
        new_score = compute_score(x)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            delta = new_score - current_score
            if delta > 0 and np.log(rng.random()) > -delta / max(t, 1e-10):
                current_score = new_score
            else:
                x[idx] *= -1  # revert
        
        t = max(t * cooling, min_t)
    
    # Phase 2: Continuous refinement
    if time.time() < deadline - 15:
        x_cont = best_x.astype(float).copy()
        s = np.sum(x_cont)
        if abs(s) > 1e-10:
            x_cont = x_cont / s  # normalize to sum=1
        
        lr = 0.001
        for it in range(100000):
            if time.time() > deadline - 5:
                break
            c = autoconv(x_cont)
            max_idx = int(np.argmax(np.abs(c)))
            sign_c = np.sign(c[max_idx])
            if sign_c == 0:
                sign_c = 1.0
            k = max_idx
            
            grad = np.zeros(N)
            if k == 0:
                grad[:] = 2.0 * sign_c * x_cont
            else:
                grad[:N-k] += sign_c * x_cont[k:]
                grad[k:] += sign_c * x_cont[:N-k]
            
            x_cont = x_cont - lr * grad
            x_cont -= (np.sum(x_cont) - 1.0) / N
            
            score = compute_score(x_cont)
            if score < best_score:
                best_score = score
                best_x = x_cont.copy()
            
            lr *= 0.9998
    
    return best_x
# EVOLVE-BLOCK-END
