"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 135.0

def _autocorr_fft(f):
    n = len(f)
    n_fft = 2 * n
    F = np.fft.rfft(f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)
    return acf[:n]

def _soft_max_sidelobe(acf, beta=100.0):
    if len(acf) <= 1:
        return 0.0
    sidelobes = np.abs(acf[1:])
    max_val = np.max(sidelobes)
    if max_val == 0:
        return 0.0
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _objective_and_grad(f, beta=100.0):
    acf = _autocorr_fft(f)
    obj = _soft_max_sidelobe(acf, beta)
    
    if len(acf) <= 1:
        return obj, np.zeros_like(f)
    
    sidelobes_abs = np.abs(acf[1:])
    max_val = np.max(sidelobes_abs)
    if max_val == 0:
        return obj, np.zeros_like(f)
    
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    n = len(f)
    grad = np.zeros(n)
    for k in range(1, n):
        if acf[k] == 0:
            continue
        sign_k = np.sign(acf[k])
        w = weights[k-1]
        if w < 1e-10:
            continue
        
        grad[:n-k] += w * sign_k * f[k:]
        grad[k:] += w * sign_k * f[:n-k]
    
    return obj, grad

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    f = np.random.choice([-1.0, 1.0], size=N).astype(float)
    best_f = f.copy()
    best_obj, _ = _objective_and_grad(f)
    
    beta = 50.0
    lr = 0.5
    n_steps = 1000
    
    for step in range(n_steps):
        if time.time() - start_time > DEADLINE:
            break
        
        obj, grad = _objective_and_grad(f, beta)
        if obj < best_obj:
            best_obj = obj
            best_f = f.copy()
        
        f = f - lr * grad
        f = np.sign(f)
        
        if step % 100 == 0 and step > 0:
            lr *= 0.9
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            for i in range(N):
                if time.time() - start_time > DEADLINE - 5:
                    break
                f[i] = -f[i]
                obj, _ = _objective_and_grad(f, beta)
                if obj < best_obj:
                    best_obj = obj
                    best_f = f.copy()
                    improved = True
                else:
                    f[i] = -f[i]
    
    return best_f
# EVOLVE-BLOCK-END
