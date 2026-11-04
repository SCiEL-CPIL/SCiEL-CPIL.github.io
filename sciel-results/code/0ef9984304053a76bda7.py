import numpy as np
from scipy.optimize import minimize
import time

N = 400

def autoconv(x):
    # Efficient autoconvolution using FFT
    x_padded = np.zeros(2 * N)
    x_padded[:N] = x
    X = np.fft.fft(x_padded)
    C = np.fft.ifft(X * X)
    return C.real

def score_continuous(x, alpha=10.0, eps=1e-6):
    c = autoconv(x)
    S = np.sum(x)
    S_sq = S * S + eps
    
    c_off = c[1:N]
    abs_c = np.abs(c_off)
    max_val = np.max(abs_c)
    if max_val > 100:
        soft_max = max_val + np.log(np.sum(np.exp((abs_c - max_val) * alpha)) / alpha)
    else:
        soft_max = np.log(np.sum(np.exp(abs_c * alpha)) / alpha)
        
    return soft_max / S_sq

def score_discrete(x):
    c = autoconv(x)
    S = np.sum(x)
    if S == 0:
        return 1e9
    c_off = c[1:N]
    max_off = np.max(np.abs(c_off))
    return max_off / (S * S)

def local_search(x, max_iters=1000, time_limit=10.0):
    start = time.time()
    current_score = score_discrete(x)
    x = x.copy()
    for _ in range(max_iters):
        if time.time() - start > time_limit:
            break
        # Try flipping a random bit
        i = np.random.randint(N)
        x[i] *= -1
        new_score = score_discrete(x)
        if new_score < current_score:
            current_score = new_score
        else:
            x[i] *= -1 # revert
    return x, current_score

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Start with all ones
    x_best = np.ones(N)
    best_score = score_discrete(x_best)
    
    # Local search from all ones
    x_ls, score_ls = local_search(x_best, max_iters=2000, time_limit=20.0)
    if score_ls < best_score:
        best_score = score_ls
        x_best = x_ls
    
    # Try continuous optimization from random starts
    for i in range(5):
        if time.time() > deadline - 20:
            break
        x0 = np.random.uniform(-1, 1, N)
        try:
            res = minimize(
                score_continuous, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(-1, 1)] * N,
                options={'maxiter': 100, 'maxfun': 1000}
            )
            x_int = np.sign(res.x)
            x_int[x_int == 0] = 1
            score_int = score_discrete(x_int)
            if score_int < best_score:
                best_score = score_int
                x_best = x_int
        except Exception:
            pass
            
    return x_best
