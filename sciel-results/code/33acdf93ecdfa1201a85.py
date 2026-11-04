import numpy as np
import time

N = 400

def autoconv(x):
    x_padded = np.zeros(2 * N)
    x_padded[:N] = x
    X = np.fft.fft(x_padded)
    C = np.fft.ifft(X * X)
    return C.real

def score_discrete(x):
    c = autoconv(x)
    S = np.sum(x)
    if S == 0:
        return 1e9
    c_off = c[1:N]
    max_off = np.max(np.abs(c_off))
    return max_off / (S * S)

def simulated_annealing(x_init, time_limit=10.0):
    start_time = time.time()
    x = x_init.copy()
    current_score = score_discrete(x)
    best_x = x.copy()
    best_score = current_score
    
    T = 0.1  # Initial temperature
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        if T < 1e-6:
            break
        i = np.random.randint(N)
        x[i] *= -1
        new_score = score_discrete(x)
        
        delta = new_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            x[i] *= -1 # revert
            
        T *= cooling
        
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Start with all ones
    x_best = np.ones(N)
    best_score = score_discrete(x_best)
    
    # Try simulated annealing from all ones
    x_sa, score_sa = simulated_annealing(x_best, time_limit=20.0)
    if score_sa < best_score:
        best_score = score_sa
        x_best = x_sa
        
    # Try different biases
    for bias in [0.9, 0.8, 0.7]:
        if time.time() > deadline - 20:
            break
        x_init = np.ones(N)
        num_flips = int(N * (1 - bias))
        indices = np.random.choice(N, num_flips, replace=False)
        x_init[indices] = -1
        
        x_sa, score_sa = simulated_annealing(x_init, time_limit=15.0)
        if score_sa < best_score:
            best_score = score_sa
            x_best = x_sa
            
    return x_best
