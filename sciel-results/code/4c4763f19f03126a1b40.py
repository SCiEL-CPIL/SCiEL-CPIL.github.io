import numpy as np
import time

def _compute_area(x, y_vals, a_vals, n):
    L = x[None, :] + a_vals[:, None] * y_vals[None, :]
    L_sorted = np.sort(L, axis=0)
    w = (1.0 - y_vals) / n
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    w_row = w[None, :]
    union_lengths = w + np.sum(np.minimum(gaps, w_row), axis=0)
    return np.mean(union_lengths)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    M = 20000
    y_vals = (np.arange(M) + 0.5) / M
    a_vals = (np.arange(n) + 1) / n
    
    # Keich init
    j = np.arange(1, k + 1)
    x = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    best_x = x.copy()
    best_area = _compute_area(best_x, y_vals, a_vals, n)
    
    start_time = time.time()
    deadline = start_time + 600.0
    
    rng = np.random.default_rng(42)
    T = 0.005
    cooling = 0.999
    min_step = 1e-7
    max_step = 0.05
    
    while time.time() < deadline - 5.0:
        order = rng.permutation(n)
        for i in order:
            if time.time() > deadline - 5.0:
                break
            step = max(min_step, max_step * T / 0.005)
            delta = rng.uniform(-step, step)
            new_x = best_x.copy()
            new_x[i] += delta
            new_area = _compute_area(new_x, y_vals, a_vals, n)
            
            if new_area < best_area:
                best_x = new_x
                best_area = new_area
            else:
                d = new_area - best_area
                if d > 0 and rng.random() < np.exp(-d / (T + 1e-15)):
                    best_x = new_x
                    best_area = new_area
        T *= cooling
    
    return best_x.astype(np.float64)
