import numpy as np
import time

N = 600
w = 0.5 / N

def _score(h):
    h = np.maximum(h, 0)
    norm = np.sqrt(np.sum(h**2) * w)
    if norm < 1e-12:
        return 1.0
    h = h / norm
    conv = np.convolve(h, h, mode='full') * w
    return np.max(conv)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    rng = np.random.default_rng(12345)
    
    best_h = np.ones(N)
    best_s = _score(best_h)
    
    def grad_descent(h0, max_iters=5000):
        nonlocal best_h, best_s
        h = h0.copy()
        lr = 0.5
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            conv = np.convolve(h, h, mode='full') * w
            max_idx = np.argmax(conv)
            grad = np.zeros(N)
            lo = max(0, max_idx - N + 1)
            hi = min(N - 1, max_idx)
            if lo <= hi:
                seg = h[max_idx - hi : max_idx - lo + 1]
                grad[lo:hi+1] = 2.0 * w * seg[::-1]
            h = h - lr * grad
            h = np.maximum(h, 0)
            norm = np.sqrt(np.sum(h**2) * w)
            if norm < 1e-12:
                break
            h = h / norm
            s = _score(h)
            if s < best_s:
                best_s = s
                best_h = h.copy()
            if it % 100 == 0:
                lr *= 0.99
    
    # Multiple starts
    starts = [
        np.ones(N),
        np.linspace(0.1, 1.0, N),
        np.linspace(1.0, 0.1, N),
        np.exp(-2.0 * (np.arange(N) - N/2)**2 / (N*0.1)),
        np.exp(-2.0 * (np.arange(N) - N*0.3)**2 / (N*0.1)),
        np.ones(N) * (1.0 + 0.2 * rng.standard_normal(N)),
        np.ones(N) * (1.0 + 0.2 * rng.standard_normal(N)),
    ]
    
    for h0 in starts:
        if time.time() > deadline - 10:
            break
        h0 = np.maximum(h0, 0)
        norm = np.sqrt(np.sum(h0**2) * w)
        if norm < 1e-12:
            continue
        h0 = h0 / norm
        grad_descent(h0)
    
    # Simulated annealing fine-tune
    if time.time() < deadline - 10:
        h = best_h.copy()
        T = 0.1
        for it in range(20000):
            if time.time() > deadline - 2:
                break
            i = rng.integers(N)
            delta = 0.1 * rng.standard_normal()
            h_new = h.copy()
            h_new[i] = max(0, h_new[i] + delta)
            norm = np.sqrt(np.sum(h_new**2) * w)
            if norm < 1e-12:
                continue
            h_new = h_new / norm
            s_new = _score(h_new)
            s_old = _score(h)
            if s_new < s_old or rng.random() < np.exp(-(s_new - s_old) / T):
                h = h_new
                if s_new < best_s:
                    best_s = s_new
                    best_h = h.copy()
            T *= 0.999
    
    return list(best_h)
