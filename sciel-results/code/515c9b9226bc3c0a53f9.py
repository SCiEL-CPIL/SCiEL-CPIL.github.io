import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Theoretical optimum: h=a on [0,1], h=b=1-a on [1,2]
    # minimize max(a, (1-a)^2) subject to a+b=1
    # a = (3 - sqrt(5))/2
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.381966
    B = 1.0 - A                        # ~0.618034

    def normalize(h):
        h = np.clip(h, 0, 1)
        s = h.sum()
        if s > 0:
            h = h / s * (N / 2.0)
        h = np.clip(h, 0, 1)
        # Fix any tiny numerical issues
        diff = N / 2.0 - h.sum()
        if abs(diff) > 1e-12:
            # Distribute correction evenly where possible
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
                h = np.clip(h, 0, 1)
        return h

    def compute_corr(h):
        """Compute circular autocorrelation using FFT."""
        fft_h = np.fft.fft(h, n=2 * N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        return corr

    def max_overlap(h):
        corr = compute_corr(h)
        min_corr = np.min(corr)
        return N / 2.0 - min_corr

    # Initial candidate: block construction
    half = N // 2
    h_block = np.zeros(N)
    h_block[:half] = A
    h_block[half:] = B
    h_block = normalize(h_block)
    score_block = max_overlap(h_block)

    # Another candidate: uniform
    h_uniform = np.full(N, 0.5)
    score_uniform = max_overlap(h_uniform)

    best_h = h_block.copy() if score_block < score_uniform else h_uniform.copy()
    best_score = min(score_block, score_uniform)

    # Phase 1: SLSQP optimization with soft-max objective
    if time.time() < deadline - 40:
        # Start from the block construction
        h0 = best_h.copy()
        
        alpha = 50.0
        
        def calc_c(h):
            """Calculate C_k for k=0..N-1 using FFT"""
            corr = compute_corr(h)
            cs = (2.0 / N) * (N / 2.0 - corr)
            return cs

        def objective(h):
            cs = calc_c(h)
            m = np.max(cs)
            # Avoid overflow in exp
            exp_terms = np.exp(alpha * (cs - m))
            log_sum = np.log(np.sum(exp_terms))
            soft_max = m + log_sum / alpha
            return soft_max

        def gradient(h):
            cs = calc_c(h)
            m = np.max(cs)
            exp_terms = np.exp(alpha * (cs - m))
            w = exp_terms / np.sum(exp_terms)
            
            grad = np.zeros(N)
            # dC_k/dh_j = -2/N * (h_{j-k} + h_{j+k}) (circular)
            # We sum over k: w_k * dC_k/dh_j
            
            # Efficiently compute using FFT or direct loops
            # N is small (100), so loops are fine
            
            for k in range(N):
                if w[k] < 1e-12:
                    continue
                factor = -2.0 / N * w[k]
                # Circular indices
                idx1 = (np.arange(N) - k) % N
                idx2 = (np.arange(N) + k) % N
                grad += factor * (h[idx1] + h[idx2])
                    
            return grad

        def constraint_sum(h):
            return np.sum(h) - N / 2.0

        constraints = [
            {'type': 'eq', 'fun': constraint_sum}
        ]
        
        bounds = [(0, 1)] * N
        
        try:
            res = minimize(objective, h0, jac=gradient, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.success and res.fun < best_score:
                h_opt = res.x
                h_opt = normalize(h_opt)
                score_opt = max_overlap(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Phase 2: Local gradient refinement
    if time.time() < deadline - 10:
        h = best_h.copy()
        lr = 0.003
        for step in range(5000):
            if time.time() > deadline - 5:
                break

            corr = compute_corr(h)
            k_max = np.argmin(corr)  # shift that gives max overlap

            # Gradient of overlap w.r.t. h[j]:
            # d/dh[j] (N/2 - corr[k]) = -(h[(j+k)%N] + h[(j-k)%N])
            idx = np.arange(N)
            grad = -(h[(idx + k_max) % N] + h[(idx - k_max) % N])

            h = h - lr * grad
            h = normalize(h)

            # Adaptive learning rate: reduce if oscillating
            if step % 100 == 0 and step > 0:
                lr = max(0.0005, lr * 0.95)

        score = max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Phase 3: Final fine-tuning with smaller steps
    if time.time() < deadline - 5:
        h = best_h.copy()
        lr = 0.001
        for step in range(2000):
            if time.time() > deadline - 2:
                break
            corr = compute_corr(h)
            k_max = np.argmin(corr)
            idx = np.arange(N)
            grad = -(h[(idx + k_max) % N] + h[(idx - k_max) % N])
            h = h - lr * grad
            h = normalize(h)

        score = max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Ensure exact constraints
    best_h = normalize(best_h)

    # Final safety check
    if best_h.min() < -1e-9 or best_h.max() > 1 + 1e-9:
        return np.full(N, 0.5)
    if abs(best_h.sum() - N / 2.0) > 1e-6:
        return np.full(N, 0.5)

    return best_h
