import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _linear_overlaps(h):
    """Compute linear overlaps O_k = sum_i h_i (1 - h_{i+k}) for k=0..N-1."""
    g = 1.0 - h
    # Zero-pad to 2N for linear convolution/correlation
    # Correlation: sum h_i g_{i+k}
    # This is cross-correlation.
    # Using FFT: ifft(fft(h_pad) * conj(fft(g_pad)))
    # Note: standard correlation is ifft(fft(h) * conj(fft(g)))
    # But we need to be careful with indexing.
    # Let's just do it directly for N=100 to be safe and clear, or use fft.
    # Direct O(N^2) is 10000 ops, very fast.
    O = np.zeros(N)
    for k in range(N):
        # Valid i: 0 <= i < N and 0 <= i+k < N
        # i >= 0
        # i <= N-1-k
        # i < N
        i_min = 0
        i_max = N - 1 - k
        if i_max < i_min:
            continue
        O[k] = np.sum(h[i_min:i_max+1] * g[i_min+k:i_max+1+k])
    return O

def _max_linear_overlap(h):
    O = _linear_overlaps(h)
    return np.max(O)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Initial guess: uniform
    h0 = np.full(N, 0.5)
    
    # Cosine start (inspired)
    h_cos = 0.5 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N)
    h_cos = np.clip(h_cos, 0.0, 1.0)
    h_cos = h_cos * (N / 2.0) / np.sum(h_cos)
    h_cos = np.clip(h_cos, 0.0, 1.0)
    
    best_h = h0.copy()
    best_val = _max_linear_overlap(h0)
    
    if _max_linear_overlap(h_cos) < best_val:
        best_val = _max_linear_overlap(h_cos)
        best_h = h_cos.copy()

    def objective_and_grad(h):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            raise RuntimeError("timeout")
        
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        if s > 0:
            h = h * (N / 2.0) / s
            h = np.clip(h, 0.0, 1.0)
        
        O = _linear_overlaps(h)
        
        # Soft-max of O
        tau = 0.05
        m = np.max(O)
        exp_vals = np.exp((O - m) / tau)
        sum_exp = np.sum(exp_vals)
        soft_max_val = m + tau * np.log(sum_exp)
        
        # Gradient
        w = exp_vals / sum_exp
        grad = np.zeros(N)
        
        # G_j = sum_k w_k [ (1 - h_{j+k}) - h_{j-k} ] with bounds
        # We can compute this by iterating k
        for k in range(N):
            wk = w[k]
            if wk < 1e-12: continue
            
            # Term 1: (1 - h_{j+k}) for j such that 0<=j<N, 0<=j+k<N
            # j ranges 0 to N-1-k
            # grad[j] += wk * (1 - h[j+k])
            j_max = N - 1 - k
            if j_max >= 0:
                grad[0:j_max+1] += wk * (1.0 - h[k:j_max+1+k])
            
            # Term 2: -h_{j-k} for j such that 0<=j<N, 0<=j-k<N
            # j-k >= 0 => j >= k
            # j-k < N => j < N+k (always true for j<N)
            # j ranges k to N-1
            if k < N:
                grad[k:N] -= wk * h[0:N-k]
        
        return soft_max_val, grad

    # Multi-start
    np.random.seed(42)
    starts = [h0, h_cos]
    for _ in range(5):
        h_r = np.random.uniform(0, 1, N)
        h_r = h_r * (N / 2.0) / np.sum(h_r)
        h_r = np.clip(h_r, 0.0, 1.0)
        starts.append(h_r)
        
    for h_init in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            result = minimize(
                objective_and_grad,
                h_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if result.success or result.fun < best_val:
                h_res = np.clip(result.x, 0.0, 1.0)
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res * (N / 2.0) / s
                    h_res = np.clip(h_res, 0.0, 1.0)
                val = _max_linear_overlap(h_res)
                if val < best_val:
                    best_val = val
                    best_h = h_res.copy()
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                objective_and_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result.fun < best_val:
                h_res = np.clip(result.x, 0.0, 1.0)
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res * (N / 2.0) / s
                    h_res = np.clip(h_res, 0.0, 1.0)
                val = _max_linear_overlap(h_res)
                if val < best_val:
                    best_val = val
                    best_h = h_res.copy()
        except Exception:
            pass

    # Final projection
    best_h = np.clip(best_h, 0.0, 1.0)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h * (N / 2.0) / s
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
