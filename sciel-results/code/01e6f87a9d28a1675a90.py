import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 115.0  # Safety margin

def _compute_overlaps(h):
    """
    Compute linear overlap values C_s for s=1..N-1 using FFT.
    C_s = sum_{i=0}^{N-s-1} h_i * (1 - h_{i+s})
    Note: The problem defines overlap as sum h[i]*(1-h[i+s]).
    """
    n = len(h)
    # We need linear autocorrelation-like terms but with (1-h)
    # C_s = sum h_i - sum h_i h_{i+s}
    # sum h_i is not constant for each s, it's sum of first N-s elements.
    # Let's just compute it directly or via FFT.
    # FFT of h gives H.
    # Circular correlation r_k = sum h_i h_{i+k}.
    # Linear overlap involves non-circular shifts.
    # For N=100, O(N^2) is 10k ops. FFT is 100*log(100) ~ 700 ops.
    # But the term is h_i * (1-h_{i+s}) = h_i - h_i h_{i+s}.
    # Sum_{i=0}^{N-s-1} h_i is prefix sum.
    # Sum_{i=0}^{N-s-1} h_i h_{i+s} is part of linear autocorrelation.
    
    # Let's use the direct method but vectorized if possible, or just the loop.
    # For N=100, a pure Python loop with NumPy slices is fine.
    # But let's try to be clever.
    # h_pad = [h, 0, ..., 0]
    # H = fft(h_pad)
    # corr = ifft(H * conj(H))
    # corr[k] = sum_{i} h[i] h[i+k] (circular)
    # For linear, we take the first N values of corr?
    # corr[k] for k=0..N-1 in the padded array corresponds to linear correlation if padded with zeros.
    
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    # Power spectrum
    power = H * np.conj(H)
    # Inverse FFT gives circular autocorrelation of the padded signal
    corr = np.fft.irfft(power, n=n_fft)
    # corr[k] = sum_{i=0}^{n_fft-1} h_pad[i] h_pad[(i+k) % n_fft]
    # Since h_pad is zero for i>=n, this is sum_{i=0}^{n-k-1} h[i] h[i+k] for k < n.
    # This is exactly the linear autocorrelation term.
    
    # We need C_s = sum_{i=0}^{N-s-1} (h[i] - h[i]h[i+s])
    # Let P_s = sum_{i=0}^{N-s-1} h[i]
    # Let A_s = sum_{i=0}^{N-s-1} h[i]h[i+s] = corr[s]
    # C_s = P_s - A_s
    
    # Compute prefix sums of h
    # P_s is sum of h[0:N-s]
    # We can compute all P_s efficiently.
    # rev_h = h[::-1]
    # cumsum_rev = np.cumsum(rev_h)
    # P_s = total_sum - sum(h[N-s:])
    # Or just use np.cumsum on h.
    # sum(h[0:N-s]) = cumsum_h[N-s-1] if N-s > 0
    
    # Let's just compute the overlaps directly to avoid index confusion, 
    # as N=100 is small. The FFT is overkill if we just do the subtraction.
    # However, the gradient is the expensive part if done naively.
    # Let's stick to the direct calculation for values, it's safe.
    
    overlaps = np.zeros(N - 1)
    for s in range(1, N):
        # h[:N-s] * (1 - h[s:])
        overlaps[s - 1] = np.sum(h[:N - s] * (1.0 - h[s:]))
    return overlaps

def _compute_overlaps_and_grad(h, weights):
    """
    Compute overlaps and the gradient of sum(weights * overlaps).
    weights: array of length N-1.
    """
    n = len(h)
    overlaps = np.zeros(n - 1)
    grad = np.zeros(n)
    
    for s in range(1, n):
        idx = s - 1
        w = weights[idx]
        if w == 0:
            continue
            
        # overlap = sum_{i=0}^{n-s-1} h[i] * (1 - h[i+s])
        # d/dh[i] for i < n-s: (1 - h[i+s])
        # d/dh[i] for i >= s: -h[i-s]
        
        # Vectorized contribution to gradient
        # grad[:n-s] += w * (1 - h[s:])
        # grad[s:] += w * (-h[:n-s])
        
        # Compute overlap value
        overlaps[idx] = np.sum(h[:n - s] * (1.0 - h[s:]))
        
        # Update gradient
        grad[:n - s] += w * (1.0 - h[s:])
        grad[s:] += w * (-h[:n - s])
        
    return overlaps, grad

def _fix_sum(h, target=None):
    """Project h to sum=target, 0<=h<=1."""
    if target is None:
        target = N / 2.0
    h = np.array(h, dtype=float)
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    
    # Distribute diff evenly, then clip and fix residual
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed sum, fix residual on free variables
    residual = target - np.sum(h)
    if abs(residual) > 1e-12:
        if residual > 0:
            # Add to smallest
            idx = np.argsort(h)
            for i in idx:
                if residual <= 0:
                    break
                room = 1.0 - h[i]
                add = min(room, residual)
                h[i] += add
                residual -= add
        else:
            # Subtract from largest
            idx = np.argsort(h)[::-1]
            for i in idx:
                if residual >= 0:
                    break
                room = h[i]
                sub = min(room, -residual)
                h[i] -= sub
                residual += sub
    return h

def _smooth_max(values, temperature):
    """Smooth max via log-sum-exp."""
    max_val = np.max(values)
    if temperature < 1e-6:
        return max_val
    return max_val + temperature * np.log(np.sum(np.exp((values - max_val) / temperature)))

def _objective_and_grad(h, temperature):
    """
    Objective: Smooth max of overlaps + penalty for sum.
    """
    overlaps = np.zeros(N - 1)
    # We need weights for gradient.
    # Weights are softmax of overlaps / temperature
    max_ov = np.max(overlaps) # Wait, we need overlaps first to get weights.
    
    # Calculate overlaps
    for s in range(1, N):
        overlaps[s - 1] = np.sum(h[:N - s] * (1.0 - h[s:]))
        
    # Calculate weights
    if temperature < 1e-6:
        max_idx = np.argmax(overlaps)
        weights = np.zeros(N - 1)
        weights[max_idx] = 1.0
    else:
        exp_vals = np.exp((overlaps - np.max(overlaps)) / temperature)
        weights = exp_vals / np.sum(exp_vals)
        
    obj = _smooth_max(overlaps, temperature)
    
    # Calculate gradient
    grad = np.zeros(N)
    for s in range(1, N):
        idx = s - 1
        w = weights[idx]
        if w == 0:
            continue
        grad[:N - s] += w * (1.0 - h[s:])
        grad[s:] += w * (-h[:N - s])
        
    # Penalty for sum constraint
    sum_val = np.sum(h)
    penalty = 100.0 * (sum_val - N / 2.0)
    grad += 100.0 / N * np.ones(N)
    
    return obj + penalty, grad

def construct():
    deadline = time.time() + DEADLINE
    np.random.seed(42)
    
    # Initial guesses
    h_thue = np.zeros(N)
    for i in range(N):
        h_thue[i] = 1.0 if bin(i).count('1') % 2 == 1 else 0.0
    h_thue = _fix_sum(h_thue)
    
    h_rand = np.random.uniform(0, 1, N)
    h_rand = _fix_sum(h_rand)
    
    h_half = np.full(N, 0.5)
    
    h_square = np.zeros(N)
    h_square[:N//2] = 1.0
    h_square = _fix_sum(h_square)
    
    h_square2 = np.zeros(N)
    h_square[N//2:] = 1.0
    h_square2 = _fix_sum(h_square2)
    
    h_alternate = np.zeros(N)
    h_alternate[::2] = 1.0
    h_alternate = _fix_sum(h_alternate)
    
    starts = [h_thue, h_rand, h_half, h_square, h_square2, h_alternate]
    
    best_h = None
    best_score = float('inf')
    
    # Optimization parameters
    temperatures = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    for start in starts:
        if time.time() > deadline - 5:
            break
            
        h = start.copy()
        
        # Run optimization with decreasing temperature
        for temp in temperatures:
            if time.time() > deadline - 5:
                break
                
            def fun(x):
                return _objective_and_grad(x, temp)
            
            try:
                res = minimize(fun, h, method='L-BFGS-B', jac=True,
                               options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
                
                if res.x is not None:
                    h_new = _fix_sum(res.x)
                    # Calculate true score (max overlap, not smooth max)
                    score = np.max(_compute_overlaps(h_new))
                    
                    if score < best_score:
                        best_score = score
                        best_h = h_new.copy()
                        
                    h = h_new.copy()
            except Exception:
                pass
    
    # Final polish with very low temperature
    if best_h is not None and time.time() < deadline - 2:
        for temp in [0.005, 0.001, 0.0]:
            if time.time() > deadline - 2:
                break
            def fun(x):
                return _objective_and_grad(x, temp)
            try:
                res = minimize(fun, best_h, method='L-BFGS-B', jac=True,
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.x is not None:
                    h_new = _fix_sum(res.x)
                    score = np.max(_compute_overlaps(h_new))
                    if score < best_score:
                        best_score = score
                        best_h = h_new.copy()
            except Exception:
                pass

    if best_h is None:
        return h_thue
        
    return _fix_sum(best_h)

# Run once to ensure no errors? No, just define.
