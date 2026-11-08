"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
def _autoconv_inf_norm(x):
    """
    Compute the infinity norm of the autoconvolution of the step function.
    
    The step function is defined on [-1/4, 1/4] with N equal intervals.
    Each interval has width h = (1/4 - (-1/4)) / N = (1/2) / N.
    The function value on interval i is x[i].
    
    The autoconvolution (correlation) at lag k (in units of intervals) is:
      (f * f)(k) = sum_i x[i] * x[i+k] * h
    where the sum is over valid i.
    
    Actually, for a step function on a grid, the autoconvolution at discrete lags
    is: C[k] = h * sum_{i} x[i] * x[i+k]
    
    The "upper bound on C3" likely refers to the maximum absolute value of the
    autoconvolution, or perhaps a specific moment. Given the task says "upper bound 
    on C3" and mentions autoconvolution, and that lower is better, I'll interpret 
    the score as the infinity norm of the autoconvolution sequence (scaled appropriately).
    
    Let's compute the full autoconvolution and return its max absolute value.
    """
    h = 0.5 / N
    # Use FFT for efficient convolution
    # Pad to avoid circular convolution effects
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    conv = np.fft.irfft(X * X, n=n_fft)
    # The linear convolution is in the first 2*N-1 samples
    # But we want the autoconvolution which is symmetric
    # The peak at lag 0 is conv[0]
    # We want max |conv[k]| for k in [0, N-1] (non-negative lags)
    # Actually, for autoconvolution of a real signal, it's symmetric, so we just take max abs
    return h * np.max(np.abs(conv[:N]))

def _construct_continuous(x):
    """
    Minimize the infinity norm of autoconvolution using projected gradient descent
    on the continuous domain x in [-1, 1]^N, then threshold to {-1, 1}.
    """
    # Start with a random initialization
    np.random.seed(42)
    x0 = np.random.uniform(-1, 1, N)
    
    # Use scipy's minimize with a smooth approximation to the infinity norm
    # We'll use the log-sum-exp trick for a smooth approximation to max|conv|
    # But first, let's try a simpler approach: minimize the sum of squares of conv values
    # which is a smooth proxy, then polish.
    
    def objective(x):
        h = 0.5 / N
        n_fft = 2 * N
        X = np.fft.rfft(x, n=n_fft)
        conv = np.fft.irfft(X * X, n=n_fft)
        # Use a smooth approximation to the infinity norm
        # log-sum-exp for the positive and negative parts
        c = h * conv[:N]
        M = 10.0  # temperature for log-sum-exp
        # max(c) approximated by (1/M) * log(sum(exp(M*c)))
        # max(-c) approximated by (1/M) * log(sum(exp(-M*c)))
        # max(|c|) = max(max(c), max(-c))
        c_clipped = np.clip(c, -1e10, 1e10)
        pos = np.log(np.sum(np.exp(M * c_clipped))) / M
        neg = np.log(np.sum(np.exp(-M * c_clipped))) / M
        return max(pos, neg)
    
    def gradient(x):
        h = 0.5 / N
        n_fft = 2 * N
        X = np.fft.rfft(x, n=n_fft)
        conv = np.fft.irfft(X * X, n=n_fft)
        c = h * conv[:N]
        M = 10.0
        c_clipped = np.clip(c, -1e10, 1e10)
        
        # Gradient of log-sum-exp
        exp_pos = np.exp(M * c_clipped)
        exp_neg = np.exp(-M * c_clipped)
        
        sum_pos = np.sum(exp_pos)
        sum_neg = np.sum(exp_neg)
        
        # d/dc_i of (1/M)log(sum(exp(M*c))) = exp(M*c_i)/sum(exp(M*c))
        # d/dc_i of (1/M)log(sum(exp(-M*c))) = -exp(-M*c_i)/sum(exp(-M*c))
        
        # We need max(pos, neg), so we take the gradient of whichever is larger
        pos_val = np.log(sum_pos) / M
        neg_val = np.log(sum_neg) / M
        
        if pos_val >= neg_val:
            # Gradient of pos
            dc_dx = h * (exp_pos / sum_pos)
        else:
            # Gradient of neg
            dc_dx = h * (-exp_neg / sum_neg)
        
        # Now we need d(conv[k])/dx[i]
        # conv[k] = sum_j x[j] * x[j+k] (before scaling by h)
        # d(conv[k])/dx[i] = x[i+k] + x[i-k] (with appropriate bounds)
        # This is equivalent to the convolution of x with the shifted x
        
        # Actually, the gradient of the autoconvolution with respect to x is:
        # grad_x[conv] = 2 * (x * x) where * is correlation
        # More precisely, d/dx_i [sum_j x_j x_{j+k}] = x_{i+k} + x_{i-k}
        
        # Let's compute the gradient of the objective with respect to x
        # Using the chain rule: grad_x = sum_k (dc/dc_k) * (d(conv[k])/dx)
        
        # d(conv[k])/dx_i = x[i+k] + x[i-k] for valid indices
        # This is the correlation of x with x, which is conv itself (up to symmetry)
        # Actually, the gradient of sum_k c_k * conv[k] with respect to x is:
        # sum_k c_k * (x shifted by k + x shifted by -k)
        
        # Let's compute this via FFT
        # The gradient of the autoconvolution energy is related to the convolution
        # d/dx_i [sum_j x_j x_{j+k}] for fixed k is x_{i+k} + x_{i-k}
        # So the total gradient is sum_k dc_k/dx * (x shifted by k + x shifted by -k)
        
        # This is equivalent to 2 * (dc * x) where * is correlation
        # Let's compute it directly
        grad_x = np.zeros(N)
        for k in range(N):
            dk = dc_dx[k]
            if dk == 0:
                continue
            # Contribution from x[i+k] and x[i-k]
            for i in range(N):
                j1 = i + k
                j2 = i - k
                if j1 < N:
                    grad_x[i] += dk * x[j1]
                if j2 >= 0:
                    grad_x[i] += dk * x[j2]
        
        # This O(N^2) gradient is too slow. Let's use FFT.
        # The gradient of sum_k c_k conv[k] with respect to x is:
        # sum_k c_k * (x_{i+k} + x_{i-k})
        # = sum_k c_k x_{i+k} + sum_k c_k x_{i-k}
        # = (c * x)[i] + (c * x)[i] where * is cross-correlation
        # Actually, it's the correlation of c with x, doubled.
        
        # Let's recompute using FFT for efficiency
        # grad_x[i] = sum_k dc_dx[k] * (x[i+k] + x[i-k])
        # This is 2 * correlation(dc_dx, x)[i]
        
        n_fft2 = 2 * N
        C = np.fft.rfft(dc_dx, n=n_fft2)
        X2 = np.fft.rfft(x, n=n_fft2)
        # Correlation: conj(C) * X2
        corr = np.fft.irfft(np.conj(C) * X2, n=n_fft2)
        grad_x = 2.0 * corr[:N]
        
        return grad_x
    
    # Use L-BFGS-B with bounds
    bounds = [(-1, 1)] * N
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 200, 'ftol': 1e-12})
    
    # Threshold to {-1, 1}
    x_bin = np.sign(result.x)
    x_bin[x_bin == 0] = 1.0  # Handle zeros
    
    # Local search polish: try flipping individual bits
    best_x = x_bin.copy()
    best_val = _autoconv_inf_norm(best_x)
    
    for _ in range(10):
        improved = False
        for i in range(N):
            old_val = best_x[i]
            best_x[i] = -old_val
            new_val = _autoconv_inf_norm(best_x)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                best_x[i] = old_val
        if not improved:
            break
    
    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    try:
        result = _construct_continuous(None)
        return result
    except Exception as e:
        # Fallback to a simple pattern if optimization fails
        x = np.ones(N)
        # Try a few simple patterns
        patterns = [
            np.ones(N),
            np.ones(N) * np.sign(np.arange(N) % 2 - 0.5),
            np.ones(N) * np.sign(np.sin(np.linspace(0, 4*np.pi, N))),
        ]
        best_x = patterns[0]
        best_val = _autoconv_inf_norm(best_x)
        for p in patterns[1:]:
            v = _autoconv_inf_norm(p)
            if v < best_val:
                best_val = v
                best_x = p
        return best_x
# EVOLVE-BLOCK-END
