import numpy as np
from scipy.optimize import minimize

N = 100

def _overlap_vector(x):
    """
    Compute the overlap vector O_k for k=1..N-1.
    x is a vector of length N.
    O_k = sum_i x_i (1 - x_{i+k}) = N/2 - A_k (if sum x = N/2).
    """
    n = len(x)
    # Circular autocorrelation
    f = np.fft.fft(x)
    ac = np.fft.ifft(f * np.conj(f)).real
    # ac[0] is sum x^2. ac[k] for k>0 is sum x_i x_{i+k}.
    # We assume sum x = n/2.
    # O_k = n/2 - ac[k]
    return n / 2.0 - ac[1:]

def _soft_max_overlap(x, beta=10.0):
    """
    Soft approximation of max overlap.
    """
    overlaps = _overlap_vector(x)
    # Shift to avoid overflow in exp
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _soft_max_overlap_grad(x, beta=10.0):
    """
    Gradient of soft max overlap.
    """
    overlaps = _overlap_vector(x)
    m = np.max(overlaps)
    weights = np.exp(beta * (overlaps - m))
    weights /= np.sum(weights)
    
    # grad overlaps w.r.t x
    # O_k = n/2 - sum_i x_i x_{i+k}
    # d O_k / d x_j = - (x_{j-k} + x_{j+k}) (circular indices)
    # Sum over k: grad F = sum_k weights_k * d O_k / d x
    n = len(x)
    grad_overlaps = np.zeros(n)
    # Efficient computation of gradient
    # d O_k / d x_j = -x_{(j-k)%n} - x_{(j+k)%n}
    # We can compute this via convolution or FFT.
    # Let's use a simple loop for clarity first, N=100 is small.
    # Actually, we can use FFT.
    # grad O_k is the derivative of the autocorrelation.
    # Let's stick to the loop for safety, or vectorize.
    
    # Vectorized gradient calculation:
    # We want G_j = - sum_{k=1}^{N-1} w_k (x_{(j-k)%N} + x_{(j+k)%N})
    # Note that sum_{k=1}^{N-1} w_k x_{(j-k)%N} is a circular convolution of x and w (reversed?).
    # Let w_rev[k] = w_k.
    # Conv of x and w_rev gives sum_i x_i w_rev[(j-i)%N] = sum_i x_i w_{(i-j)%N} = sum_k x_{(j+k)%N} w_k.
    # So term 1: sum_k w_k x_{(j+k)%N} is conv(x, w)[j].
    # Term 2: sum_k w_k x_{(j-k)%N} is conv(x, w_rev)[j] where w_rev is w reversed?
    # Actually, sum_k w_k x_{(j-k)%N} is the same as conv(x, w)[j] if w is symmetric? No.
    # Let's just use the fact that the gradient of the quadratic form x^T C x is (C + C^T)x.
    # Here we have a weighted sum of autocorrelations.
    
    # Simpler: Finite difference is too slow.
    # Let's use the property:
    # grad O_k = - (shift(-k, x) + shift(k, x))
    # G = - sum_k w_k (shift(-k, x) + shift(k, x))
    # G = - (conv(x, w) + conv(x, w_rev)) where w is zero-padded appropriately?
    
    # For N=100, a loop over k (99 iterations) and vector ops is fast enough.
    G = np.zeros(n)
    for k in range(1, n):
        wk = weights[k-1]
        if wk > 1e-6:
            shift_plus = np.roll(x, -k) # x_{i+k}
            shift_minus = np.roll(x, k)  # x_{i-k}
            G -= wk * (shift_plus + shift_minus)
            
    return G

def construct():
    n = N
    # Initial guess: random balanced or Thue-Morse
    x0 = np.random.default_rng(42).random(n)
    # Normalize to sum n/2
    x0 = x0 / x0.sum() * (n / 2.0)
    
    # Optimization
    beta = 5.0
    def objective(x):
        return _soft_max_overlap(x, beta)
        
    def gradient(x):
        return _soft_max_overlap_grad(x, beta)
        
    # Constraints: sum x = n/2
    # Bounds: [0, 1]
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - n/2.0})
    bounds = [(0, 1) for _ in range(n)]
    
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                   bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-10})
    
    x_opt = res.x
    
    # Round to binary
    # Use a threshold or iterative rounding
    x_bin = (x_opt > 0.5).astype(np.float64)
    
    # Fix sum
    target = n // 2
    s = int(x_bin.sum())
    if s > target:
        # Flip largest x_opt values to 0
        idx = np.argsort(x_opt)[::-1]
        for k in range(s - target):
            x_bin[idx[k]] = 0
    elif s < target:
        # Flip smallest x_opt values to 1
        idx = np.argsort(x_opt)
        for k in range(target - s):
            x_bin[idx[k]] = 1
            
    # Polish with SA
    # ... (reuse SA logic from original but shorter)
    
    return x_bin
