import numpy as np
import time

N = 400
M = 2 * N  # FFT size for full autocorrelation

def _calc_score_fft(h):
    """
    Calculate the score using FFT-based autocorrelation.
    Score = max_{k != 0} |C[k]| / ||h||^2
    """
    h_padded = np.zeros(M)
    h_padded[:N] = h
    
    H = np.fft.fft(h_padded)
    C = np.fft.ifft(H * np.conj(H)).real
    
    # C[0] is lag 0, C[1..N-1] are positive lags, C[M-N+1..M-1] are negative lags
    # We want max |C[k]| for k != 0
    # Since C is real and symmetric, we can just look at C[1:N] and C[M-N+1:M]
    # Actually for real sequences, C[k] = C[M-k], so we just need C[1:N]
    # But let's be safe and check all non-zero lags
    mask = np.ones(M, dtype=bool)
    mask[0] = False
    # Also mask out the negative lag part that wraps around? 
    # No, for linear autocorrelation via zero-padding:
    # C[0] = sum h[i]^2
    # C[k] = sum h[i]h[i+k] for k=1..N-1
    # C[M-k] = sum h[i]h[i-k] for k=1..N-1 (which equals C[k] for real h)
    # So we just need to exclude index 0.
    
    peak = np.max(np.abs(C[mask]))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _calc_score(h):
    """
    Wrapper for backward compatibility if needed, but we use FFT version internally.
    """
    return _calc_score_fft(h)

def _soft_max(x, beta):
    """
    Smooth approximation of max(x).
    """
    return np.log(np.sum(np.exp(beta * x))) / beta

def _objective_and_grad(h, beta=50.0):
    """
    Compute the soft-max objective and its gradient for continuous h.
    We want to minimize max_{k!=0} |C[k]|.
    Let y_k = |C[k]|. We approximate max(y_k) with soft_max(y_k, beta).
    Gradient of soft_max(y, beta) w.r.t. y_k is (1/beta) * exp(beta*y_k) / sum(exp(beta*y_j)).
    Then we need d(y_k)/d(h).
    y_k = |C_k|. d|C_k|/dh = sign(C_k) * dC_k/dh.
    C_k = sum_i h_i h_{i+k}. dC_k/dh_i = h_{i+k} + h_{i-k} (with appropriate bounds).
    
    This is a bit complex. Let's use a simpler approach:
    Minimize sum_k exp(beta * C_k^2) or similar.
    Actually, let's minimize the sum of squares of the sidelobes, or a soft max of the absolute values.
    
    Let's try minimizing: J = (1/N) * sum_{k!=0} (C_k)^2 / N^2
    This is the mean square error of the sidelobes. Minimizing this often correlates with minimizing the peak.
    Gradient of J w.r.t h_i:
    J = (1/N^3) * sum_{k=1}^{N-1} (sum_{j=0}^{N-1-k} h_j h_{j+k})^2
    dJ/dh_i = (2/N^3) * sum_{k=1}^{N-1} C_k * (dC_k/dh_i)
    dC_k/dh_i = h_{i+k} (if i+k < N) + h_{i-k} (if i-k >= 0)
    
    This is computable in O(N^2) which is 160,000 operations. Very fast.
    """
    # C is the autocorrelation vector of length M
    # We only care about k=1..N-1
    h_padded = np.zeros(M)
    h_padded[:N] = h
    H = np.fft.fft(h_padded)
    C = np.fft.ifft(H * np.conj(H)).real
    
    # Extract sidelobes C[1:N]
    sidelobes = C[1:N]
    
    # Objective: sum of squares of sidelobes
    obj = np.sum(sidelobes**2)
    
    # Gradient:
    # d(obj)/d(h_i) = 2 * sum_{k=1}^{N-1} C_k * dC_k/dh_i
    # dC_k/dh_i = h_{i+k} + h_{i-k} (with bounds)
    # This is equivalent to 2 * (h * C)_i where * is the circular convolution? No.
    # Let's compute it directly.
    # grad[i] = 2 * ( sum_{k=1}^{N-1, i+k<N} C_k * h[i+k] + sum_{k=1}^{N-1, i-k>=0} C_k * h[i-k] )
    # This is 2 * ( (h * C)[i] - C[0]*h[i] )? No, C[0] is not in the sum.
    # Let's define an array G where G[k] = C[k] for k=1..N-1, 0 otherwise.
    # Then grad[i] = 2 * sum_{k} G[k] * (h[i+k] + h[i-k])
    # This is 2 * ( (h * G)[i] + (h * G)[i] )? No.
    # (h * G)[i] = sum_j h[j] G[i-j] = sum_j h[j] G[j-i] (since G is symmetric? No, G[k]=C[k], C is even)
    # C[k] = C[-k]. So G[k] = G[-k].
    # sum_k G[k] h[i+k] = (G * h)[i] (cross correlation)
    # sum_k G[k] h[i-k] = (G * h)[i] (cross correlation)
    # So grad[i] = 4 * (G * h)[i]?
    # Let's verify.
    # C_k = sum_j h_j h_{j+k}
    # dC_k/dh_i = h_{i+k} + h_{i-k}
    # sum_k C_k (h_{i+k} + h_{i-k}) = sum_k C_k h_{i+k} + sum_k C_k h_{i-k}
    # Let j = i+k => k = j-i. Sum_j C_{j-i} h_j. This is (C * h)[i] (convolution).
    # Let j = i-k => k = i-j. Sum_j C_{i-j} h_j. This is (C * h)[i] (convolution).
    # So the sum is 2 * (C * h)[i].
    # But we only sum k=1..N-1. C[0] is excluded.
    # So it's 2 * ( (C - C[0]*delta) * h )[i].
    
    # Let's compute convolution C * h
    # C is length M, h is length N (padded to M).
    # Convolution of two length M sequences.
    # We only need the first N elements.
    
    # Create C_full with C[0] set to 0 for the gradient part?
    # No, the derivative of C[0] w.r.t h is 2h, but C[0] is not in the objective.
    # So we use C_masked where C_masked[0] = 0.
    C_masked = C.copy()
    C_masked[0] = 0.0
    
    # Convolution C_masked * h_padded
    # Use FFT for speed
    H_c = np.fft.fft(C_masked)
    H_h = np.fft.fft(h_padded)
    Conv = np.fft.ifft(H_c * H_h).real
    
    # The gradient of the sum of squares is 2 * Conv
    # Wait, d(sum C_k^2)/dh = 2 * sum C_k dC_k/dh = 2 * (2 * Conv) = 4 * Conv?
    # Let's re-check.
    # J = sum_k C_k^2
    # dJ/dh_i = 2 sum_k C_k dC_k/dh_i = 2 sum_k C_k (h_{i+k} + h_{i-k}) = 4 * (C_masked * h)[i]
    
    grad = 4.0 * Conv[:N]
    
    return obj, grad

def _local_search(h, max_iter=1000):
    """
    Discrete local search: flip bits to reduce the score.
    """
    h = h.copy()
    current_score = _calc_score_fft(h)
    
    for _ in range(max_iter):
        improved = False
        # Try flipping each bit
        for i in range(N):
            h[i] *= -1
            new_score = _calc_score_fft(h)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                h[i] *= -1  # Revert
        
        if not improved:
            break
            
    return h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    # Strategy:
    # 1. Initialize with random +/- 1 sequences.
    # 2. Use continuous optimization (L-BFGS-B or simple gradient descent) on a soft objective.
    # 3. Round to +/- 1.
    # 4. Discrete local search.
    
    # We will run multiple iterations with different initializations.
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 60:
            break
        
        # Random initialization
        h_init = np.random.choice([-1, 1], N)
        
        # Continuous optimization
        # We will use a simple gradient descent with line search, or L-BFGS-B if available.
        # Since we don't have scipy.optimize guaranteed in the prompt's "standard library, NumPy and SciPy", 
        # but it says SciPy is available, let's use scipy.optimize.minimize.
        
        from scipy.optimize import minimize
        
        # Define the objective and gradient for scipy
        def obj_func(x):
            obj, _ = _objective_and_grad(x)
            return obj
        
        def grad_func(x):
            _, grad = _objective_and_grad(x)
            return grad
        
        # Start from random continuous values in [-1, 1]
        x0 = np.random.uniform(-1, 1, N)
        
        # Use L-BFGS-B
        # Bounds: [-1, 1]
        bounds = [(-1, 1)] * N
        
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', jac=grad_func, bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        except Exception:
            # Fallback to simple gradient descent if minimize fails
            x = x0
            lr = 0.1
            for i in range(500):
                obj, grad = _objective_and_grad(x)
                x -= lr * grad
                x = np.clip(x, -1, 1)
        
        # Round to +/- 1
        h_cont = res.x if 'res' in locals() else x
        h_bin = np.where(h_cont > 0, 1, -1).astype(int)
        
        # Discrete local search
        h_refined, score = _local_search(h_bin, max_iter=200)
        
        if score < best_score:
            best_score = score
            best_h = h_refined.copy()
            
        # Check time
        if time.time() > deadline - 30:
            break

    # If we still have time, try a few more random initializations with just local search
    if time.time() < deadline - 10:
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            h_init = np.random.choice([-1, 1], N)
            h_refined, score = _local_search(h_init, max_iter=100)
            if score < best_score:
                best_score = score
                best_h = h_refined.copy()

    # Fallback
    if best_h is None:
        best_h = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
        best_score = _calc_score_fft(best_h)

    return best_h
