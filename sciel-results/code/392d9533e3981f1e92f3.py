"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft


def _autocorr_full(x):
    """Compute full autocorrelation via FFT (linear, not circular)."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    X = fft(x, size)
    ac = ifft(X * np.conj(X)).real[:2 * n - 1]
    return ac


def _objective(x):
    """Compute max |autocorrelation| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if s == 0:
        return float('inf')
    ac = _autocorr_full(x)
    return np.max(np.abs(ac)) / (s * s)


def _soft_max_abs_ac(x, beta):
    """
    Compute smooth approximation of max |autocorr(x)| using log-sum-exp.
    Returns (soft_max_value, gradient_wrt_x)
    """
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    
    X = fft(x, size)
    # Power spectrum
    P = X * np.conj(X)
    
    # Autocorrelation is ifft(P)
    # We want max |ac_k| for k in [0, 2n-2]
    # ac_k = sum_j x_j x_{j+k}
    
    # Gradient of ac_k wrt x:
    # d(ac_k)/dx_j = x_{j+k} + x_{j-k} (with boundary handling, but FFT handles circular)
    # Actually, for linear autocorr, it's slightly different, but for large N and small shifts, 
    # the circular approx is close. However, to be exact, let's use the property:
    # ac = ifft(|X|^2)
    # Let's compute the gradient of the soft-max.
    
    ac = ifft(P).real[:2 * n - 1]
    abs_ac = np.abs(ac)
    
    # Log-sum-exp for stability
    max_val = np.max(abs_ac)
    if max_val == 0:
        return 0.0, np.zeros_like(x)
    
    # weights = exp(beta * (abs_ac - max_val))
    # To avoid overflow, we subtract max.
    # beta is inverse temperature. High beta -> sharper max.
    weights = np.exp(beta * (abs_ac - max_val))
    sum_w = np.sum(weights)
    
    soft_max_val = max_val + np.log(sum_w) / beta
    
    # Gradient of soft_max_val wrt x:
    # d/dx [ log(sum exp(beta * |ac_k|)) / beta ]
    # = (1/beta) * sum_k [ exp(beta * |ac_k|) / sum_j exp(beta * |ac_j|) * beta * sign(ac_k) * d(ac_k)/dx ]
    # = sum_k [ w_k / sum_w * sign(ac_k) * d(ac_k)/dx ]
    
    # d(ac_k)/dx = ?
    # ac_k = sum_i x_i x_{i+k}
    # d(ac_k)/dx_j = x_{j+k} + x_{j-k} (circular)
    # Let's construct the matrix or use FFT for the gradient.
    # Note: ac is circular autocorr if we use full FFT without zero padding? 
    # Wait, _autocorr_full uses zero padding to 2N-1, so it's linear.
    # The gradient of linear autocorr is not simply circular convolution.
    # However, for the purpose of optimization, we can approximate the gradient 
    # or compute it exactly.
    
    # Exact gradient of linear autocorr ac_k = sum_{i=0}^{N-1-k} x_i x_{i+k}
    # d(ac_k)/dx_j = sum_{i: i=j or i+k=j} x_{partner}
    # This is equivalent to: 
    # grad_ac_k[j] = x_{j+k} (if j+k < N) + x_{j-k} (if j-k >= 0)
    
    # We need: grad_total = sum_k (w_k / sum_w) * sign(ac_k) * grad_ac_k
    # Let's compute this efficiently.
    # grad_ac_k is a vector of length N.
    # We can form a matrix G where G[k, j] = d(ac_k)/dx_j.
    # But N=400, so 2N-1 ~ 800. Matrix is 800x400. This is small enough to construct explicitly.
    
    grad_ac = np.zeros((2 * n - 1, n))
    for k in range(2 * n - 1):
        if k < n:
            # ac_k = sum_{i=0}^{n-1-k} x_i x_{i+k}
            # d/dx_j:
            # if j in [0, n-1-k], term x_j x_{j+k} -> x_{j+k}
            # if j-k in [0, n-1-k], term x_{j-k} x_j -> x_{j-k}
            # So for a given j:
            # contrib 1: if j+k < n, then x_{j+k}
            # contrib 2: if j-k >= 0, then x_{j-k}
            
            # Vectorized:
            # grad_ac[k, :] = 0
            # indices where j+k < n: j in [0, n-1-k]
            # values: x[j+k] for j in [0, n-1-k] -> x[k:n]
            # indices where j-k >= 0: j in [k, n-1]
            # values: x[j-k] for j in [k, n-1] -> x[0:n-k]
            
            if k == 0:
                grad_ac[k, :] = 2 * x
            else:
                # Part 1: x[k:n] added to indices 0:n-k
                grad_ac[k, 0:n-k] += x[k:n]
                # Part 2: x[0:n-k] added to indices k:n
                grad_ac[k, k:n] += x[0:n-k]
        else:
            # k >= n
            # ac_k = sum_{i=0}^{n-1-k} ... wait, if k >= n, the range is empty?
            # Linear autocorr is usually defined for k=0 to N-1.
            # My _autocorr_full returns 2N-1 values.
            # For k >= N, ac_k = sum_{i=0}^{N-1} x_i x_{i+k} is 0 because i+k >= N.
            # Actually, standard linear autocorr is symmetric: ac_{-k} = ac_k.
            # The FFT method with zero padding gives the linear convolution of x and reversed x.
            # Let's check the indices.
            # ifft(|fft(x, 2N)|^2) gives the circular convolution of length 2N.
            # Since x is zero-padded to 2N, the circular conv is the linear conv.
            # The result is ac[0] to ac[2N-2].
            # For k >= N, the overlap is zero. So ac_k = 0.
            # So we only need to handle k < N.
            pass
    
    # Weights vector w
    w = weights / sum_w
    signs = np.sign(ac)
    
    # Combine
    # grad_x = sum_k (w[k] * signs[k]) * grad_ac[k, :]
    # This is a matrix-vector product: grad_x = (w * signs) @ grad_ac
    coeffs = w * signs
    grad_x = coeffs @ grad_ac
    
    return soft_max_val, grad_x


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_score = float('inf')
    best_x = np.ones(N)
    
    # Evaluate baseline
    sc = _objective(best_x)
    if sc < best_score:
        best_score = sc
    
    # Continuous optimization
    # Initialize with random signs or ones
    rng = np.random.RandomState(42)
    
    # Try a few random initializations
    for trial in range(5):
        if time.time() > deadline:
            break
            
        # Start with random continuous values or random signs
        # Let's start with random signs to be in the "signed step" neighborhood
        x = rng.choice([-1.0, 1.0], size=N).astype(float)
        
        # Gradient descent on continuous x
        # Objective: soft_max_abs_ac(x) / (sum x)^2
        # We minimize this.
        
        lr = 0.1
        beta = 10.0  # Start moderate
        current_val = None
        
        # Pre-compute gradient structure? No, x changes.
        
        for it in range(200):
            if time.time() > deadline:
                break
                
            # Compute objective and gradient
            s = np.sum(x)
            if abs(s) < 1e-6:
                # Perturb
                x += 0.1 * rng.randn(N)
                continue
                
            soft_max, grad_soft = _soft_max_abs_ac(x, beta)
            
            # Objective: f(x) = soft_max / s^2
            # grad f = (grad_soft * s^2 - soft_max * 2 * s * 1) / s^4
            #        = grad_soft / s^2 - 2 * soft_max / s^3
            
            grad_f = grad_soft / (s * s) - 2.0 * soft_max / (s * s * s)
            
            # Update
            x_new = x - lr * grad_f
            
            # Project to signed constraint? 
            # The task asks for "signed step heights". 
            # The seed program uses +/- 1. 
            # If we allow continuous values, the "signed step" might imply we should threshold.
            # However, the objective is defined on the sequence. 
            # If the evaluator checks for +/- 1, we must project.
            # "sequence of signed step heights" usually implies discrete signs in this context (Rademacher).
            # Let's project to +/- 1 by sign() after each step or at the end.
            # Projecting at the end might lose gradient info.
            # Let's try projecting at the end of the optimization loop.
            
            x = x_new
            
            # Optional: Project to +/- 1 to stay in the feasible set if required.
            # If we project, the gradient step is not valid for the next step on the manifold.
            # But for a heuristic, it's okay.
            # Let's keep it continuous for now and project at the end.
            # Actually, if we keep it continuous, the "signed step" description might be violated if the evaluator expects exactly +/-1.
            # Let's assume the evaluator computes the score on the returned array.
            # If I return continuous values, is it valid? "signed step heights" is ambiguous.
            # Given the seed uses +/-1, I should probably return +/-1.
            # I will optimize continuously and then snap to +/-1.
            
            # Increase beta to sharpen the max approximation
            beta *= 1.01
            
        # Snap to signs
        x_signed = np.sign(x)
        # Handle zeros
        x_signed[x_signed == 0] = 1.0
        
        sc = _objective(x_signed)
        if sc < best_score:
            best_score = sc
            best_x = x_signed.copy()
            
        # Local search on the best found so far
        if time.time() < deadline:
            x_local = best_x.copy()
            improved = True
            max_iter = 1000
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if not improved:
                    break
                improved = False
                # Random flips
                idx = rng.randint(N)
                x_local[idx] = -x_local[idx]
                sc2 = _objective(x_local)
                if sc2 < best_score:
                    best_score = sc2
                    best_x = x_local.copy()
                    improved = True
                else:
                    x_local[idx] = -x_local[idx]
                    
    return best_x
# EVOLVE-BLOCK-END
