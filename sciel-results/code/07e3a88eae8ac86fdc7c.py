"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1]."""
    h = h.copy()
    for _ in range(20):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    s = np.sum(h)
    diff = N/2.0 - s
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _get_basis(K):
    """
    Precompute the basis matrix for Fourier modes 1..K.
    Basis shape: (N, 2*K)
    Columns 2k-1: cos(2 pi k i / N)
    Columns 2k: sin(2 pi k i / N)
    """
    i = np.arange(N)
    B = np.zeros((N, 2*K))
    for k in range(1, K + 1):
        B[:, 2*(k-1)] = np.cos(2 * np.pi * k * i / N)
        B[:, 2*(k-1)+1] = np.sin(2 * np.pi * k * i / N)
    return B

def _project_to_coefficients(h, B):
    """
    Project h onto the column space of B (Fourier subspace) via least squares.
    Returns coefficients c such that B @ c is the best approximation of h.
    """
    # Pseudo-inverse projection: c = (B^T B)^-1 B^T h
    # Since B has orthogonal columns (for integer k < N/2), B^T B is diagonal.
    # Specifically, sum cos^2 = N/2, sum sin^2 = N/2.
    # So c_j = (2/N) sum h_i B_ij
    c = np.zeros(B.shape[1])
    for j in range(B.shape[1]):
        c[j] = (2.0 / N) * np.dot(h, B[:, j])
    return c

def _pgd_fourier(h_init, K, max_iter=500, alpha=100.0):
    """
    Projected Gradient Descent on Fourier coefficients.
    """
    B = _get_basis(K)
    
    # Initial coefficients
    c = _project_to_coefficients(h_init, B)
    
    # Precompute B^T for gradient calculation
    BT = B.T
    
    best_h = h_init.copy()
    best_score = _score(h_init)
    
    step_size = 0.01
    min_step_size = 1e-5
    
    for it in range(max_iter):
        # Reconstruct h
        h = 0.5 * np.ones(N) + B @ c
        
        # Calculate overlap
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        
        # Score for tracking
        current_score = np.max(overlap)
        if current_score < best_score:
            best_score = current_score
            best_h = _normalize(h)
        
        # Gradient of smooth max w.r.t h
        # d/dh_i [ smooth_max(overlap) ]
        # overlap_j = h_j (1 - h_{j+1})
        # d overlap_j / d h_i:
        # if i == j: (1 - h_{j+1})
        # if i == j+1: -h_j
        # Note: h is treated as independent variables for the gradient step before projection.
        
        exp_term = np.exp(alpha * (overlap - np.max(overlap)))
        weights = exp_term / np.sum(exp_term)
        
        grad_h = np.zeros(N)
        # d overlap_j / d h_j = 1 - h_{j+1}
        # Note: h_{N} is 0 in the shift logic? 
        # In _score, shifted[:-1] = h[1:], shifted[N-1]=0.
        # So overlap[N-1] = h[N-1] * (1 - 0) = h[N-1].
        # d overlap[N-1] / d h[N-1] = 1.
        # d overlap[j] / d h[j] = 1 - h[j+1] for j < N-1.
        # d overlap[j] / d h[j+1] = -h[j] for j < N-1.
        
        # Vectorized:
        # term1: contribution to h[j] from overlap[j]
        # d overlap[j] / d h[j] = 1 - h[j+1]
        # We need h[j+1] for j=0..N-2, and 0 for j=N-1.
        h_shifted_for_grad = np.zeros(N)
        h_shifted_for_grad[:-1] = h[1:]
        d_overlap_d_h_self = 1.0 - h_shifted_for_grad
        
        grad_h += weights * d_overlap_d_h_self
        
        # term2: contribution to h[j+1] from overlap[j]
        # d overlap[j] / d h[j+1] = -h[j]
        # This affects h[1]...h[N-1]
        # grad_h[j] gets contribution from overlap[j-1]
        # d overlap[j-1] / d h[j] = -h[j-1]
        
        # So for j=1..N-1: grad_h[j] += weights[j-1] * (-h[j-1])
        if N > 1:
            grad_h[1:] += -weights[:-1] * h[:-1]
            
        # Gradient w.r.t coefficients c
        # h = 0.5 * 1 + B c
        # d h / d c = B
        # grad_c = B^T grad_h
        grad_c = BT @ grad_h
        
        # Step
        c_new = c - step_size * grad_c
        
        # Reconstruct h_new
        h_new = 0.5 * np.ones(N) + B @ c_new
        
        # Project h_new to feasible set
        h_proj = _normalize(h_new)
        
        # Map h_proj back to coefficients
        c = _project_to_coefficients(h_proj, B)
        
        # Adaptive step size
        score_proj = _score(h_proj)
        if score_proj < current_score:
            step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.8
            if step_size < min_step_size:
                break
                
    return best_h, best_score

def _local_polish(h_init, maxiter=500):
    """
    Full local polish on the N-dimensional vector using SLSQP and smooth max.
    """
    def _smooth_objective(params, alpha):
        h = params
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha=alpha)
        sum_penalty = 100.0 * (np.sum(h) - N/2.0)**2
        return smax + sum_penalty

    def _constraint_sum(params):
        return np.sum(params) - N/2.0

    constraints = [{'type': 'eq', 'fun': _constraint_sum}]
    bounds = [(0.0, 1.0)] * N

    result = minimize(
        lambda x: _smooth_objective(x, 200.0),
        h_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False}
    )
    
    if result.fun < 1.0:
        h_opt = _normalize(result.x)
        return h_opt, _score(h_opt)
    return None, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # 1. Generate diverse initial guesses
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    
    candidates = [h0]
    
    # Add random smooth candidates
    np.random.seed(42)
    for _ in range(3):
        h_rand = 0.5 + 0.2 * np.sin(np.random.uniform(0, 2*np.pi) + np.arange(N)*0.1)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = _normalize(h_rand)
        candidates.append(h_rand)

    best_h = h0.copy()
    best_score = _score(h0)

    # 2. Try PGD with different Fourier ranks
    # This is the structural change: low-dimensional parameterization + PGD
    
    for K in [5, 10, 15]:
        try:
            if time.time() < deadline - 30.0:
                h_pgd, s_pgd = _pgd_fourier(best_h, K, max_iter=1000, alpha=100.0)
                if s_pgd < best_score:
                    best_h = h_pgd
                    best_score = s_pgd
        except Exception:
            pass

    # 3. Local Polish on the best candidate found
    try:
        if time.time() < deadline - 5.0:
            h_polished, s_polished = _local_polish(best_h, maxiter=1000)
            if s_polished < best_score:
                best_h = h_polished
                best_score = s_polished
    except Exception:
        pass

    return best_h
