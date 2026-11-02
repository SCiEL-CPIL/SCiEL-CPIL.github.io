import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = time.time() + 140.0  # 10s margin before 150s kill

def _time_remaining():
    return max(0.0, DEADLINE - time.time())

def _compute_overlaps(h):
    """
    Compute the raw overlap values for shifts k=1 to N-1.
    Overlap_k = sum_{i=0}^{N-k-1} h_i (1 - h_{i+k})
             = sum h_i - sum h_i h_{i+k}
    Since sum h_i = N/2, Overlap_k = N/2 - sum h_i h_{i+k}
    We want to minimize max_k Overlap_k.
    """
    n = len(h)
    overlaps = np.zeros(n - 1)
    for k in range(1, n):
        # Linear shift (non-circular) based on inspiration
        # h_shifted is h[k:]
        # h is h[:n-k]
        dot = np.dot(h[:n-k], h[k:])
        overlaps[k-1] = np.sum(h[:n-k]) - dot
    return overlaps

def _score(h):
    """
    Returns the value to minimize: max overlap.
    Also checks validity.
    """
    if not np.all(np.isfinite(h)):
        return float('inf')
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return float('inf')
    if abs(np.sum(h) - N/2.0) > 1e-6:
        return float('inf')
    if N == 1:
        return 0.0
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _project_to_constraints(h, target_sum=N/2.0):
    """
    Project h to satisfy sum(h) = target_sum and 0 <= h <= 1.
    Uses a simple iterative projection.
    """
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    
    # Distribute diff to elements not at bounds
    for _ in range(100):
        mask = (h > 0) & (h < 1)
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-12:
            break
            
    # If still not close enough, try adjusting all
    if abs(diff) > 1e-9:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        
    return h

def construct():
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    if best_score == float('inf'):
        return best_h
        
    # Generate diverse starting points
    starts = []
    
    # 1. Constant
    starts.append(np.full(N, 0.5))
    
    # 2. Linear gradients
    starts.append(np.linspace(0, 1, N))
    starts.append(np.linspace(1, 0, N))
    
    # 3. Sinusoids
    x = np.linspace(0, 2 * np.pi, N)
    starts.append(0.5 + 0.4 * np.sin(x))
    starts.append(0.5 + 0.4 * np.cos(x))
    starts.append(0.5 + 0.3 * np.sin(2 * x))
    starts.append(0.5 + 0.3 * np.cos(2 * x))
    starts.append(0.5 + 0.2 * np.sin(3 * x))
    
    # 4. Block patterns
    starts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(N)]))
    starts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)]))
    starts.append(np.array([0.0 if i % 3 == 0 else 0.75 for i in range(N)]))
    
    # 5. Random starts
    np.random.seed(42)
    for _ in range(20):
        starts.append(np.random.uniform(0, 1, N))
    
    # 6. Structured random (low freq noise)
    for _ in range(10):
        low_freq = np.random.uniform(-0.5, 0.5, 5)
        h_rand = np.interp(np.linspace(0, N, N), np.linspace(0, 4, 5), low_freq)
        h_rand = 0.5 + h_rand
        starts.append(h_rand)

    # Optimization using L-BFGS-B with a smooth approximation of the max function
    # We minimize: sum(exp(beta * (overlap_k - max_overlap))) / beta + max_overlap?
    # Or simpler: minimize sum(exp(beta * overlap_k)) / beta
    # This approximates max(overlap_k) as beta -> infinity.
    
    def _obj_and_grad(h, beta):
        h = np.clip(h, 0, 1) # Ensure bounds during intermediate steps for stability
        overlaps = _compute_overlaps(h)
        
        # Soft max approximation
        # To avoid overflow, subtract max
        max_o = np.max(overlaps)
        exp_overlaps = np.exp(beta * (overlaps - max_o))
        sum_exp = np.sum(exp_overlaps)
        soft_max = max_o + np.log(sum_exp) / beta
        
        # Gradient
        # d(soft_max)/d(overlap_k) = exp(beta * (overlap_k - max_o)) / sum_exp
        weights = exp_overlaps / sum_exp
        
        # d(overlap_k)/d(h)
        # overlap_k = sum_{i=0}^{N-k-1} h_i - h_i h_{i+k}
        # d(overlap_k)/d(h_j) = 1 - h_{j+k} (if j < N-k) - h_{j-k} (if j >= k)
        # Note: h_j appears in the sum term and potentially in the product term.
        
        grad_h = np.zeros(N)
        for k in range(1, N):
            w = weights[k-1]
            if w == 0: continue
            
            # Term 1: sum h_i
            # d/dh_j (sum_{i=0}^{N-k-1} h_i) = 1 if j < N-k else 0
            grad_h[:N-k] += w * 1.0
            
            # Term 2: - sum h_i h_{i+k}
            # d/dh_j (- sum h_i h_{i+k})
            # If j is i: -h_{j+k}
            # If j is i+k: -h_{j-k}
            
            # j is i: j in [0, N-k-1]
            if N-k > 0:
                grad_h[:N-k] -= w * h[k:]
            
            # j is i+k: j in [k, N-1]
            if k < N:
                grad_h[k:] -= w * h[:N-k]
                
        return soft_max, grad_h

    # Multi-stage beta optimization
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for start in starts:
        if _time_remaining() < 5.0:
            break
            
        h0 = _project_to_constraints(start)
        current_h = h0.copy()
        current_score = _score(current_h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
            
        # Run optimization for increasing beta
        for beta in betas:
            if _time_remaining() < 2.0:
                break
                
            try:
                res = minimize(
                    _obj_and_grad,
                    current_h,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * N,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                
                h_new = res.x
                h_new = _project_to_constraints(h_new)
                new_score = _score(h_new)
                
                if new_score < current_score:
                    current_h = h_new
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_h = h_new.copy()
                        
            except Exception:
                pass

    # Final local search / polish if time remains
    if _time_remaining() > 10.0:
        # Try a few random perturbations of the best solution
        for _ in range(20):
            if _time_remaining() < 2.0:
                break
            h_perturbed = best_h + np.random.normal(0, 0.01, N)
            h_perturbed = _project_to_constraints(h_perturbed)
            
            # Quick optimization
            try:
                res = minimize(
                    _obj_and_grad,
                    h_perturbed,
                    args=(200.0,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                h_new = res.x
                h_new = _project_to_constraints(h_new)
                new_score = _score(h_new)
                
                if new_score < best_score:
                    best_score = new_score
                    best_h = h_new.copy()
            except Exception:
                pass

    # Ensure final result is valid
    best_h = _project_to_constraints(best_h)
    if _score(best_h) == float('inf'):
        return np.full(N, 0.5)
        
    return best_h
