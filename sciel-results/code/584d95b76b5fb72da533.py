"""
Strategy: Direct Min-Max Convex Optimization via Projected Subgradient Descent.
Instead of assuming a periodic block structure (3D parameter space), we treat the
100 step heights as independent variables in a 100D space. The objective is the
maximum overlap over all shifts k, which is a convex function of h (max of linear
functions). We use projected subgradient descent on the feasible set {h in [0,1]^N,
sum(h)=N/2}. The projection onto this set is done via bisection on a threshold
for the clipped variables. This allows the optimizer to find non-periodic,
globally better solutions that the parametric approach might miss.
"""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
    Uses bisection on the threshold tau such that sum(clip(h - tau, 0, 1)) = N/2.
    """
    target_sum = N / 2.0
    
    # If h is already feasible, return it
    if np.all(h >= -1e-9) and np.all(h <= 1 + 1e-9) and abs(np.sum(h) - target_sum) < 1e-9:
        return np.clip(h, 0, 1)
    
    # Bisection bounds
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    # If target is outside the range of possible sums, handle edge cases
    # Min sum is 0, max sum is N. Target is N/2, which is in [0, N].
    
    for _ in range(50):  # Sufficient precision for double
        mid = (lo + hi) / 2.0
        # Compute sum of projected variables
        h_proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        
        if s > target_sum:
            lo = mid
        else:
            hi = mid
        
        if abs(hi - lo) < 1e-12:
            break
    
    tau = (lo + hi) / 2.0
    return np.clip(h - tau, 0.0, 1.0)

def _compute_overlap_vector(h):
    """
    Compute the overlap vector for all shifts k.
    Overlap(k) = sum_i h[i] * (1 - h[(i+k) % N]) = N/2 - autocorr[k]
    where autocorr is the circular autocorrelation of h.
    """
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    overlap = N / 2.0 - autocorr
    return overlap

def _objective_and_grad(h):
    """
    Compute the objective (max overlap) and its subgradient.
    Returns: (obj_val, grad)
    """
    overlap = _compute_overlap_vector(h)
    max_idx = np.argmax(overlap)
    obj_val = overlap[max_idx]
    
    # The subgradient of max_k overlap(k) with respect to h is the gradient of overlap(max_idx).
    # overlap(k) = sum_i h[i] - sum_i h[i]h[i+k]
    # d(overlap(k))/dh[i] = 1 - h[(i+k) % N]
    
    k = max_idx
    # grad[i] = 1 - h[(i+k) % N]
    shifted_h = np.roll(h, -k)  # shifted_h[i] = h[(i+k) % N]
    grad = 1.0 - shifted_h
    
    return obj_val, grad

def _subgradient_descent(h0, max_iter=5000, lr_init=0.01, lr_decay=0.999, tol=1e-10):
    """
    Projected subgradient descent for minimizing max overlap.
    """
    h = h0.copy()
    h = _project_to_feasible(h)
    
    best_h = h.copy()
    best_obj = _objective_and_grad(h)[0]
    
    lr = lr_init
    obj_prev = best_obj
    
    for i in range(max_iter):
        obj_val, grad = _objective_and_grad(h)
        
        if obj_val < best_obj:
            best_obj = obj_val
            best_h = h.copy()
        
        # Check convergence
        if i > 10 and abs(obj_prev - obj_val) < tol:
            break
        obj_prev = obj_val
        
        # Update: h_new = h - lr * grad
        h = h - lr * grad
        
        # Project back to feasible set
        h = _project_to_feasible(h)
        
        # Decay learning rate
        lr *= lr_decay
        
        if lr < 1e-8:
            break
    
    return best_h, best_obj

def _random_feasible():
    """
    Generate a random feasible point: h in [0,1]^N, sum(h)=N/2.
    """
    # Generate random points and project
    h = np.random.rand(N)
    return _project_to_feasible(h)

def _structured_initial(h_type='block'):
    """
    Generate structured initial points to guide the search.
    """
    if h_type == 'block':
        # Alternating blocks
        h = np.zeros(N)
        h[::2] = 1.0
        # Adjust to sum N/2 (already is if N is even)
        return _project_to_feasible(h)
    elif h_type == 'smooth':
        # Smooth sine wave
        x = np.linspace(0, 2 * np.pi, N)
        h = 0.5 + 0.5 * np.sin(x)
        return _project_to_feasible(h)
    elif h_type == 'ramp':
        # Ramp up then down
        h = np.linspace(0, 1, N)
        h[N//2:] = np.linspace(1, 0, N - N//2)
        return _project_to_feasible(h)
    else:
        return _random_feasible()

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 second safety margin
    
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # Phase 1: Try several structured and random initializations with subgradient descent
    initializations = []
    for _ in range(5):
        initializations.append(_random_feasible())
    initializations.append(_structured_initial('block'))
    initializations.append(_structured_initial('smooth'))
    initializations.append(_structured_initial('ramp'))
    
    # Also try some specific patterns
    # Periodic patterns with different periods
    for p in [2, 4, 5, 10, 20, 25, 50]:
        h = np.zeros(N)
        for i in range(N):
            if (i % p) < p // 2:
                h[i] = 1.0
        h = _project_to_feasible(h)
        initializations.append(h)
    
    # Shuffle to ensure diversity
    np.random.shuffle(initializations)
    
    for h0 in initializations:
        if time.time() > deadline - 5.0:
            break
        
        try:
            h_opt, score_opt = _subgradient_descent(h0, max_iter=2000, lr_init=0.05, lr_decay=0.9995)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    # Phase 2: If time remains, do a second round with smaller learning rates and more iterations
    if time.time() < deadline - 10.0 and best_h is not None:
        # Refine the best solution
        h_refined, score_refined = _subgradient_descent(best_h, max_iter=5000, lr_init=0.005, lr_decay=0.9998)
        
        if score_refined < best_score:
            best_score = score_refined
            best_h = h_refined.copy()
    
    # Phase 3: Local search with coordinate descent on the best solution
    if time.time() < deadline - 5.0 and best_h is not None:
        h_local = best_h.copy()
        for _ in range(1000):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline - 2.0:
                    break
                
                # Try increasing h[i]
                old_val = h_local[i]
                
                # Max possible increase: min(1 - old_val, remaining sum)
                # But we need to maintain sum = N/2, so we must decrease another element.
                # This is complex for coordinate descent. Instead, let's use a simple perturbation.
                
                # Try small random perturbations
                for _ in range(10):
                    h_test = h_local.copy()
                    # Randomly pick two indices and swap a small amount
                    j = np.random.randint(N)
                    if j == i:
                        continue
                    delta = np.random.uniform(-0.01, 0.01)
                    
                    h_test[i] += delta
                    h_test[j] -= delta
                    
                    # Project to feasible set
                    h_test = _project_to_feasible(h_test)
                    
                    # Check if sum is still N/2 (projection ensures this)
                    score_test = _objective_and_grad(h_test)[0]
                    
                    if score_test < best_score - 1e-12:
                        h_local = h_test
                        best_score = score_test
                        best_h = h_test.copy()
                        improved = True
                        break
            
            if not improved:
                break
    
    # Final check and projection
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Ensure sum is exactly N/2 and bounds are respected
    best_h = _project_to_feasible(best_h)
    
    # Verify sum is exactly N/2
    current_sum = np.sum(best_h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    if abs(diff) > 1e-12:
        # Adjust the largest element to absorb the difference
        if diff > 0:
            idx = np.argmin(best_h)
            best_h[idx] += diff
        else:
            idx = np.argmax(best_h)
            best_h[idx] += diff
        best_h = np.clip(best_h, 0, 1)
    
    return best_h

# EVOLVE-BLOCK-END
