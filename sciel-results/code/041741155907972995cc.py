"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _score(h):
    """The objective: 1 - min(R_k). Lower is better."""
    corrs = _compute_overlaps(h)
    return 1.0 - np.min(corrs)

def _project_to_simplex(h, target_sum):
    """Project h to satisfy sum(h) = target_sum and 0<=h<=1."""
    h = np.clip(h, 0, 1)
    diff = target_sum - np.sum(h)
    for _ in range(30):
        if abs(diff) < 1e-13:
            break
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        if np.sum(mask) == 0:
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
        diff = target_sum - np.sum(h)
    # Final exact adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute remaining diff to elements not at bounds
        mask = (h > 1e-12) & (h < 1 - 1e-12)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
        else:
            h += diff / len(h)
            h = np.clip(h, 0, 1)
    # Ensure exact sum with floating point
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-15:
        h[0] += diff
        h[0] = np.clip(h[0], 0, 1)
    return h

def _soft_min_subgrad(h, beta=50.0):
    """
    Compute subgradient of soft-min of R_k values.
    soft_min = -log(sum exp(-beta*R_k))/beta
    d(soft_min)/d(h_j) = -sum_k w_k * dR_k/d(h_j) / beta
    where w_k = exp(-beta*R_k) / sum exp(-beta*R_k)
    
    Since we want to MINIMIZE soft_min (which approximates min R_k),
    we need the gradient of soft_min to do gradient ascent,
    or equivalently gradient descent on -soft_min.
    
    Actually, let's think about this differently.
    We want to maximize min(R_k), which is equivalent to minimizing -min(R_k).
    The score is 1 - min(R_k), so minimizing score = maximizing min(R_k).
    
    Let f(h) = min_k R_k. We want to maximize f(h).
    Gradient of f at h points in direction that increases the minimum.
    
    Using soft-min: g(h) = -(1/beta) log(sum_k exp(-beta R_k))
    g approximates min_k R_k for large beta.
    dg/dh_j = sum_k w_k * dR_k/dh_j where w_k = exp(-beta R_k) / sum_k exp(-beta R_k)
    
    So to maximize g (and thus min R_k), we do gradient ascent on g:
    h_new = h + lr * dg/dh
    """
    n = len(h)
    corrs = _compute_overlaps(h)
    
    # Soft-min weights
    min_c = np.min(corrs)
    exps = np.exp(-beta * (corrs - min_c))
    w = exps / np.sum(exps)
    
    # Compute gradient of soft-min w.r.t. h
    grad = np.zeros(n)
    for k in range(1, n):
        wk = w[k-1]
        if wk < 1e-15:
            continue
        # dR_k/dh_j = (2/n) * h[j+k] for j < n-k
        # dR_k/dh_j = (2/n) * h[j-k] for j >= k
        factor = (2.0 / n) * wk
        grad[:n-k] += factor * h[k:]
        grad[k:] += factor * h[:n-k]
    
    return grad

def _max_overlap_subgrad(h):
    """
    Compute subgradient of max_k R_k w.r.t. h.
    We want to minimize max_k R_k, so we do gradient descent.
    """
    n = len(h)
    corrs = _compute_overlaps(h)
    max_val = np.max(corrs)
    
    # Find which k achieves max (or near-max for smoothing)
    threshold = max_val - 1e-4
    active = corrs >= threshold
    
    subgrad = np.zeros(n)
    for k in range(1, n):
        if not active[k-1]:
            continue
        factor = (2.0 / n)
        subgrad[:n-k] += factor * h[k:]
        subgrad[k:] += factor * h[:n-k]
    
    # Normalize subgradient
    norm = np.linalg.norm(subgrad)
    if norm > 1e-15:
        subgrad /= norm
    
    return max_val, subgrad

def construct():
    n = N
    target_sum = n / 2.0
    
    best_h = np.full(n, 0.5)
    best_score = _score(best_h)
    
    np.random.seed(42)
    
    # Strategy 1: Maximize min(R_k) using soft-min gradient ascent
    # with projection to feasible set
    
    # Initialize with various patterns
    initializations = []
    
    # Uniform
    initializations.append(np.full(n, 0.5))
    
    # Periodic patterns
    for period in [2, 3, 4, 5, 7, 10]:
        h_init = np.zeros(n)
        for i in range(n):
            h_init[i] = 0.5 + 0.3 * np.cos(2 * np.pi * i / period)
        h_init = _project_to_simplex(h_init, target_sum)
        initializations.append(h_init)
    
    # Random smooth patterns
    for trial in range(10):
        h_init = 0.5 + 0.3 * np.random.randn(n)
        # Smooth it
        h_init = np.convolve(h_init, np.ones(5)/5, mode='same')
        h_init = _project_to_simplex(h_init, target_sum)
        initializations.append(h_init)
    
    # Fourier-based initializations
    for n_harm in [1, 2, 3]:
        for trial in range(5):
            coeffs = np.zeros(n, dtype=complex)
            for k in range(1, n_harm+1):
                amp = np.random.uniform(0.1, 0.35)
                phase = np.random.uniform(0, 2*np.pi)
                coeffs[k] = amp * np.exp(1j * phase)
            h_init = np.full(n, 0.5)
            for k in range(1, n_harm+1):
                h_init += np.real(coeffs[k] * np.exp(2j * np.pi * k * np.arange(n) / n))
            h_init = _project_to_simplex(h_init, target_sum)
            initializations.append(h_init)
    
    # Optimize each initialization
    for h0 in initializations:
        if _time_remaining() < 5.0:
            break
        
        h = h0.copy()
        current_score = _score(h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Adaptive learning rate
        lr = 0.02
        patience = 0
        
        for step in range(300):
            if _time_remaining() < 2.0:
                break
            
            # Compute soft-min gradient
            grad = _soft_min_subgrad(h, beta=30.0)
            
            # Gradient ascent step (to maximize min R_k)
            h_new = h + lr * grad
            h_new = _project_to_simplex(h_new, target_sum)
            
            new_score = _score(h_new)
            
            if new_score < current_score:
                h = h_new
                current_score = new_score
                patience = 0
                lr = min(lr * 1.02, 0.05)
                
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                patience += 1
                lr *= 0.95
                if lr < 1e-6 or patience > 50:
                    break
        
        # Polish with max-overlap subgradient descent
        if _time_remaining() > 3.0:
            h_polish = h.copy()
            lr_p = 0.01
            for step in range(100):
                if _time_remaining() < 1.0:
                    break
                
                max_val, subgrad = _max_overlap_subgrad(h_polish)
                
                # We want to minimize max overlap, but our score is 1 - min overlap
                # So actually we want to maximize min overlap.
                # Let's use a different approach: minimize the spread of overlaps
                # Or just do local search
                
                # Local search: try small perturbations
                improved = False
                for _ in range(5):
                    i = np.random.randint(n)
                    delta = np.random.uniform(-0.05, 0.05)
                    h_trial = h_polish.copy()
                    h_trial[i] += delta
                    h_trial = _project_to_simplex(h_trial, target_sum)
                    
                    trial_score = _score(h_trial)
                    if trial_score < current_score:
                        h_polish = h_trial
                        current_score = trial_score
                        improved = True
                        if trial_score < best_score:
                            best_score = trial_score
                            best_h = h_polish.copy()
                        break
                
                if not improved:
                    break
    
    # Strategy 2: Try bang-bang (0/1) patterns with local search
    if _time_remaining() > 15.0:
        for trial in range(20):
            if _time_remaining() < 3.0:
                break
            
            # Random bang-bang with correct sum
            h_bb = np.zeros(n)
            indices = np.random.permutation(n)
            h_bb[indices[:n//2]] = 1.0
            h_bb = _project_to_simplex(h_bb, target_sum)
            
            score = _score(h_bb)
            if score < best_score:
                best_score = score
                best_h = h_bb.copy()
            
            # Local search: flip individual bits
            for step in range(150):
                if _time_remaining() < 1.5:
                    break
                
                i = np.random.randint(n)
                h_trial = h_bb.copy()
                h_trial[i] = 1.0 - h_trial[i]
                h_trial = _project_to_simplex(h_trial, target_sum)
                
                new_score = _score(h_trial)
                if new_score < score:
                    h_bb = h_trial
                    score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_h = h_bb.copy()
                else:
                    # Try adjacent swap
                    if i < n - 1:
                        h_trial2 = h_bb.copy()
                        h_trial2[i], h_trial2[i+1] = h_trial2[i+1], h_trial2[i]
                        h_trial2 = _project_to_simplex(h_trial2, target_sum)
                        new_score2 = _score(h_trial2)
                        if new_score2 < score:
                            h_bb = h_trial2
                            score = new_score2
                            if new_score2 < best_score:
                                best_score = new_score2
                                best_h = h_bb.copy()
    
    # Final polish: fine-tune the best solution
    if _time_remaining() > 5.0:
        h_final = best_h.copy()
        current_score = _score(h_final)
        
        # Very fine local search
        for step in range(500):
            if _time_remaining() < 1.0:
                break
            
            improved = False
            for _ in range(10):
                i = np.random.randint(n)
                delta = np.random.uniform(-0.01, 0.01)
                h_trial = h_final.copy()
                h_trial[i] += delta
                h_trial = _project_to_simplex(h_trial, target_sum)
                
                trial_score = _score(h_trial)
                if trial_score < current_score:
                    h_final = h_trial
                    current_score = trial_score
                    improved = True
                    if trial_score < best_score:
                        best_score = trial_score
                        best_h = h_final.copy()
                    break
            
            if not improved:
                break
    
    return best_h
# EVOLVE-BLOCK-END
