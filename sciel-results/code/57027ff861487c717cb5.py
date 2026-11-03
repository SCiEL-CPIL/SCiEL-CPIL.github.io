"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _all_corrs_fft(h):
    """Compute all non-circular cross-correlations R(k) for k=1..n-1 using FFT.
    R(k) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    """
    n = len(h)
    size = 1
    while size < 2 * n:
        size *= 2
    H = np.fft.rfft(h, size)
    corr_full = np.fft.irfft(H * np.conj(H), size)
    corrs = (2.0 / n) * corr_full[1:n]
    return corrs

def _max_overlap(h):
    """Compute the maximum overlap value = 1 - min_k R(k)."""
    corrs = _all_corrs_fft(h)
    return 1.0 - np.min(corrs)

def _project(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    w = np.clip(w, 0, 1)
    # Iterative adjustment for box constraints
    for _ in range(50):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    return w

def _soft_min_obj_grad_fft(h, tau):
    """Compute soft-min of correlations and its gradient using FFT.
    Returns (soft_min_value, gradient_of_soft_min).
    soft_min = -tau * log(sum(exp(-R(k)/tau)))
    gradient = sum(exp(-R(k)/tau) * grad_R(k)) / sum(exp(-R(k)/tau))
    """
    n = len(h)
    corrs = _all_corrs_fft(h)
    # For numerical stability
    min_corr = np.min(corrs)
    exp_vals = np.exp(-(corrs - min_corr) / tau)
    Z = np.sum(exp_vals)
    soft_min_val = min_corr - tau * np.log(Z)
    
    # Gradient of soft min w.r.t. h
    # grad of R(k) = (2/n) * (h shifted by k + h shifted by -k)
    # We need sum_k p_k * grad_R(k) where p_k = exp_vals[k]/Z
    weights = exp_vals / Z
    grad = np.zeros(n)
    for k in range(1, n):
        w_k = weights[k - 1]  # corrs[0] corresponds to k=1
        if w_k < 1e-15:
            continue
        grad[:n-k] += (2.0/n) * w_k * h[k:]
        grad[k:] += (2.0/n) * w_k * h[:n-k]
    return soft_min_val, grad

def _epigraph_obj_grad(h, t, tau):
    """Epigraph formulation: minimize t + penalty for R(k) < t.
    Returns (objective, grad_h, grad_t).
    """
    n = len(h)
    corrs = _all_corrs_fft(h)
    # Soft violation: sum of softplus(t - R(k))
    diffs = t - corrs
    # softplus(x) = log(1+exp(x)), for large x approx x
    sp = np.where(diffs > 50, diffs, np.log1p(np.exp(diffs)))
    penalty = np.sum(sp)
    obj = t + penalty
    
    # Gradient w.r.t. t
    # d/dt softplus(t - R(k)) = sigmoid(t - R(k))
    sig = 1.0 / (1.0 + np.exp(-np.clip(diffs, -50, 50)))
    grad_t = 1.0 + np.sum(sig)
    
    # Gradient w.r.t. h
    # d/dh softplus(t - R(k)) = -sigmoid(t - R(k)) * dR(k)/dh
    grad_h = np.zeros(n)
    for k in range(1, n):
        s_k = sig[k - 1]
        if s_k < 1e-15:
            continue
        grad_h[:n-k] -= (2.0/n) * s_k * h[k:]
        grad_h[k:] -= (2.0/n) * s_k * h[:n-k]
    
    return obj, grad_h, grad_t

def _subgradient_step(h, n_iter, lr_init, target_sum, use_softmin=True, tau_init=0.05):
    """Subgradient ascent to maximize min_k R(k), with projection."""
    n = len(h)
    h = h.copy()
    lr = lr_init
    best_obj = _max_overlap(h)
    best_h = h.copy()
    patience = 0
    tau = tau_init
    
    for it in range(n_iter):
        if _time_remaining() < 2.0:
            break
        
        if use_softmin:
            # Decay tau over time
            tau = max(0.005, tau_init * (1.0 - it / n_iter) + 0.005)
            val, grad = _soft_min_obj_grad_fft(h, tau)
            # We want to maximize min R, so we ascend in direction of grad
            # (grad points in direction of increasing soft_min, which approximates min R)
            step = grad
        else:
            corrs = _all_corrs_fft(h)
            min_idx = np.argmin(corrs)
            tol = 0.02
            active = np.where(corrs < corrs[min_idx] + tol)[0]
            grad = np.zeros(n)
            for idx in active:
                k = idx + 1
                grad[:n-k] += (2.0/n) * h[k:]
                grad[k:] += (2.0/n) * h[:n-k]
            grad /= len(active)
            step = grad
        
        gnorm = np.linalg.norm(step)
        if gnorm > 1e-10:
            step = step / gnorm
        
        h_new = h + lr * step
        h_new = _project(h_new, target_sum)
        obj = _max_overlap(h_new)
        if obj < best_obj - 1e-12:
            best_obj = obj
            best_h = h_new.copy()
            patience = 0
        else:
            patience += 1
            if patience > 30:
                lr *= 0.95
                patience = 0
        h = h_new
    
    return best_h, best_obj

def _coordinate_polish(h, n_sweeps, target_sum):
    """Coordinate descent: for each coordinate, find optimal value given others."""
    n = len(h)
    h = h.copy()
    best_obj = _max_overlap(h)
    
    for sweep in range(n_sweeps):
        if _time_remaining() < 2.0:
            break
        improved = False
        for j in range(n):
            if _time_remaining() < 2.0:
                break
            old_val = h[j]
            # Try candidate values
            candidates = [0.0, 1.0]
            if 0 < old_val < 1:
                candidates.extend([old_val * 0.5, old_val * 1.5, 0.25, 0.75, 0.1, 0.9])
            
            best_cand_obj = best_obj
            best_cand_val = old_val
            for cv in candidates:
                h[j] = cv
                h_proj = _project(h, target_sum)
                obj_proj = _max_overlap(h_proj)
                if obj_proj < best_cand_obj - 1e-13:
                    best_cand_obj = obj_proj
                    best_cand_val = cv
                    h = h_proj.copy()
                h[j] = old_val
            
            if best_cand_val != old_val:
                h[j] = best_cand_val
                h = _project(h, target_sum)
                best_obj = _max_overlap(h)
                improved = True
        
        if not improved and sweep > 1:
            break
    
    return h, best_obj

def _local_refine(h, target_sum, n_refine=500):
    """Local refinement using epigraph formulation with smooth optimization."""
    n = len(h)
    h = h.copy()
    t = _max_overlap(h)  # Initial t = current max overlap
    
    # Simple gradient-based refinement
    lr_t = 0.01
    lr_h = 0.005
    tau = 0.1
    
    for it in range(n_refine):
        if _time_remaining() < 2.0:
            break
        
        obj, grad_h, grad_t = _epigraph_obj_grad(h, t, tau)
        
        # Update t to minimize
        t_new = t - lr_t * grad_t
        # Update h to minimize (so we descend)
        h_new = h - lr_h * grad_h
        h_new = _project(h_new, target_sum)
        
        # Check improvement in actual objective
        obj_new = _max_overlap(h_new)
        if obj_new < _max_overlap(h):
            h = h_new
            t = obj_new
            # Decay learning rates
            lr_t *= 0.99
            lr_h *= 0.99
            tau *= 0.995
        else:
            # Revert and reduce step
            lr_t *= 0.9
            lr_h *= 0.9
            tau = max(0.01, tau * 0.9)
    
    return h, _max_overlap(h)

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    np.random.seed(42)
    
    # Generate diverse initializations
    restarts = []
    restarts.append(np.full(n, 0.5))
    
    # Block structures
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = int(n * frac)
        if 0 < k < n:
            v1 = 2.0 * target_sum / (n - k) if n - k > 0 else 0
            v2 = 2.0 * target_sum / k if k > 0 else 0
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, min(v1, 1.0))]))
            restarts.append(np.concatenate([np.full(k, min(v2, 1.0)), np.full(n-k, 0.0)]))
    
    # Alternating
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    
    # Random
    for _ in range(20):
        restarts.append(np.random.uniform(0, 1, n))
    
    # Smooth
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    restarts.append(0.5 + 0.3*np.sin(x))
    restarts.append(0.5 + 0.3*np.cos(x))
    restarts.append(0.5 + 0.4*np.sin(2*x))
    restarts.append(0.5 + 0.3*np.sin(3*x))
    restarts.append(0.5 + 0.25*np.sin(4*x))
    
    # Triangular wave
    tri = 2.0 * np.abs(2.0 * (np.arange(n) / n - 0.5)) - 1.0
    restarts.append(0.5 + 0.4 * tri)
    
    # Sawtooth
    saw = 2.0 * (np.arange(n) / n - 0.5)
    restarts.append(0.5 + 0.3 * saw)
    
    best_x = np.full(n, 0.5)
    best_obj = _max_overlap(best_x)
    
    # Phase 1: Soft-min subgradient ascent from each restart
    for x0 in restarts:
        if _time_remaining() < 8.0:
            break
        x = _project(x0, target_sum)
        h_res, obj_res = _subgradient_step(x, n_iter=1500, lr_init=0.008, target_sum=target_sum, use_softmin=True, tau_init=0.08)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Phase 2: Annealed subgradient from best
    if _time_remaining() > 12.0:
        h_res, obj_res = _subgradient_step(best_x, n_iter=8000, lr_init=0.003, target_sum=target_sum, use_softmin=True, tau_init=0.05)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Phase 3: Coordinate descent polish
    if _time_remaining() > 8.0:
        h_res, obj_res = _coordinate_polish(best_x, n_sweeps=25, target_sum=target_sum)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Phase 4: Local refinement with epigraph
    if _time_remaining() > 5.0:
        h_res, obj_res = _local_refine(best_x, target_sum, n_refine=800)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Phase 5: Final subgradient polish with very small steps
    if _time_remaining() > 3.0:
        h_res, obj_res = _subgradient_step(best_x, n_iter=2000, lr_init=0.001, target_sum=target_sum, use_softmin=True, tau_init=0.02)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Final coordinate polish
    if _time_remaining() > 2.0:
        h_res, obj_res = _coordinate_polish(best_x, n_sweeps=10, target_sum=target_sum)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Ensure exact sum
    best_x = _project(best_x, target_sum)
    
    return best_x
# EVOLVE-BLOCK-END
