"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 138.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _all_corrs_fft(h):
    """Compute all non-circular cross-correlations R(k) for k=1..n-1 using FFT.
    R(k) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    """
    n = len(h)
    # Use FFT to compute autocorrelation
    # Pad to avoid circular wrap
    size = 1
    while size < 2 * n:
        size *= 2
    H = np.fft.rfft(h, size)
    # Autocorrelation via inverse FFT of |H|^2
    corr_full = np.fft.irfft(H * np.conj(H), size)
    # corr_full[0] = sum h[i]^2, corr_full[k] = sum_{i=0}^{n-k-1} h[i]*h[i+k]
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
    for _ in range(20):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-13:
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

def _subgradient_ascent(h, n_iter, lr_init, target_sum):
    """Subgradient ascent to maximize min_k R(k), with projection."""
    n = len(h)
    h = h.copy()
    lr = lr_init
    best_obj = _max_overlap(h)
    best_h = h.copy()
    patience = 0
    for it in range(n_iter):
        if _time_remaining() < 2.0:
            break
        corrs = _all_corrs_fft(h)
        min_idx = np.argmin(corrs)  # 0-indexed, corresponds to k=min_idx+1
        # Collect active shifts (within tolerance of min)
        tol = 0.02
        active = np.where(corrs < corrs[min_idx] + tol)[0]
        # Gradient of min correlation (subgradient: average over active)
        grad = np.zeros(n)
        for idx in active:
            k = idx + 1
            grad[:n-k] += (2.0/n) * h[k:]
            grad[k:] += (2.0/n) * h[:n-k]
        grad /= len(active)
        # Ascent step
        h_new = h + lr * grad
        h_new = _project(h_new, target_sum)
        obj = _max_overlap(h_new)
        if obj < best_obj - 1e-10:
            best_obj = obj
            best_h = h_new.copy()
            patience = 0
        else:
            patience += 1
            if patience > 50:
                lr *= 0.9
                patience = 0
        h = h_new
    return best_h, best_obj

def _coordinate_descent(h, n_sweeps, target_sum):
    """Coordinate descent: for each coordinate, find optimal value given others.
    This is a structural change: direct per-coordinate optimization of the max-min objective.
    """
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
            # Try a few candidate values for h[j]
            old_val = h[j]
            # Candidate: 0, 1, old_val, and a few interpolated
            candidates = [0.0, 1.0, old_val]
            if 0 < old_val < 1:
                candidates.extend([old_val * 0.5, old_val * 1.5, 0.25, 0.75])
            best_cand_obj = best_obj
            best_cand_val = old_val
            for cv in candidates:
                # Adjust to maintain sum: distribute the difference to other coords
                diff = old_val - cv
                h[j] = cv
                # Quick evaluation
                obj = _max_overlap(h)
                # Project to fix sum
                h_proj = _project(h, target_sum)
                obj_proj = _max_overlap(h_proj)
                if obj_proj < best_cand_obj - 1e-12:
                    best_cand_obj = obj_proj
                    best_cand_val = cv
                    h = h_proj.copy()
                h[j] = old_val  # restore
            if best_cand_val != old_val:
                h[j] = best_cand_val
                h = _project(h, target_sum)
                best_obj = _max_overlap(h)
                improved = True
        if not improved and sweep > 2:
            break
    return h, best_obj

def _anneal_subgradient(h, target_sum, total_iter=8000):
    """Subgradient ascent with simulated-annealing-like learning rate schedule."""
    n = len(h)
    h = h.copy()
    best_obj = _max_overlap(h)
    best_h = h.copy()
    # Learning rate schedule: start moderate, decay
    lr_schedule = []
    for it in range(total_iter):
        frac = it / max(total_iter - 1, 1)
        # Cosine decay from 0.01 to 0.0001
        lr = 0.0001 + 0.5 * (0.01 - 0.0001) * (1 + np.cos(np.pi * frac))
        lr_schedule.append(lr)
    
    for it in range(total_iter):
        if _time_remaining() < 2.0:
            break
        lr = lr_schedule[it]
        corrs = _all_corrs_fft(h)
        min_idx = np.argmin(corrs)
        # Active set with adaptive tolerance
        tol = max(0.01, 0.05 * (1.0 - it / total_iter))
        active = np.where(corrs < corrs[min_idx] + tol)[0]
        grad = np.zeros(n)
        for idx in active:
            k = idx + 1
            grad[:n-k] += (2.0/n) * h[k:]
            grad[k:] += (2.0/n) * h[:n-k]
        grad /= len(active)
        # Normalize gradient to control step size
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-10:
            grad = grad / gnorm
        h_new = h + lr * grad
        h_new = _project(h_new, target_sum)
        obj = _max_overlap(h_new)
        if obj < best_obj - 1e-12:
            best_obj = obj
            best_h = h_new.copy()
        h = h_new
    return best_h, best_obj

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
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
            restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # Random
    for _ in range(15):
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
    # Triangular wave
    tri = 2.0 * np.abs(2.0 * (np.arange(n) / n - 0.5)) - 1.0
    restarts.append(0.5 + 0.4 * tri)
    
    best_x = np.full(n, 0.5)
    best_obj = _max_overlap(best_x)
    
    # Phase 1: Subgradient ascent from each restart
    for x0 in restarts:
        if _time_remaining() < 10.0:
            break
        x = _project(x0, target_sum)
        # Quick subgradient run
        h_res, obj_res = _subgradient_ascent(x, n_iter=2000, lr_init=0.005, target_sum=target_sum)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Phase 2: Annealed subgradient from best
    if _time_remaining() > 15.0:
        h_res, obj_res = _anneal_subgradient(best_x, target_sum, total_iter=10000)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Phase 3: Coordinate descent polish
    if _time_remaining() > 10.0:
        h_res, obj_res = _coordinate_descent(best_x, n_sweeps=30, target_sum=target_sum)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Phase 4: Final subgradient polish with very small steps
    if _time_remaining() > 5.0:
        h_res, obj_res = _subgradient_ascent(best_x, n_iter=3000, lr_init=0.001, target_sum=target_sum)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Final coordinate polish
    if _time_remaining() > 3.0:
        h_res, obj_res = _coordinate_descent(best_x, n_sweeps=10, target_sum=target_sum)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Ensure exact sum
    best_x = _project(best_x, target_sum)
    
    return best_x
# EVOLVE-BLOCK-END
