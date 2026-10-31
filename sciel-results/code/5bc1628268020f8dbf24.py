"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution, fmin_l_bfgs_b

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _overlap_vector(h):
    """Compute the overlap vector h[i]*(1-h[i+1])."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    return h * (1.0 - shifted)


def _fix_sum(h, target=None):
    """Fix sum of h to be exactly N/2, keeping elements in [0,1]."""
    if target is None:
        target = N / 2.0
    h = h.copy()
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        s = h.sum()
        diff2 = target - s
        if abs(diff2) > 1e-15:
            mask = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                s = h.sum()
                diff3 = target - s
                if abs(diff3) > 1e-15:
                    mask2 = (h > 1e-12) & (h < 1.0 - 1e-12)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max(x, beta=100.0):
    """Soft max using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0
    shifted = x - max_x
    return max_x + math.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_max_grad(x, beta=100.0):
    """Soft max and its gradient with respect to x."""
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0, np.zeros_like(x)
    shifted = x - max_x
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    value = max_x + math.log(wsum) / beta
    grad = weights / wsum
    return value, grad


def _overlap_smooth(h, beta=100.0):
    """Smooth surrogate for max overlap."""
    overlap = _overlap_vector(h)
    return _smooth_max(overlap, beta)


def _overlap_smooth_grad(h, beta=100.0):
    """Smooth surrogate for max overlap and its gradient."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    val, grad_overlap = _smooth_max(overlap, beta)
    # d(overlap[i])/d(h[i]) = 1 - shifted[i] = 1 - h[i+1]
    # d(overlap[i])/d(h[i+1]) = -h[i]  (from shifted[i] = h[i+1])
    grad_h = np.zeros(n)
    for i in range(n):
        grad_h[i] += grad_overlap[i] * (1.0 - (h[i+1] if i+1 < n else 0.0))
        if i + 1 < n:
            grad_h[i+1] += grad_overlap[i] * (-h[i])
    return val, grad_h


def _build_from_blocks(boundaries, heights):
    """Build h from block boundaries and heights, then fix sum."""
    n = N
    h = np.zeros(n)
    for i in range(len(boundaries)):
        start = int(round(boundaries[i]))
        end = int(round(boundaries[i + 1])) if i + 1 < len(boundaries) else n
        start = max(0, min(start, n))
        end = max(start, min(end, n))
        h[start:end] = heights[i]
    return _fix_sum(h)


def _sinusoidal_h(k, a, b, phase=0.0):
    """Build h from sinusoidal modulation: h[i] = a + b*cos(2*pi*k*i/N + phase)."""
    n = N
    i = np.arange(n, dtype=float)
    h = a + b * np.cos(2.0 * math.pi * k * i / n + phase)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h


def _optimize_sinusoidal(deadline):
    """Optimize sinusoidal ansatz: h[i] = a + b*cos(2*pi*k*i/N + phase)."""
    n = N
    best_score = float('inf')
    best_h = None
    
    for k in range(1, n // 2 + 1):
        if time.time() > deadline - 5:
            break
        for phase in [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi]:
            if time.time() > deadline - 3:
                break
            # For given k, phase: optimize a, b
            def obj_sin(x, k=k, phase=phase):
                a, b = x
                if a < 0 or a > 1 or b < 0 or b > 1:
                    return 1.0
                h = _sinusoidal_h(k, a, b, phase)
                return _overlap_smooth(h, beta=200.0) + 10.0 * (h.sum() - n/2.0)**2
            
            # Multiple starts
            for a0 in [0.3, 0.5, 0.7]:
                for b0 in [0.1, 0.3, 0.5]:
                    try:
                        res = minimize(obj_sin, np.array([a0, b0]),
                                       method='Nelder-Mead',
                                       bounds=[(0, 1), (0, 1)],
                                       options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                        if res.fun < 1.0:
                            h = _sinusoidal_h(k, res.x[0], res.x[1], phase)
                            actual = _score(h)
                            if actual < best_score:
                                best_score = actual
                                best_h = h.copy()
                    except Exception:
                        pass
    return best_h, best_score


def _optimize_smooth_2block(deadline):
    """Optimize 2-block construction using smooth objective with SLSQP."""
    n = N
    best_score = float('inf')
    best_h = None
    
    for k in range(1, n):
        if time.time() > deadline - 20:
            break
        a_min = max(0.0, (n/2.0 - n + k) / k)
        a_max = min(1.0, n / (2.0 * k))
        if a_min >= a_max:
            continue
        
        def obj_smooth(x, k=k):
            a, b = x
            if a < 0 or a > 1 or b < 0 or b > 1:
                return 1.0
            s = k * a + (n - k) * b
            if abs(s - n/2.0) > 0.1:
                return 1.0 + abs(s - n/2.0)
            h = np.zeros(n)
            h[:k] = a
            h[k:] = b
            h = _fix_sum(h)
            return _overlap_smooth(h, beta=200.0)
        
        for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            a_init = a_min + frac * (a_max - a_min)
            b_init = (n/2.0 - k * a_init) / (n - k)
            if b_init < 0 or b_init > 1:
                continue
            try:
                res = minimize(obj_smooth, np.array([a_init, b_init]),
                               method='SLSQP',
                               bounds=[(0, 1), (0, 1)],
                               options={'maxiter': 300, 'ftol': 1e-12})
                if res.fun < 1.0:
                    a_opt, b_opt = res.x
                    h = np.zeros(n)
                    h[:k] = a_opt
                    h[k:] = b_opt
                    h = _fix_sum(h)
                    actual = _score(h)
                    if actual < best_score:
                        best_score = actual
                        best_h = h.copy()
            except Exception:
                pass
    
    return best_h, best_score


def _optimize_full_lbfgsb(h_init, deadline):
    """Optimize full h vector with L-BFGS-B using smooth objective."""
    n = N
    best_score = float('inf')
    best_h = None
    beta = 300.0
    
    def obj_and_grad(x):
        h = np.clip(x, 0.0, 1.0)
        val, grad = _overlap_smooth_grad(h, beta)
        # Sum penalty
        s = h.sum()
        penalty = (s - n/2.0) ** 2 * 50.0
        penalty_grad = 2.0 * (s - n/2.0) * 50.0
        grad_total = grad + penalty_grad
        return val + penalty, grad_total
    
    try:
        res = fmin_l_bfgs_b(obj_and_grad, h_init,
                             fprime=None,  # let it compute numerically if needed
                             pgtol=1e-10, maxiter=200, iprint=0)
        h = _fix_sum(res)
        actual = _score(h)
        if actual < best_score:
            best_score = actual
            best_h = h.copy()
    except Exception:
        pass
    
    return best_h, best_score


def _local_search(h_init, deadline):
    """Coordinate descent local search on the full vector."""
    n = N
    h = h_init.copy()
    best_score = _score(h)
    step_size = 0.05
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n):
            # Try small perturbations
            for delta in [-step_size, -step_size/2, step_size/2, step_size]:
                h_trial = h.copy()
                h_trial[i] = np.clip(h_trial[i] + delta, 0.0, 1.0)
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < best_score - 1e-12:
                    best_score = s
                    h = h_trial
                    improved = True
                    break
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    return h, best_score


def _optimize_direct(deadline):
    """Direct optimization of h with smooth objective using L-BFGS-B."""
    n = N
    best_score = float('inf')
    best_h = None
    
    # Multiple initializations
    inits = []
    
    # 2-block theoretical
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h1 = np.zeros(n)
    h1[:n//2] = A
    h1[n//2:] = B
    inits.append(_fix_sum(h1))
    
    # Sinusoidal k=1
    h2 = _sinusoidal_h(1, 0.5, 0.3, 0.0)
    inits.append(h2)
    
    # Sinusoidal k=2
    h3 = _sinusoidal_h(2, 0.5, 0.2, 0.0)
    inits.append(h3)
    
    # Random
    np.random.seed(42)
    h4 = np.random.uniform(0.3, 0.7, n)
    inits.append(_fix_sum(h4))
    
    for h_init in inits:
        if time.time() > deadline - 10:
            break
        h_opt, s_opt = _optimize_full_lbfgsb(h_init, deadline)
        if h_opt is not None and s_opt < best_score:
            best_score = s_opt
            best_h = h_opt
    
    return best_h, best_score


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Fast 2-block optimization
    if time.time() < deadline - 80:
        h2, s2 = _optimize_smooth_2block(deadline - 80)
        if h2 is not None and s2 < best_score:
            best_score = s2
            best_h = h2
    
    # Phase 2: Sinusoidal ansatz optimization
    if time.time() < deadline - 50:
        hs, ss = _optimize_sinusoidal(deadline - 50)
        if hs is not None and ss < best_score:
            best_score = ss
            best_h = hs
    
    # Phase 3: Full vector L-BFGS-B optimization from multiple starts
    if time.time() < deadline - 30:
        hd, sd = _optimize_direct(deadline - 30)
        if hd is not None and sd < best_score:
            best_score = sd
            best_h = hd
    
    # Phase 4: Local search polish on best found
    if time.time() < deadline - 5 and best_h is not None:
        hp, sp = _local_search(best_h, deadline - 2)
        if sp < best_score:
            best_score = sp
            best_h = hp
    
    # Fallback
    if best_h is None:
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = _fix_sum(h)
    
    return best_h
