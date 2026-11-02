import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        h = np.maximum(h, 0.0)
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    l2sq = float(np.dot(h, h))
    # linear autoconvolution
    hh = np.convolve(h, h, mode='full')
    sup = float(np.max(np.abs(hh)))
    if sup < 1e-15:
        return 0.0
    return l2sq / (l1 * sup)

def _smooth_ratio(h, beta=20.0):
    """
    Smooth approximation of the ratio using soft-max for the supremum norm.
    This is differentiable and allows gradient-based optimization.
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    l2sq = float(np.dot(h, h))
    # linear autoconvolution
    hh = np.convolve(h, h, mode='full')
    # soft-max approximation of max
    hh_clipped = np.clip(hh, -1e10, 1e10)
    m = np.max(hh_clipped)
    if m < 1e-15:
        return 0.0
    # soft-max: m + (1/beta) * log(sum(exp(beta*(x - m))))
    exp_terms = np.exp(beta * (hh_clipped - m))
    soft_max = m + (1.0 / beta) * np.log(np.sum(exp_terms))
    if soft_max < 1e-15:
        return 0.0
    return l2sq / (l1 * soft_max)

def _negative_smooth_ratio(h, beta=20.0):
    """Negative of smooth ratio for minimization."""
    return -_smooth_ratio(h, beta)

def _gradient_descent(h, max_iter=500, beta=20.0, deadline=_DEADLINE):
    """
    Gradient-based optimization using L-BFGS-B with smooth approximation.
    """
    h = h.copy()
    current_score = _ratio(h)
    
    bounds = [(0.0, None)] * N
    
    try:
        res = minimize(
            _negative_smooth_ratio,
            h,
            args=(beta,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.x is not None:
            h_new = np.maximum(res.x, 0.0)
            s_new = _ratio(h_new)
            if s_new > current_score:
                return h_new, s_new
    except Exception:
        pass
    
    return h, current_score

def _coordinate_descent(h, max_iter=200, deadline=_DEADLINE):
    """
    Cyclic coordinate descent maximizing the ratio.
    For each coordinate, try multiplicative and additive perturbations.
    """
    h = h.copy()
    current_score = _ratio(h)
    
    # Multiplicative factors to try
    mult_factors = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.7, 2.0, 3.0, 5.0, 10.0]
    
    for iteration in range(max_iter):
        if _elapsed() > deadline - 5.0:
            break
        improved = False
        for i in range(N):
            old = h[i]
            best_local = old
            best_local_score = current_score
            
            # Try multiplicative changes
            for f in mult_factors:
                candidate = old * f
                if candidate < 0:
                    candidate = 0.0
                h[i] = candidate
                s = _ratio(h)
                if s > best_local_score + 1e-12:
                    best_local_score = s
                    best_local = candidate
            
            # Try setting to zero
            if old > 0:
                h[i] = 0.0
                s = _ratio(h)
                if s > best_local_score + 1e-12:
                    best_local_score = s
                    best_local = 0.0
            
            # Try small additive perturbations (for sparse structures)
            for delta in [0.01, 0.05, 0.1, 0.2, 0.5]:
                candidate = old + delta
                h[i] = candidate
                s = _ratio(h)
                if s > best_local_score + 1e-12:
                    best_local_score = s
                    best_local = candidate
            
            h[i] = best_local
            if best_local != old:
                improved = True
        
        if not improved:
            break
    
    final_score = _ratio(h)
    return h, final_score

def _refine_hybrid(h, deadline=_DEADLINE):
    """
    Hybrid refinement: coordinate descent followed by gradient-based optimization.
    """
    h = h.copy()
    current_score = _ratio(h)
    
    # Phase 1: Coordinate descent for robust local optimization
    h_cd, score_cd = _coordinate_descent(h, max_iter=100, deadline=deadline)
    if score_cd > current_score:
        h = h_cd
        current_score = score_cd
    
    # Phase 2: Gradient-based optimization with smooth approximation
    if _elapsed() < deadline - 10.0:
        h_grad, score_grad = _gradient_descent(h, max_iter=300, beta=20.0, deadline=deadline)
        if score_grad > current_score:
            h = h_grad
            current_score = score_grad
    
    # Phase 3: Final coordinate descent polish
    if _elapsed() < deadline - 5.0:
        h_final, score_final = _coordinate_descent(h, max_iter=50, deadline=deadline)
        if score_final > current_score:
            h = h_final
            current_score = score_final
    
    return h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = None
    best_score = 0.0
    
    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12):
            h = np.maximum(h, 0.0)
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s
    
    # Generate diverse structured initializations
    structures = []
    
    # 1. Single spike (known baseline)
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    structures.append(h_spike)
    
    # 2. Two-element patterns
    for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        h2 = np.zeros(N)
        h2[0] = a
        h2[1] = 1.0 - a
        structures.append(h2)
    
    # 3. Ramp functions
    for slope in [0.5, 1.0, 2.0, 3.0]:
        h = np.linspace(0, slope, N)
        structures.append(h)
    
    # 4. Symmetric bumps of various widths
    center = N // 2
    for width in [3, 5, 8, 10, 15, 20, 25]:
        h = np.zeros(N)
        for i in range(max(0, center - width), min(N, center + width + 1)):
            h[i] = np.exp(-((i - center) ** 2) / (2 * width ** 2))
        structures.append(h)
    
    # 5. Periodic patterns
    for period in [2, 3, 4, 5, 7, 10]:
        for phase in range(min(period, 5)):
            h = np.zeros(N)
            for i in range(N):
                if (i + phase) % period == 0:
                    h[i] = 1.0
            if np.sum(h) > 0:
                structures.append(h)
    
    # 6. Random sparse patterns
    rng = np.random.default_rng(42)
    for sparsity in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for trial in range(5):
            h = np.zeros(N)
            n_nonzero = max(1, int(N * sparsity))
            idx = rng.choice(N, n_nonzero, replace=False)
            h[idx] = rng.uniform(0.1, 1.0, n_nonzero)
            structures.append(h)
    
    # 7. Power-law decay
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = (i + 1) ** (-alpha)
        structures.append(h)
    
    # 8. Two-bump structures
    for pos1 in [5, 10, 15, 20]:
        for pos2 in [30, 35, 40, 45]:
            h = np.zeros(N)
            h[pos1] = 1.0
            h[pos2] = 1.0
            structures.append(h)
    
    # 9. Cosine windows
    for freq in [1, 2, 3, 4, 5]:
        h = np.cos(np.pi * np.arange(N) * freq / N)
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-10:
            structures.append(h)
    
    # 10. Random smooth patterns
    for trial in range(10):
        h = rng.random(N)
        # Smooth it
        kernel = np.ones(5) / 5.0
        h_smooth = np.convolve(h, kernel, mode='same')
        h_smooth = np.maximum(h_smooth, 0)
        if np.sum(h_smooth) > 0:
            structures.append(h_smooth)
    
    # 11. Triangular and piecewise linear functions
    for peak_pos in [10, 20, 25, 30, 40]:
        h = np.zeros(N)
        for i in range(N):
            if i <= peak_pos:
                h[i] = i / peak_pos
            else:
                h[i] = max(0, (N - i) / (N - peak_pos))
        structures.append(h)
    
    # 12. Sigmoid-like transitions
    for center_pos in [10, 20, 30, 40]:
        for width in [5, 10, 15]:
            h = 1.0 / (1.0 + np.exp(-(np.arange(N) - center_pos) / width))
            structures.append(h)
    
    # Evaluate and refine each structure
    for h_init in structures:
        if _elapsed() > deadline - 15.0:
            break
        h_init = np.asarray(h_init, dtype=float)
        if np.sum(h_init) < 1e-15:
            continue
        
        # Quick evaluation
        s = try_candidate(h_init)
        
        # Refine with hybrid approach
        h_refined, s_refined = _refine_hybrid(h_init, deadline=deadline)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Final intensive refinement on the best found
    if best_h is not None and _elapsed() < deadline - 10.0:
        h_final, s_final = _refine_hybrid(best_h, deadline=deadline)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    # Ensure we have a valid result
    if best_h is None or np.sum(best_h) < 1e-15:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return [float(v) for v in best_h]
