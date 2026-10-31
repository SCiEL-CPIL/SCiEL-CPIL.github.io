"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Fix sum of h to be exactly N/2, keeping elements in [0,1]."""
    if target is None:
        target = N / 2.0
    h = h.copy()
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        # Second pass for elements not at boundary
        s = h.sum()
        diff2 = target - s
        if abs(diff2) > 1e-15:
            mask = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                # Final fix
                s = h.sum()
                diff3 = target - s
                if abs(diff3) > 1e-15:
                    # Spread across all non-boundary elements
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


def _overlap_smooth(h, beta=100.0):
    """Smooth surrogate for max overlap."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return _smooth_max(overlap, beta)


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


def _optimize_smooth_2block(deadline):
    """Optimize 2-block construction using smooth objective with SLSQP."""
    n = N
    best_score = float('inf')
    best_h = None
    
    for k in range(1, n):
        a_min = max(0.0, (n/2.0 - n + k) / k)
        a_max = min(1.0, n / (2.0 * k))
        if a_min >= a_max:
            continue
        
        # Use smooth objective
        def obj_smooth(x, k=k):
            a, b = x
            if a < 0 or a > 1 or b < 0 or b > 1:
                return 1.0
            # Check sum constraint
            s = k * a + (n - k) * b
            if abs(s - n/2.0) > 0.1:
                return 1.0 + abs(s - n/2.0)
            h = np.zeros(n)
            h[:k] = a
            h[k:] = b
            h = _fix_sum(h)
            return _overlap_smooth(h, beta=200.0)
        
        # Try multiple starting points
        for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            a_init = a_min + frac * (a_max - a_min)
            b_init = (n/2.0 - k * a_init) / (n - k)
            if b_init < 0 or b_init > 1:
                continue
            try:
                res = minimize(obj_smooth, np.array([a_init, b_init]),
                               method='SLSQP',
                               bounds=[(0, 1), (0, 1)],
                               options={'maxiter': 500, 'ftol': 1e-12})
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
        
        if time.time() > deadline - 40:
            break
    
    return best_h, best_score


def _optimize_smooth_multiblock(deadline):
    """Optimize multi-block construction using smooth objective."""
    n = N
    best_score = float('inf')
    best_h = None
    
    # Try 3-block and 4-block with smooth optimization
    for n_blocks in [3, 4]:
        if time.time() > deadline - 20:
            break
        
        # Parameterize: n_blocks-1 boundary positions + n_blocks heights
        # Use differential evolution for global search, then polish with SLSQP
        
        def de_obj(params, n_blocks=n_blocks):
            # params: [b1, b2, ..., b_{n-1}, h1, h2, ..., h_{n}]
            # boundaries are in [0, n]
            boundaries = np.zeros(n_blocks + 1)
            boundaries[0] = 0
            boundaries[-1] = n
            for i in range(n_blocks - 1):
                boundaries[i+1] = int(round(params[i]))
            # Ensure sorted
            for i in range(1, n_blocks):
                if boundaries[i+1] <= boundaries[i]:
                    boundaries[i+1] = boundaries[i] + 1
            heights = params[n_blocks-1:]
            
            # Check sum
            total = 0
            for i in range(n_blocks):
                length = boundaries[i+1] - boundaries[i]
                total += length * heights[i]
            penalty = max(0, abs(total - n/2.0) - 0.05) * 5
            
            h = np.zeros(n)
            for i in range(n_blocks):
                start = int(boundaries[i])
                end = int(boundaries[i+1])
                if end > start:
                    h[start:end] = heights[i]
            h = _fix_sum(h)
            return _overlap_smooth(h, beta=100.0) + penalty
        
        de_bounds = [(0, n)] * (n_blocks - 1) + [(0, 1)] * n_blocks
        
        try:
            result = differential_evolution(
                de_obj, de_bounds,
                seed=42, maxiter=100, tol=1e-10,
                popsize=15, mutation=(0.3, 1.0), recombination=0.8,
                polish=True, init='sobol'
            )
            
            if result.fun < best_score + 0.1:
                # Extract and polish
                boundaries = np.zeros(n_blocks + 1)
                boundaries[0] = 0
                boundaries[-1] = n
                for i in range(n_blocks - 1):
                    boundaries[i+1] = int(round(result.x[i]))
                for i in range(1, n_blocks):
                    if boundaries[i+1] <= boundaries[i]:
                        boundaries[i+1] = boundaries[i] + 1
                heights = result.x[n_blocks-1:]
                
                # Polish with SLSQP
                def polish_obj(x, n_blocks=n_blocks, boundaries=boundaries):
                    h = np.zeros(n)
                    for i in range(n_blocks):
                        start = int(boundaries[i])
                        end = int(boundaries[i+1])
                        if end > start:
                            h[start:end] = x[i]
                    h = _fix_sum(h)
                    return _overlap_smooth(h, beta=500.0)
                
                res2 = minimize(polish_obj, heights,
                               method='SLSQP',
                               bounds=[(0, 1)] * n_blocks,
                               options={'maxiter': 1000, 'ftol': 1e-14})
                
                if res2.fun < 1.0:
                    h = np.zeros(n)
                    for i in range(n_blocks):
                        start = int(boundaries[i])
                        end = int(boundaries[i+1])
                        if end > start:
                            h[start:end] = res2.x[i]
                    h = _fix_sum(h)
                    actual = _score(h)
                    if actual < best_score:
                        best_score = actual
                        best_h = h.copy()
        except Exception:
            pass
    
    return best_h, best_score


def _optimize_direct(deadline):
    """Direct optimization of h with smooth objective."""
    n = N
    best_score = float('inf')
    best_h = None
    
    # Start from a good initial guess
    # Use the theoretical 2-block solution as starting point
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h_init = np.zeros(n)
    h_init[:n//2] = A
    h_init[n//2:] = B
    h_init = _fix_sum(h_init)
    
    # Optimize directly with SLSQP on all N variables
    # This is expensive but might find better solutions
    
    def obj_direct(x):
        h = np.clip(x, 0.0, 1.0)
        # Enforce sum constraint via penalty
        s = h.sum()
        penalty = (s - n/2.0) ** 2 * 10
        return _overlap_smooth(h, beta=200.0) + penalty
    
    try:
        res = minimize(obj_direct, h_init,
                      method='SLSQP',
                      bounds=[(0, 1)] * n,
                      options={'maxiter': 50, 'ftol': 1e-10})
        if res.fun < 1.0:
            h = _fix_sum(res.x)
            actual = _score(h)
            if actual < best_score:
                best_score = actual
                best_h = h.copy()
    except Exception:
        pass
    
    return best_h, best_score


def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Fast 2-block optimization with smooth objective
    if time.time() < deadline - 60:
        h2, s2 = _optimize_smooth_2block(deadline - 60)
        if h2 is not None and s2 < best_score:
            best_score = s2
            best_h = h2
    
    # Phase 2: Multi-block optimization
    if time.time() < deadline - 30:
        hm, sm = _optimize_smooth_multiblock(deadline - 30)
        if hm is not None and sm < best_score:
            best_score = sm
            best_h = hm
    
    # Phase 3: Direct optimization from best found
    if time.time() < deadline - 10 and best_h is not None:
        hd, sd = _optimize_direct(deadline - 10)
        if hd is not None and sd < best_score:
            best_score = sd
            best_h = hd
    
    # Fallback
    if best_h is None:
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = _fix_sum(h)
    
    return best_h
