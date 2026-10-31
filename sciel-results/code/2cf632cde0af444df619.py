"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b

N = 100
_START_TIME = None

def _score_sinusoidal(A, phi):
    """Compute max overlap for sinusoidal h_i = 0.5 + A sin(2*pi*i/N + phi)."""
    n = N
    i = np.arange(n, dtype=float)
    h = 0.5 + A * np.sin(2.0 * math.pi * i / n + phi)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return _score(h)

def _optimize_sinusoidal(deadline):
    """Optimize sinusoidal parameters A and phi."""
    best_score = float('inf')
    best_A = 0.5
    best_phi = 0.0
    
    # Coarse grid search for A and phi
    n_A = 40
    n_phi = 80
    A_vals = np.linspace(0.0, 0.5, n_A)
    phi_vals = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    
    for A in A_vals:
        for phi in phi_vals:
            if time.time() > deadline - 10:
                break
            s = _score_sinusoidal(A, phi)
            if s < best_score:
                best_score = s
                best_A = A
                best_phi = phi
        if time.time() > deadline - 10:
            break
    
    # Local refinement using Nelder-Mead
    if time.time() < deadline - 5:
        def obj_sin(x):
            A = x[0]
            phi = x[1]
            return _score_sinusoidal(A, phi)
        
        # Start from best grid point
        x0 = np.array([best_A, best_phi])
        try:
            res = minimize(obj_sin, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_A = res.x[0]
                best_phi = res.x[1]
        except Exception:
            pass
            
    # Construct final h
    i = np.arange(N, dtype=float)
    h = 0.5 + best_A * np.sin(2.0 * math.pi * i / N + best_phi)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    final_score = _score(h)
    
    return h, final_score


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
    grad_h = np.zeros(n)
    for i in range(n):
        grad_h[i] += grad_overlap[i] * (1.0 - (h[i+1] if i+1 < n else 0.0))
        if i + 1 < n:
            grad_h[i+1] += grad_overlap[i] * (-h[i])
    return val, grad_h


def _overlap_for_2block(k, a, b):
    """Compute max overlap for 2-block with first k elements = a, rest = b."""
    n = N
    if k <= 0 or k >= n:
        return float('inf')
    vals = []
    if k >= 2:
        vals.append(a * (1.0 - a))
    vals.append(a * (1.0 - b))
    if n - k >= 2:
        vals.append(b * (1.0 - b))
    vals.append(b)
    return max(vals)


def _optimize_2block(k, deadline):
    """Optimize 2-block construction for given k."""
    n = N
    a_min = max(0.0, (n/2.0 - n + k) / k)
    a_max = min(1.0, n / (2.0 * k))
    if a_min >= a_max:
        return None, float('inf')
    
    # Coarse grid search
    n_samples = 500
    a_vals = np.linspace(a_min + 1e-10, a_max - 1e-10, n_samples)
    
    best_score = float('inf')
    best_a = None
    
    for a_val in a_vals:
        b_val = (n/2.0 - k * a_val) / (n - k)
        if b_val < -1e-12 or b_val > 1.0 + 1e-12:
            continue
        b_val = np.clip(b_val, 0.0, 1.0)
        s = _overlap_for_2block(k, a_val, b_val)
        if s < best_score:
            best_score = s
            best_a = a_val
    
    if best_a is None:
        return None, float('inf')
    
    # Fine grid refinement around best
    step = (a_max - a_min) / n_samples
    a_ref_min = max(a_min + 1e-10, best_a - 3*step)
    a_ref_max = min(a_max - 1e-10, best_a + 3*step)
    if a_ref_min < a_ref_max:
        a_ref_vals = np.linspace(a_ref_min, a_ref_max, 1000)
        for a_val in a_ref_vals:
            b_val = (n/2.0 - k * a_val) / (n - k)
            if b_val < -1e-12 or b_val > 1.0 + 1e-12:
                continue
            b_val = np.clip(b_val, 0.0, 1.0)
            s = _overlap_for_2block(k, a_val, b_val)
            if s < best_score:
                best_score = s
                best_a = a_val
    
    # Scipy polish
    if time.time() < deadline - 3.0:
        def obj_2b(x):
            a = x[0]
            if a < 0 or a > 1:
                return 1.0
            b = (n/2.0 - k * a) / (n - k)
            if b < -1e-6 or b > 1.0 + 1e-6:
                return 1.0
            b = np.clip(b, 0.0, 1.0)
            return _overlap_for_2block(k, a, b)
        
        for frac in [0.3, 0.5, 0.7]:
            a_init = a_min + frac * (a_max - a_min)
            try:
                res = minimize(obj_2b, np.array([a_init]), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16})
                if res.fun < best_score and 0 <= res.x[0] <= 1:
                    a_opt = np.clip(res.x[0], 0, 1)
                    b_opt = (n/2.0 - k * a_opt) / (n - k)
                    b_opt = np.clip(b_opt, 0, 1)
                    s_exact = _overlap_for_2block(k, a_opt, b_opt)
                    if s_exact < best_score:
                        best_score = s_exact
                        best_a = a_opt
            except Exception:
                pass
    
    if best_a is None:
        return None, float('inf')
    
    b_opt = (n/2.0 - k * best_a) / (n - k)
    b_opt = np.clip(b_opt, 0, 1)
    h = np.zeros(n)
    h[:k] = best_a
    h[k:] = b_opt
    h = _fix_sum(h)
    actual = _score(h)
    return h, actual


def _cosine_init(k=1, phase=0.0):
    """Generate cosine wave initialization."""
    n = N
    i = np.arange(n, dtype=float)
    h = 0.5 + 0.5 * np.cos(2.0 * math.pi * k * i / n + phase)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h


def _optimize_full_lbfgsb(h_init, deadline):
    """Optimize full h vector with L-BFGS-B using smooth objective."""
    n = N
    best_score = float('inf')
    best_h = None
    beta = 200.0
    
    def obj_and_grad(x):
        h = np.clip(x, 0.0, 1.0)
        val, grad = _overlap_smooth_grad(h, beta)
        s = h.sum()
        penalty = (s - n/2.0) ** 2 * 100.0
        penalty_grad = 2.0 * (s - n/2.0) * 100.0
        grad_total = grad + penalty_grad
        return val + penalty, grad_total
    
    try:
        res = fmin_l_bfgs_b(obj_and_grad, h_init,
                             fprime=None,
                             pgtol=1e-10, maxiter=300, iprint=0)
        h = _fix_sum(res)
        actual = _score(h)
        if actual < best_score:
            best_score = actual
            best_h = h.copy()
    except Exception:
        pass
    
    return best_h, best_score


def _local_search(h_init, deadline):
    """Coordinate descent local search with adaptive step size."""
    n = N
    h = h_init.copy()
    best_score = _score(h)
    step_size = 0.01
    improved = True
    no_improve_count = 0
    
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n):
            # Try a range of deltas around current value
            for delta in [-step_size, -step_size*0.5, step_size*0.5, step_size]:
                h_trial = h.copy()
                h_trial[i] = np.clip(h_trial[i] + delta, 0.0, 1.0)
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < best_score - 1e-14:
                    best_score = s
                    h = h_trial
                    improved = True
                    no_improve_count = 0
                    break
            if time.time() > deadline - 3:
                return h, best_score
        if not improved:
            no_improve_count += 1
            step_size *= 0.5
            if step_size < 1e-7 or no_improve_count > 8:
                break
    
    # Final tiny-step polish
    step_size = 1e-5
    for _ in range(5):
        if time.time() > deadline - 1:
            break
        improved = False
        for i in range(n):
            for delta in [-step_size, step_size]:
                h_trial = h.copy()
                h_trial[i] = np.clip(h_trial[i] + delta, 0.0, 1.0)
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < best_score - 1e-15:
                    best_score = s
                    h = h_trial
                    improved = True
                    break
            if time.time() > deadline - 1:
                break
        if not improved:
            step_size *= 0.5
            if step_size < 1e-9:
                break
    
    return h, best_score


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    deadline = _START_TIME + 145.0  # safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Sinusoidal optimization (low-dimensional, fast)
    if time.time() < deadline - 30:
        h_sin, s_sin = _optimize_sinusoidal(deadline - 30)
        if h_sin is not None and s_sin < best_score:
            best_score = s_sin
            best_h = h_sin

    # Phase 2: Cosine wave initializations + full L-BFGS-B
    # Use the sinusoidal solution as a strong initialization
    if time.time() < deadline - 15:
        inits = []
        if best_h is not None:
            inits.append(best_h.copy())
        # Add some diverse inits
        inits.append(_cosine_init(1, 0.0))
        inits.append(_cosine_init(1, math.pi/2))
        inits.append(_cosine_init(2, 0.0))
        
        for h_init in inits:
            if time.time() > deadline - 15:
                break
            h_opt, s_opt = _optimize_full_lbfgsb(h_init, deadline - 15)
            if h_opt is not None and s_opt < best_score:
                best_score = s_opt
                best_h = h_opt
    
    # Phase 3: Local search polish
    if time.time() < deadline - 3 and best_h is not None:
        hp, sp = _local_search(best_h, deadline - 1)
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
