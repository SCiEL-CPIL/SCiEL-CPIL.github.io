"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, linprog

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _max_overlap(h):
    """Compute maximum overlap of h with shifted (1-h)."""
    n = len(h)
    overlaps = []
    for k in range(1, n):
        shifted = np.roll(h, k)
        overlap = np.sum(h * (1.0 - shifted))
        overlaps.append(overlap)
    return max(overlaps)

def _soft_min_obj_and_grad(h, beta):
    """
    Soft-min of circular autocorrelation at lags 1..N-1.
    Objective: minimize -soft_min(autocorr) = maximize soft_min(autocorr)
    """
    n = len(h)
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    
    a = autocorr[1:]  # lags 1 to N-1
    min_val = np.min(a)
    shifted = a - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    if sum_exp < 1e-300:
        soft_min = min_val
        weights = np.full(len(a), 1.0/len(a))
    else:
        soft_min = min_val - np.log(sum_exp) / beta
        weights = exp_terms / sum_exp
    
    obj = -soft_min
    
    # Gradient: d(soft_min)/d(h_i) = sum_k weights[k-1] * (h[(i+k)%n] + h[(i-k)%n])
    grad_h = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1]
        if w < 1e-15:
            continue
        grad_h += w * (np.roll(h, -k) + np.roll(h, k))
    
    grad_h = -grad_h
    return obj, grad_h

def _project_to_sum(h, target_sum):
    """Project h to sum = target_sum with bounds [0,1]."""
    h = np.clip(h, 0, 1)
    for _ in range(50):
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-13:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.sum(mask) == 0:
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT - 15.0  # Safety margin
    
    target_sum = N / 2.0
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.full(N, 0.5))
    
    # 2. Alternating (period 2)
    h_alt = np.zeros(N)
    h_alt[0::2] = 1.0
    candidates.append(h_alt)
    
    # 3. Periodic patterns with various periods
    for period in [3, 4, 5, 7, 10, 11, 13, 17, 20, 25]:
        if period > N:
            continue
        # Try different phase patterns
        for num_high in range(1, period):
            pattern = np.zeros(period)
            pattern[:num_high] = 1.0
            # Tile to length N
            h_pat = np.tile(pattern, N // period + 1)[:N]
            # Adjust to satisfy sum constraint
            h_pat = _project_to_sum(h_pat, target_sum)
            candidates.append(h_pat)
    
    # 4. Sinusoidal patterns
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 7, 10, 13]:
        for amp in [0.3, 0.4, 0.45, 0.5]:
            for phase in [0, np.pi/4, np.pi/2]:
                h_sin = 0.5 + amp * np.cos(freq * x + phase)
                h_sin = _project_to_sum(h_sin, target_sum)
                candidates.append(h_sin)
    
    # 5. Random candidates
    np.random.seed(42)
    for _ in range(30):
        h_r = np.random.uniform(0, 1, N)
        h_r = _project_to_sum(h_r, target_sum)
        candidates.append(h_r)
    
    # 6. Structured: blocks of high/low values
    for block_len in [2, 3, 4, 5, 10]:
        for high_val in [0.7, 0.8, 0.9]:
            pattern = np.tile([high_val, (2.0 - high_val) * (block_len - 1) / block_len], N // block_len + 1)[:N]
            candidates.append(_project_to_sum(pattern, target_sum))
    
    # Evaluate all candidates
    for h in candidates:
        h = np.clip(h, 0, 1)
        h = _project_to_sum(h, target_sum)
        s = _max_overlap(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Multi-stage optimization with increasing beta
    if _time_remaining() > 30.0:
        for beta in [50.0, 100.0, 200.0, 400.0]:
            if _time_remaining() < 10.0:
                break
            
            # Try optimizing from best and a few good candidates
            init_points = [best_h]
            # Add a few other good candidates
            for h in candidates[:10]:
                if h is not best_h:
                    init_points.append(h)
            
            for h_init in init_points[:5]:
                if _time_remaining() < 5.0:
                    break
                
                try:
                    res = minimize(
                        _soft_min_obj_and_grad,
                        h_init.copy(),
                        args=(beta,),
                        method='L-BFGS-B',
                        jac=True,
                        bounds=[(0, 1)] * N,
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                    )
                    h_new = np.clip(res.x, 0, 1)
                    h_new = _project_to_sum(h_new, target_sum)
                    s_new = _max_overlap(h_new)
                    if s_new < best_score:
                        best_score = s_new
                        best_h = h_new.copy()
                except Exception:
                    pass
    
    # Final local refinement: coordinate descent
    if _time_remaining() > 15.0:
        h_cd = best_h.copy()
        improved = True
        max_iter = 200
        for iteration in range(max_iter):
            if _time_remaining() < 5.0:
                break
            if not improved:
                break
            improved = False
            for i in range(N):
                old_val = h_cd[i]
                # Try adjusting h_cd[i] while keeping sum constant
                # We'll adjust h_cd[i] and distribute the difference to other elements
                # Simpler: try small perturbations
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    new_val = old_val + delta
                    if new_val < 0 or new_val > 1:
                        continue
                    h_test = h_cd.copy()
                    h_test[i] = new_val
                    h_test = _project_to_sum(h_test, target_sum)
                    s_test = _max_overlap(h_test)
                    if s_test < best_score - 1e-12:
                        best_score = s_test
                        best_h = h_test.copy()
                        h_cd = h_test.copy()
                        improved = True
                        break
    
    # Ensure final validity
    h_final = np.clip(best_h, 0, 1)
    h_final = _project_to_sum(h_final, target_sum)
    
    # Fix any remaining floating point issues
    diff = target_sum - np.sum(h_final)
    if abs(diff) > 1e-10:
        for i in range(N):
            new_val = h_final[i] + diff
            if 0 <= new_val <= 1:
                h_final[i] = new_val
                break
    
    return h_final
# EVOLVE-BLOCK-END

