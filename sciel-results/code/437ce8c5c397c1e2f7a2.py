"""Combined program: precomputed DFT matrix, 1-flip local search, 2-flip escape, golden-section refinement."""
import numpy as np
import time
import random

_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin before 150s kill

N = 24
SQRTN = np.sqrt(N)
M = 2 ** 15  # 32768 grid points for fast evaluation

# Precompute DFT matrix: D[i, k] = exp(-2j*pi*i*k/M)
_k = np.arange(M)
_n_idx = np.arange(N)
_phases = np.outer(_k, _n_idx)
_D = np.exp(-2j * np.pi * _phases / M)

_KS = np.arange(N)

def _eval_fast(coeffs):
    """Fast evaluation using precomputed DFT matrix."""
    return _D @ coeffs

def _supnorm_fast(coeffs):
    """Fast sup-norm estimate using precomputed DFT."""
    F = _eval_fast(coeffs)
    return float(np.max(np.abs(F)))

def _eval_poly(coeffs, theta):
    """Evaluate |P(e^{i theta})| using vectorized dot product."""
    z = np.exp(1j * theta)
    return float(np.abs(np.dot(coeffs, z ** _KS)))

def _refine_peak(coeffs, ang, iters=50):
    """Golden-section refine max of |P(e^{i theta})| near ang."""
    a, b = ang - 2.0 / M, ang + 2.0 / M
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = _eval_poly(coeffs, c)
    fd = _eval_poly(coeffs, d)
    
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _eval_poly(coeffs, c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _eval_poly(coeffs, d)
    
    return max(_eval_poly(coeffs, (a + b) / 2), fc, fd)

def _supnorm_precise(coeffs, n_top=8):
    """Sup-norm with golden-section refinement of top peaks."""
    F = _eval_fast(coeffs)
    mags = np.abs(F)
    
    # Find top peaks
    idx = np.argsort(mags)[-n_top:][::-1]
    best = 0.0
    for i in idx:
        ang = 2 * np.pi * i / M
        v = _refine_peak(coeffs, ang)
        if v > best:
            best = v
    return best

def _local_search_1flip(c, F, max_val):
    """Thorough 1-flip local search until no improvement."""
    improved = True
    while improved:
        improved = False
        best_flip = -1
        best_max_val = max_val
        for n in range(N):
            F_new = F - 2.0 * c[n] * _D[:, n]
            mv = float(np.max(np.abs(F_new)))
            if mv < best_max_val - 1e-12:
                best_max_val = mv
                best_flip = n
        if best_flip != -1:
            c[best_flip] *= -1.0
            F = _eval_fast(c)
            max_val = best_max_val
            improved = True
    return c, F, max_val

def _local_search_2flip(c, F, max_val):
    """Try 2-flips to escape local minima."""
    best_flip_2 = (-1, -1)
    best_max_2 = max_val
    for n1 in range(N):
        for n2 in range(n1 + 1, N):
            F_new = F - 2.0 * (c[n1] * _D[:, n1] + c[n2] * _D[:, n2])
            mv = float(np.max(np.abs(F_new)))
            if mv < best_max_2 - 1e-12:
                best_max_2 = mv
                best_flip_2 = (n1, n2)
    if best_flip_2 != (-1, -1):
        n1, n2 = best_flip_2
        c[n1] *= -1.0
        c[n2] *= -1.0
        F = _eval_fast(c)
        max_val = best_max_2
    return c, F, max_val

def _gr_sequence(n):
    """Generate Golay-Rudin-Shapiro sequence of length n."""
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]

def construct():
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    gr = np.array(_gr_sequence(N), dtype=float)
    candidates = [gr, -gr]
    
    # Add structured seeds
    for period in [3, 4, 5, 6, 7, 8, 12]:
        candidates.append(np.array([1 if (k % period) < period // 2 else -1 for k in range(N)], dtype=float))
    
    # Add random seeds
    for _ in range(20):
        candidates.append(rng.choice([-1, 1], size=N).astype(float))
    
    best_c = None
    best_val = float('inf')
    
    # Evaluate all candidates with local search
    for c in candidates:
        if time.time() > _DEADLINE:
            break
        F = _eval_fast(c)
        max_val = float(np.max(np.abs(F)))
        
        # Do 1-flip local search
        c, F, max_val = _local_search_1flip(c.copy(), F, max_val)
        
        # Try 2-flip escape
        c2, F2, max_val2 = _local_search_2flip(c.copy(), F, max_val)
        if max_val2 < max_val:
            c, F, max_val = c2, F2, max_val2
            # Do another round of 1-flip search
            c, F, max_val = _local_search_1flip(c, F, max_val)
        
        if max_val < best_val:
            best_val = max_val
            best_c = c.copy()
    
    # Simulated annealing with 1-2 bit flips
    cur = best_c.copy()
    cur_F = _eval_fast(cur)
    cur_val = float(np.max(np.abs(cur_F)))
    T = 0.3
    
    while time.time() < _DEADLINE:
        # Flip 1 or 2 coefficients
        n_flips = rng.integers(1, 3)
        cand = cur.copy()
        for _ in range(n_flips):
            k = rng.integers(0, N)
            cand[k] *= -1
        
        # Fast screening
        cand_F = _eval_fast(cand)
        cand_val = float(np.max(np.abs(cand_F)))
        
        if cand_val < cur_val * 1.005:
            # Promising: do precise evaluation
            valr = _supnorm_precise(cand)
            if valr < cur_val:
                cur = cand
                cur_F = cand_F
                cur_val = valr
                if valr < best_val:
                    best_val = valr
                    best_c = cand.copy()
            elif random.random() < np.exp(-(valr - cur_val) / T):
                cur = cand
                cur_F = cand_F
                cur_val = valr
        else:
            # Accept worse with probability
            if random.random() < np.exp(-(cand_val - cur_val) / T):
                cur = cand
                cur_F = cand_F
                cur_val = cand_val
        
        # Cool down
        T *= 0.9998
    
    # Final precise evaluation of best
    final_val = _supnorm_precise(best_c)
    if final_val < best_val:
        best_val = final_val
    
    return [int(c) for c in best_c]
