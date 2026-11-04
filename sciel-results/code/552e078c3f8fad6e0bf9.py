"""
Search for nonnegative step heights maximizing (L2 norm)^2 / (L1 norm * sup autoconvolution).

Strategy:
- The ratio R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) is scale-invariant.
- We use a multi-start local search strategy.
- Initial candidates are generated from:
  1. "Flat Spectrum" vectors: h = |IDFT(Z)|^2 where Z has random phases. This tends to produce low autocorrelation peaks.
  2. Random sparse vectors.
  3. Greedy Sidon sets.
- Optimization uses Nelder-Mead on the parameterization h_i = x_i^2 to ensure non-negativity and handle the non-smooth max operator robustly.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_dx = (0.25 - (-0.25)) / N  # 0.5/50 = 0.01
_DEADLINE = 115.0  # safety margin before 120s kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _elapsed()


def _autoconv_peak(h):
    """Compute supremum norm of autoconvolution using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    return np.max(conv)


def _ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15) or np.sum(h) < 1e-15:
        return 0.0
    h = np.maximum(h, 0)
    l2sq = np.sum(h * h)
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    if l1 < 1e-15 or peak < 1e-30:
        return 0.0
    return l2sq / (l1 * peak)


def _objective_x(x):
    """
    Objective to minimize: 1 / Ratio.
    x is a real vector of length N.
    h = x^2.
    """
    h = x * x
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e15
    h_norm = h / l1
    
    l2sq = np.sum(h_norm * h_norm)
    
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h_norm, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    peak = np.max(conv)
    
    if peak < 1e-30:
        return 1e15

    return 1.0 / (l2sq / peak) # Minimize 1/R


def _generate_flat_spectrum_candidates(count=20):
    """Generate candidates from flat spectrum complex vectors."""
    candidates = []
    for _ in range(count):
        # Random phases
        phases = np.random.uniform(0, 2 * np.pi, N)
        Z = np.exp(1j * phases)
        # Inverse DFT
        h = np.abs(np.fft.ifft(Z)) ** 2
        # Normalize
        l1 = np.sum(h)
        if l1 > 0:
            h = h / l1
        candidates.append(h)
    return candidates


def _generate_sparse_candidates(count=20, sparsity_range=(5, 20)):
    """Generate random sparse candidates."""
    candidates = []
    for _ in range(count):
        k = np.random.randint(sparsity_range[0], sparsity_range[1] + 1)
        indices = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[indices] = 1.0
        l1 = np.sum(h)
        if l1 > 0:
            h = h / l1
        candidates.append(h)
    return candidates


def _generate_sidon_candidates(count=10):
    """Generate Sidon set candidates."""
    candidates = []
    for _ in range(count):
        S = set()
        # Greedy Sidon with randomization
        for i in range(N):
            if np.random.rand() > 0.7: # Skip some to allow randomization
                continue
            valid = True
            # Check sums
            sums = set()
            for j in S:
                s = i + j
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            if valid:
                S.add(i)
        
        if len(S) >= 4:
            h = np.zeros(N)
            for idx in S:
                h[idx] = 1.0
            l1 = np.sum(h)
            if l1 > 0:
                h = h / l1
                candidates.append(h)
    return candidates


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_val = -np.inf

    def consider(h):
        nonlocal best_h, best_val
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)
        v = _ratio(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
        return v

    # 1. Generate initial candidates
    candidates = []
    
    # Flat spectrum
    candidates.extend(_generate_flat_spectrum_candidates(30))
    
    # Sparse
    candidates.extend(_generate_sparse_candidates(30))
    
    # Sidon
    candidates.extend(_generate_sidon_candidates(20))
    
    # Uniform (baseline)
    candidates.append(np.ones(N) / N)
    
    # Single spike
    s = np.zeros(N); s[N//2] = 1.0; candidates.append(s)

    # Evaluate all candidates
    for c in candidates:
        consider(c)

    # 2. Local optimization using Nelder-Mead on h = x^2
    # We will optimize the top K candidates
    # Sort candidates by initial ratio
    scored_candidates = []
    for c in candidates:
        v = _ratio(c)
        scored_candidates.append((v, c))
    scored_candidates.sort(key=lambda x: -x[0])
    
    top_k = 15
    # Ensure we have at least 1 candidate
    if not scored_candidates:
        return [1.0/N] * N
        
    # Optimization budget
    # Nelder-Mead can be slow. We will run it on the best few.
    # We need to map h -> x. x = sqrt(h).
    # We will run minimize on x.
    
    elapsed = time.time() - _start
    time_per_opt = max(0.5, (_DEADLINE - 10.0 - elapsed) / top_k)
    
    for i in range(top_k):
        if time.time() > _DEADLINE - 10.0:
            break
            
        v_init, c_init = scored_candidates[i]
        
        # Initial x
        x0 = np.sqrt(c_init)
        
        # Add small noise to x0 to escape local minima if needed? 
        # Nelder-Mead is a local search.
        
        try:
            # Use Nelder-Mead
            # x can be negative, but h=x^2 is symmetric. 
            # To avoid converging to x=0 (which gives h=0, invalid), 
            # we rely on the objective returning large values for small L1.
            # However, if x becomes all zeros, we fail.
            # We can add a small penalty or just restart if it fails.
            
            res = minimize(
                _objective_x, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            x_opt = res.x
            h_opt = x_opt * x_opt
            h_opt = np.maximum(h_opt, 0)
            
            # Normalize h_opt to ensure it's not degenerate
            l1 = np.sum(h_opt)
            if l1 > 1e-15:
                h_opt = h_opt / l1
                consider(h_opt)
            else:
                # Fallback to initial
                consider(c_init)
                
        except Exception:
            consider(c_init)

    # 3. Final refinement: Coordinate descent on the best h
    if best_h is not None and time.time() < _DEADLINE - 5.0:
        h = best_h.copy()
        step = 1e-3
        # Simple hill climbing
        for it in range(1000):
            if time.time() > _DEADLINE - 5.0:
                break
            improved = False
            for i in range(N):
                old = h[i]
                for delta in (step, -step, 2 * step, 0.5 * step):
                    cand = h.copy()
                    cand[i] = max(0.0, old + delta)
                    if _ratio(cand) > best_val:
                        h = cand
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-7:
                    break
        consider(h)

    if best_h is None:
        return [1.0/N] * N

    # Return as a plain list of floats
    return [float(x) for x in best_h]
