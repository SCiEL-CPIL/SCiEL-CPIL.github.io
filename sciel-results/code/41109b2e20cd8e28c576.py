import numpy as np
import time

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)
# precompute for fast ratio: use a fine grid for the convolution
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# map fine grid to step cells
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def fast_ratio_from_f(f):
    """Compute ratio directly from fine-grid function f (already mapped)."""
    if np.sum(f) < 1e-15:
        return float('inf')
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def param_to_h(base, peak, width, skew):
    g = np.exp(-0.5 * ((X - skew * 0.05) / width) ** 2)
    return base + peak * g

def h_to_f(h):
    """Convert step heights to fine-grid representation."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    return h[CELL]

def construct():
    deadline = time.time() + 138.0

    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)

    # ---- Phase 1: grid search over parametric family ----
    bases = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    peaks = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    widths = [0.005, 0.008, 0.012, 0.018, 0.025, 0.035, 0.05, 0.07, 0.1]
    skews = [-1.0, 0.0, 1.0]

    for base in bases:
        for peak in peaks:
            for width in widths:
                if time.time() > deadline:
                    break
                for skew in skews:
                    h = param_to_h(base, peak, width, skew)
                    v = ratio_from_h(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()

    # ---- Phase 2: local refinement on parameters with extended search ----
    if time.time() < deadline - 30.0:
        # Denser initial scan
        cand_params = []
        for base in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]:
            for peak in [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0]:
                for width in [0.004, 0.006, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06]:
                    for skew in [-0.5, 0.0, 0.5]:
                        cand_params.append((base, peak, width, skew))
        # evaluate and keep best few
        scored = []
        for p in cand_params:
            if time.time() > deadline - 30.0:
                break
            h = param_to_h(*p)
            v = ratio_from_h(h)
            scored.append((v, p))
        scored.sort(key=lambda t: t[0])
        # Local random walk on top candidates with more iterations
        for v, p in scored[:8]:
            cur = list(p)
            cur_v = v
            for it in range(500):
                if time.time() > deadline - 25.0:
                    break
                trial = list(cur)
                trial[0] = max(0.0, cur[0] + np.random.randn() * 0.015)
                trial[1] = max(0.1, cur[1] + np.random.randn() * cur[1] * 0.1)
                trial[2] = max(0.002, cur[2] + np.random.randn() * cur[2] * 0.15)
                trial[3] = cur[3] + np.random.randn() * 0.15
                h = param_to_h(*trial)
                vv = ratio_from_h(h)
                if vv < cur_v:
                    cur, cur_v = trial, vv
                    if cur_v < best_val:
                        best_val = cur_v
                        best_h = h.copy()

    # ---- Phase 3: fast coordinate descent polish on fine-grid representation ----
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(12345)
        # Work directly on fine-grid for speed
        f_best = h_to_f(best_h)
        best_f_val = fast_ratio_from_f(f_best)
        
        # Coordinate descent: modify small blocks of the fine grid
        n_coords = FINE_N
        for it in range(15000):
            if time.time() > deadline - 5.0:
                break
            # Pick a random center and width for modification
            c = int(rng.integers(0, n_coords))
            w = int(rng.integers(5, 40))
            lo = max(0, c - w // 2)
            hi = min(n_coords, lo + w)
            
            # Try different perturbation strategies
            strategy = rng.integers(0, 3)
            f_try = f_best.copy()
            
            if strategy == 0:
                # Add a smooth bump
                amp = float(rng.normal(0.0, 0.05))
                mask = np.linspace(1.0, 0.2, hi - lo)
                f_try[lo:hi] += amp * mask
            elif strategy == 1:
                # Scale a region
                f = float(rng.uniform(0.7, 1.3))
                f_try[lo:hi] *= f
            else:
                # Subtract a smooth dip
                amp = float(rng.normal(0.0, 0.03))
                mask = np.linspace(0.2, 1.0, hi - lo)
                f_try[lo:hi] -= amp * mask
            
            f_try = np.clip(f_try, 0.0, None)
            v = fast_ratio_from_f(f_try)
            if v < best_f_val:
                best_f_val = v
                f_best = f_try
        
        # Convert back to step heights
        if best_f_val < best_val:
            # Map fine grid back to step cells (average over each cell)
            h_polished = np.zeros(N)
            for i in range(N):
                mask = (CELL == i)
                if np.any(mask):
                    h_polished[i] = np.mean(f_best[mask])
            best_val = best_f_val
            best_h = h_polished
    
    # Final: ensure nonnegative and return
    best_h = np.clip(best_h, 0.0, None)
    return list(best_h)
