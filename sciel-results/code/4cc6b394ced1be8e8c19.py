import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for accurate convolution
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
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
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    return float(np.max(conv) / integral_sq)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = None
    best_val = float('inf')
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
            
    # Candidate 1: Uniform Function (Theoretical Minimizer)
    h_uniform = np.ones(N)
    val_uniform = ratio_from_h(h_uniform)
    update_best(h_uniform, val_uniform)
    
    # Candidate 2: Triangle Function
    h_triangle = np.maximum(0, 1 - 4 * np.abs(X))
    val_triangle = ratio_from_h(h_triangle)
    update_best(h_triangle, val_triangle)
    
    # Candidate 3: Trapezoids (Interpolating between Uniform and Triangle)
    # h(x) = 1 for |x| < b, linearly to 0 at 0.25
    # Support is always [-0.25, 0.25]
    for b in [0.0, 0.05, 0.1, 0.15, 0.2, 0.24]:
        if b >= 0.25:
            continue
        # Linear slope from (b, 1) to (0.25, 0)
        # y - 0 = m (x - 0.25) => m = -1 / (0.25 - b)
        # h(x) = (0.25 - |x|) / (0.25 - b) for b < |x| < 0.25
        # h(x) = 1 for |x| <= b
        h_trap = np.ones(N)
        mask = np.abs(X) > b
        h_trap[mask] = (0.25 - np.abs(X[mask])) / (0.25 - b)
        h_trap = np.clip(h_trap, 0, None)
        val_trap = ratio_from_h(h_trap)
        update_best(h_trap, val_trap)
        
    # If we have time, do a very quick local polish on the best candidate
    # to ensure we are at the exact local minimum (which should be the flat one)
    if best_h is not None and time.time() < deadline - 10.0:
        try:
            # Use L-BFGS-B to refine
            # The uniform function is a plateau, so gradients might be 0.
            # We just verify if a tiny perturbation helps (it shouldn't)
            res = opt.minimize(
                lambda h: ratio_from_h(h), 
                best_h, 
                method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            h_opt = np.clip(res.x, 0, None)
            val_opt = ratio_from_h(h_opt)
            update_best(h_opt, val_opt)
        except Exception:
            pass
            
    if best_h is None:
        return list(np.ones(N))
        
    return list(best_h)
