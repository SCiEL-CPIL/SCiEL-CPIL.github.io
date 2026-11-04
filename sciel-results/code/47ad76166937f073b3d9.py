import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute angle grid for polynomial representation
# theta maps [-1/4, 1/4] to [0, 2*pi)
THETA = 2.0 * np.pi * (X + 0.25) / 0.5  # [0, 2*pi)
# Precompute powers of z = e^{i*theta} for up to degree 20
MAX_DEG = 20
Z_POWERS = np.zeros((N, MAX_DEG + 1), dtype=np.complex128)
for d in range(MAX_DEG + 1):
    Z_POWERS[:, d] = np.exp(1j * d * THETA)

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
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

def quick_ratio(h):
    """Fast ratio computation for coordinate descent."""
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def poly_to_h(coeffs):
    """
    Convert complex polynomial coefficients to nonnegative step function.
    coeffs: array of shape (MAX_DEG+1,) of complex numbers
    h(x) = |sum_{k=0}^{MAX_DEG} c_k * e^{i*k*theta}|^2
    This is automatically nonnegative.
    """
    z = Z_POWERS  # (N, MAX_DEG+1)
    # h_n = |sum_k c_k * z_n^k|^2
    # Compute the complex sum first
    csum = np.sum(coeffs[None, :] * z, axis=1)  # (N,)
    h = np.abs(csum) ** 2
    return h

def poly_ratio(coeffs):
    """Compute ratio from polynomial coefficients."""
    h = poly_to_h(coeffs)
    return ratio_from_h(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # === Phase 1: Polynomial representation with Differential Evolution ===
    # Represent h as |P(e^{i theta})|^2 where P is a polynomial of degree D
    # Coefficients are complex, so we parameterize as real+imag pairs
    # For degree D, we have 2*(D+1) real parameters
    
    rng = np.random.default_rng(42)
    
    # Try different polynomial degrees
    for D in [4, 6, 8, 10]:
        if time.time() > deadline - 60.0:
            break
        
        n_params = 2 * (D + 1)
        
        # Define bounds for DE
        # Center coefficient larger, others smaller
        bounds = []
        for k in range(D + 1):
            if k == 0:
                bounds.append((-3.0, 3.0))  # real part
                bounds.append((-3.0, 3.0))  # imag part
            else:
                bounds.append((-1.5, 1.5))
                bounds.append((-1.5, 1.5))
        
        # Multiple random starts for DE
        n_de_starts = 3
        for de_start in range(n_de_starts):
            if time.time() > deadline - 50.0:
                break
            
            # Initial population
            pop = np.random.uniform(
                [b[0] for b in bounds],
                [b[1] for b in bounds],
                size=(20, n_params)
            )
            # Ensure at least one nonzero coefficient
            for i in range(20):
                if np.all(np.abs(pop[i]) < 0.1):
                    pop[i, 0] = 1.0
            
            def de_objective(x):
                # x is real array of length n_params
                coeffs = x[0::2] + 1j * x[1::2]
                return poly_ratio(coeffs)
            
            try:
                result = opt.differential_evolution(
                    de_objective,
                    bounds,
                    maxiter=80,
                    popsize=15,
                    tol=1e-4,
                    seed=42 + de_start,
                    polish=True,
                    init=pop
                )
                if result.fun < best_val:
                    coeffs_opt = result.x[0::2] + 1j * result.x[1::2]
                    h_opt = poly_to_h(coeffs_opt)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 2: Local refinement using Nelder-Mead on polynomial coefficients ===
    if time.time() < deadline - 30.0:
        # Use the best polynomial found, or start from a reasonable guess
        D_refine = 8
        n_params = 2 * (D_refine + 1)
        
        # Try several initial points
        n_refine_starts = 5
        for rs in range(n_refine_starts):
            if time.time() > deadline - 25.0:
                break
            
            coeffs_init = np.zeros(D_refine + 1, dtype=np.complex128)
            coeffs_init[0] = 1.0 + 0.5j
            for k in range(1, D_refine + 1):
                coeffs_init[k] = complex(
                    float(rng.uniform(-0.3, 0.3)),
                    float(rng.uniform(-0.3, 0.3))
                )
            
            x0 = np.zeros(n_params)
            x0[0::2] = coeffs_init.real
            x0[1::2] = coeffs_init.imag
            
            def refine_objective(x):
                coeffs = x[0::2] + 1j * x[1::2]
                return poly_ratio(coeffs)
            
            try:
                result = opt.minimize(
                    refine_objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                if result.fun < best_val:
                    coeffs_opt = result.x[0::2] + 1j * result.x[1::2]
                    h_opt = poly_to_h(coeffs_opt)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 3: Direct bin-level coordinate descent polish ===
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val
        
        n_iter = 8000
        for it in range(n_iter):
            if time.time() > deadline - 3.0:
                break
            
            # Pick random coordinates
            n_coords = int(rng2.integers(1, 8))
            idxs = rng2.choice(N, size=n_coords, replace=False)
            
            h_try = cur_h.copy()
            for idx in idxs:
                factor = float(rng2.uniform(0.85, 1.15))
                h_try[idx] = max(0.0, cur_h[idx] * factor)
            
            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
    
    # === Phase 4: Hand-crafted shapes ===
    if time.time() < deadline - 3.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = [
            np.exp(-2.0 * x_norm**2),
            np.exp(-0.5 * x_norm**2),
            0.5 + 0.5 * np.exp(-8.0 * x_norm**2),
            np.exp(-20.0 * (x_norm - 0.3)**2) + np.exp(-20.0 * (x_norm + 0.3)**2),
            0.1 + np.exp(-30.0 * (x_norm - 0.4)**2) + np.exp(-30.0 * (x_norm + 0.4)**2),
            1.0 - 0.3 * np.exp(-5.0 * x_norm**2),
            0.5 + 0.5 * x_norm**2,
            0.5 + 0.5 * np.cos(np.pi * x_norm),
            np.exp(-4.0 * x_norm**2) + 0.2 * np.exp(-10.0 * (x_norm - 0.5)**2),
            np.exp(-4.0 * x_norm**2) + 0.2 * np.exp(-10.0 * (x_norm + 0.5)**2),
        ]
        
        for h in crafted:
            if time.time() > deadline - 2.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            update_best(h, v)

    return list(best_h)
