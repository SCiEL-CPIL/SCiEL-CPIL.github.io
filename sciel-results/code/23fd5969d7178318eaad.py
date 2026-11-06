import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 145.0  # Safety margin

# Precompute indices
_i_arr = np.arange(N, dtype=float)

# Precompute atom templates for sparse representation
# Use multiple diverse atoms for richer representation
_ATOM_TEMPLATES = []
_ATOM_TEMPLATES.append(np.hamming(8))
_ATOM_TEMPLATES.append(np.hanning(12))
_ATOM_TEMPLATES.append(np.blackman(16))
_ATOM_TEMPLATES.append(np.ones(6) * 0.5)  # Flat atom
_ATOM_TEMPLATES.append(np.hamming(20))
# Normalize each template
for i in range(len(_ATOM_TEMPLATES)):
    _ATOM_TEMPLATES[i] = _ATOM_TEMPLATES[i] / np.sum(_ATOM_TEMPLATES[i])

def _value(h):
    """Compute the exact task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    # Fast autoconvolution via FFT
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    m = float(np.max(conv))
    val = abs(2.0 * N * m / (s * s))
    return val

def _sparse_to_h(params):
    """
    Reconstruct h from sparse atom parameters.
    params: [amp_0, offset_0, atom_idx_0, amp_1, offset_1, atom_idx_1, ...]
    Each atom is placed at offset position with given amplitude.
    """
    h = np.zeros(N)
    n_atoms = len(params) // 3
    for i in range(n_atoms):
        amp = params[3*i]
        offset = int(round(params[3*i + 1]))
        atom_idx = int(round(params[3*i + 2]))
        
        # Clamp offset to valid range
        template = _ATOM_TEMPLATES[atom_idx % len(_ATOM_TEMPLATES)]
        tmpl_len = len(template)
        offset = np.clip(offset, 0, N - tmpl_len)
        
        # Add atom to h
        h[offset:offset+tmpl_len] += amp * template
    
    return h

def _smooth_sparse_obj(params, beta=50.0):
    """Smooth surrogate for sparse atom objective."""
    h = _sparse_to_h(params)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    # LogSumExp approximation of max
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    return val

def _exact_sparse_obj(params):
    """Exact objective for sparse atom parameters."""
    h = _sparse_to_h(params)
    return _value(h)

def _smooth_value(h, beta=50.0):
    """
    Smooth surrogate for the objective.
    Uses LogSumExp to approximate max(conv).
    """
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    # Shift to prevent overflow in exp
    m = np.max(conv)
    scaled = beta * (conv - m)
    # Clip to avoid numerical issues
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    return val

def _dct_h(coeffs):
    """Reconstruct h from DCT-II coefficients."""
    h = scipy_idct(coeffs, type=2, n=N, norm='ortho')
    return h

def _dct_smooth_obj(coeffs, beta=50.0):
    """Objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_value(h, beta)

def _dct_exact_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

def _generate_initial_dct(rng, num_coeffs):
    """Generate random DCT coefficients with low-frequency bias."""
    coeffs = np.zeros(num_coeffs)
    # Add a strong DC component (index 0) to ensure non-zero sum
    coeffs[0] = rng.normal(0, 1.0) * 10.0
    
    # Add random energy to other frequencies
    weights = 1.0 / (1.0 + np.arange(num_coeffs))
    coeffs[1:] = rng.normal(0, 1.0, size=num_coeffs - 1) * weights[1:] * 2.0
    
    return coeffs

def _generate_initial_sparse(rng, n_atoms):
    """Generate random sparse atom parameters with diverse atom types."""
    params = np.zeros(n_atoms * 3)
    n_templates = len(_ATOM_TEMPLATES)
    for i in range(n_atoms):
        # Random amplitude
        params[3*i] = rng.normal(0, 1.0) * 3.0
        # Random offset - spread out more
        params[3*i + 1] = rng.integers(0, N - 1)
        # Random atom type
        params[3*i + 2] = rng.integers(0, n_templates)
    return params

def _coordinate_descent_polish(params, max_iter=200, tol=1e-8):
    """
    Coordinate descent polish on sparse atom parameters.
    Cycles through each parameter, optimizing it one at a time.
    """
    current_val = _exact_sparse_obj(params)
    if not np.isfinite(current_val):
        return params
    
    n_params = len(params)
    for iteration in range(max_iter):
        if not deadline_ok():
            break
            
        improved = False
        for i in range(n_params):
            if not deadline_ok():
                    break
                    
            # Try small perturbations around current value
            base = params.copy()
            val_i = current_val
            
            # Determine step size based on parameter type
            if i % 3 == 0:  # amplitude
                step = 0.1
            elif i % 3 == 1:  # offset (discrete)
                step = 1
            else:  # atom index (discrete)
                step = 1
            
            for delta in [-step, -step/2, step/2, step]:
                test_params = base.copy()
                test_params[i] = base[i] + delta
                
                # For discrete params, round
                if i % 3 != 0:
                    test_params[i] = round(test_params[i])
                    if i % 3 == 2:  # atom index
                        test_params[i] = int(test_params[i]) % len(_ATOM_TEMPLATES)
                
                test_val = _exact_sparse_obj(test_params)
                if test_val < val_i - tol:
                    val_i = test_val
                    base = test_params.copy()
                    improved = True
        
        if not improved:
            break
            
        params = base
        current_val = val_i
    
    return params

def construct():
    global _t0
    _t0 = time.time()
    
    # Start with the seed solution
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    rng = np.random.default_rng(42)
    
    # Strategy 1: DCT-based optimization (keep as fallback)
    K = 400
    dct_restarts = 5
    
    for r in range(dct_restarts):
        if not deadline_ok():
            break
            
        try:
            coeffs0 = _generate_initial_dct(rng, K)
            
            res = minimize(
                lambda c: _dct_smooth_obj(c, beta=20.0),
                coeffs0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if not deadline_ok():
                break
                
            coeffs_opt = res.x if res.x is not None else coeffs0
            
            if deadline_ok():
                res2 = minimize(
                    lambda c: _dct_smooth_obj(c, beta=100.0),
                    coeffs_opt,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res2.x is not None:
                    coeffs_opt = res2.x

            h_candidate = _dct_h(coeffs_opt)
            
            if np.all(np.isfinite(h_candidate)):
                update_best(h_candidate)
                
        except Exception:
            pass

    # Strategy 2: Sparse atom optimization with diverse atoms
    # Try different numbers of atoms
    for n_atoms in [8, 12, 16, 24, 32]:
        if not deadline_ok():
            break
            
        sparse_restarts = 3
        
        for r in range(sparse_restarts):
            if not deadline_ok():
                break
                
            try:
                params0 = _generate_initial_sparse(rng, n_atoms)
                
                # Phase 1: Very smooth coarse optimization
                res = minimize(
                    lambda p: _smooth_sparse_obj(p, beta=5.0),
                    params0,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if not deadline_ok():
                    break
                    
                params_opt = res.x if res.x is not None else params0
                
                # Phase 2: Medium smoothness
                if deadline_ok():
                    res2 = minimize(
                        lambda p: _smooth_sparse_obj(p, beta=20.0),
                        params_opt,
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res2.x is not None:
                        params_opt = res2.x

                # Phase 3: Sharp smoothness
                if deadline_ok():
                    res3 = minimize(
                        lambda p: _smooth_sparse_obj(p, beta=100.0),
                        params_opt,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res3.x is not None:
                        params_opt = res3.x

                # Phase 4: Coordinate descent polish
                if deadline_ok():
                    params_opt = _coordinate_descent_polish(params_opt, max_iter=100)

                h_candidate = _sparse_to_h(params_opt)
                
                if np.all(np.isfinite(h_candidate)):
                    update_best(h_candidate)
                    
            except Exception:
                pass

    # Strategy 3: Symmetric patterns - try specific structured patterns
    if deadline_ok():
        # Try alternating sign patterns with varying block sizes
        for block_size in [2, 4, 8, 16]:
            if not deadline_ok():
                break
            h_sym = np.zeros(N)
            for i in range(N):
                block = i // block_size
                h_sym[i] = 1.0 if block % 2 == 0 else -1.0
            update_best(h_sym)
        
        # Try sine-modulated patterns
        for freq in [1, 2, 3, 5, 8]:
            if not deadline_ok():
                break
            t = np.linspace(0, 2*np.pi*freq, N)
            h_sine = np.cos(t)
            update_best(h_sine)
        
        # Try random sign patterns with smoothing
        for _ in range(5):
            if not deadline_ok():
                break
            h_rand = rng.choice([-1.0, 1.0], size=N)
            # Smooth with a short window
            kernel = np.ones(5) / 5.0
            h_smooth = np.convolve(h_rand, kernel, mode='same')
            update_best(h_smooth)

    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    # Simple self-test
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Shape: {h.shape}")
