"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_value(h):
    """Compute the objective value."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    mx = np.max(conv)
    return abs(2.0 * N * mx / (s * s))

def _fft_convolve(h):
    """Fast full convolution via FFT."""
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for the max."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = _fft_convolve(h)
    # log-sum-exp trick for smooth max
    c = np.max(conv)
    if c == 0:
        smax = 0.0
    else:
        smax = c + np.log(np.sum(np.exp((conv - c) / beta))) / beta
    return abs(2.0 * N * smax / (s * s))

def _atom_basis():
    """Create atom basis vectors: shifted, scaled, sign-flipped copies of a Hamming window."""
    atom_len = 12
    atom = np.hamming(atom_len)
    basis = []
    # Positive and negative atoms at various positions
    for pos in range(0, N, atom_len // 2):
        if pos + atom_len <= N:
            # Positive atom
            v = np.zeros(N)
            v[pos:pos+atom_len] = atom
            basis.append(v)
            # Negative atom
            v_neg = np.zeros(N)
            v_neg[pos:pos+atom_len] = -atom
            basis.append(v_neg)
    return np.array(basis)  # shape (n_basis, N)

def _atom_to_h(params, basis):
    """Convert atom parameters to h vector."""
    # params[0] is global offset, params[1:] are atom amplitudes
    offset = params[0]
    amps = params[1:]
    h = offset + amps @ basis
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # = 2.0
    
    # Strategy 1: Atom-based parameterization
    basis = _atom_basis()
    n_basis = basis.shape[0]
    
    def atom_objective(params):
        h = _atom_to_h(params, basis)
        return _smooth_max_value(h, beta=30.0)
    
    # Try multiple initializations for atom parameters
    rng = np.random.RandomState(42)
    
    for init_idx in range(10):
        if time.time() > deadline - 60:
            break
        
        if init_idx == 0:
            # Start with all zeros (pure offset)
            params_init = np.zeros(n_basis + 1)
            params_init[0] = 1.0  # offset = 1
        elif init_idx == 1:
            # Alternating pattern encoded in atoms
            params_init = np.zeros(n_basis + 1)
            params_init[0] = 0.5
            for j in range(n_basis):
                # Alternate signs based on position
                pos = j * (atom_len // 2)
                if pos % (2 * atom_len) < atom_len:
                    params_init[j + 1] = 1.0
                else:
                    params_init[j + 1] = -1.0
        else:
            # Random initialization
            params_init = np.zeros(n_basis + 1)
            params_init[0] = 1.0
            params_init[1:] = rng.randn(n_basis) * 0.1
        
        try:
            res = minimize(atom_objective, params_init, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                h_candidate = _atom_to_h(res.x, basis)
                if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    val = _compute_value(h_candidate)
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate
        except Exception:
            pass
    
    # Strategy 2: Coordinate descent on raw h with smooth surrogate
    if time.time() < deadline - 40:
        def cd_objective(h):
            return _smooth_max_value(h, beta=20.0)
        
        # Several restarts from perturbed best_h
        for restart in range(8):
            if time.time() > deadline - 20:
                break
            
            if restart == 0:
                h_init = best_h.copy()
            else:
                h_init = best_h + 0.05 * rng.randn(N)
            
            # Coordinate descent: iterate over elements
            h_curr = h_init.copy()
            f_curr = cd_objective(h_curr)
            
            step_size = 0.1
            for it in range(50):
                if time.time() > deadline - 15:
                    break
                improved = False
                for i in range(0, N, 2):  # Step by 2 for speed
                    if time.time() > deadline - 15:
                        break
                    old_val = h_curr[i]
                    h_curr[i] = old_val + step_size
                    f_new = cd_objective(h_curr)
                    if f_new < f_curr:
                        f_curr = f_new
                        improved = True
                    else:
                        h_curr[i] = old_val - step_size
                        f_new = cd_objective(h_curr)
                        if f_new < f_curr:
                            f_curr = f_new
                            improved = True
                        else:
                            h_curr[i] = old_val
                
                if not improved:
                    break
            
            val = _compute_value(h_curr)
            if val < best_val and np.all(np.isfinite(h_curr)) and abs(np.sum(h_curr)) > 1e-10:
                best_val = val
                best_h = h_curr.copy()
    
    # Strategy 3: Final L-BFGS-B polish on raw h
    if time.time() < deadline - 10:
        def polish_objective(h):
            return _smooth_max_value(h, beta=100.0)
        
        try:
            res = minimize(polish_objective, best_h, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                h_polished = res.x
                if np.all(np.isfinite(h_polished)) and abs(np.sum(h_polished)) > 1e-10:
                    val = _compute_value(h_polished)
                    if val < best_val:
                        best_val = val
                        best_h = h_polished
        except Exception:
            pass
    
    # Final validation
    final_val = _compute_value(best_h)
    if final_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-10:
        return best_h
    else:
        return np.ones(N)
# EVOLVE-BLOCK-END
