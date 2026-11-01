import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0
M = 4 * N
DX = 0.5 / M

def _compute_c3(signed_step):
    """Compute C3 = max |autoconvolution| for a signed step function on [-1/4, 1/4]."""
    f_fine = np.repeat(signed_step, M // N)
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * DX
    return np.max(np.abs(autoconv))

def _soft_c3(x, beta=25.0):
    """Smooth approximation of C3 for continuous signal x (tanh-transformed)."""
    f_signal = np.tanh(x)
    f_fine = np.repeat(f_signal, M // N)
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * DX
    abs_autoconv = np.abs(autoconv)
    m = np.max(abs_autoconv)
    if m < 1e-15:
        return 0.0
    exp_vals = np.exp(beta * (abs_autoconv - m))
    return m + np.log(np.sum(exp_vals)) / beta

def _local_search(signed_step, max_iters=30000, deadline_check=None):
    """Local search: flip single bits to reduce C3."""
    best = signed_step.copy()
    best_val = _compute_c3(best)
    no_improve = 0
    patience = 3000
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if deadline_check is not None and deadline_check():
            break
        idx = rng.randint(N)
        best[idx] *= -1
        new_val = _compute_c3(best)
        if new_val < best_val - 1e-12:
            best_val = new_val
            no_improve = 0
        else:
            best[idx] *= -1
            no_improve += 1
        if no_improve >= patience:
            break
    
    return best, best_val

def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() - start_time > DEADLINE - 5.0
    
    # Start with all ones
    best_step = np.ones(N, dtype=np.int8)
    best_val = _compute_c3(best_step)
    
    # Phase 1: Continuous relaxation via L-BFGS-B with tanh mapping
    beta = 25.0
    
    def obj(x):
        return _soft_c3(x, beta)
    
    np.random.seed(42)
    x0 = np.random.randn(N) * 0.5
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if deadline_check():
            cont_sol = np.sign(result.x)
            cont_sol[cont_sol == 0] = 1
            cont_val = _compute_c3(cont_sol)
            if cont_val < best_val:
                best_step = cont_sol.astype(np.int8)
                best_val = cont_val
            return best_step
        
        cont_sol = np.sign(result.x)
        cont_sol[cont_sol == 0] = 1
        cont_val = _compute_c3(cont_sol)
        
        if cont_val < best_val:
            best_step = cont_sol.astype(np.int8)
            best_val = cont_val
        
    except Exception:
        pass
    
    # Phase 2: Local search refinement
    if not deadline_check():
        best_step, best_val = _local_search(best_step, max_iters=20000, deadline_check=deadline_check)
    
    # Phase 3: Additional restarts if time permits
    if not deadline_check():
        for restart in range(3):
            if deadline_check():
                break
            x_r = np.random.randn(N) * 0.8
            try:
                result_r = minimize(
                    obj, x_r, method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                if deadline_check():
                    break
                sol_r = np.sign(result_r.x)
                sol_r[sol_r == 0] = 1
                val_r = _compute_c3(sol_r)
                if val_r < best_val:
                    best_step = sol_r.astype(np.int8)
                    best_val = val_r
                    if not deadline_check():
                        best_step, best_val = _local_search(best_step, max_iters=10000, deadline_check=deadline_check)
            except Exception:
                continue
    
    return best_step
