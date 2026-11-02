"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv(x):
    """Compute autoconvolution via FFT with zero-padding to 3N-1."""
    n = len(x)
    size = 3 * n - 1
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:2*n-1]
    return ac

def _objective_and_grad(x):
    """Objective: max(|autoconv|) / (sum(x))^2, with gradient."""
    ac = _autoconv(x)
    abs_ac = np.abs(ac)
    s = np.sum(x)
    if abs(s) < 1e-12:
        s = 1e-12
    obj = np.max(abs_ac) / (s * s)
    
    # Gradient: d/dx [max_k |ac_k| / s^2]
    # Let M = max_k |ac_k|, s = sum(x)
    # obj = M / s^2
    # d(obj)/dx = (dM/dx * s^2 - M * 2s * ds/dx) / s^4
    #           = (dM/dx / s^2) - (2M * ds/dx) / s^3
    # ds/dx = 1 for all i
    # dM/dx: M = max_k |ac_k|, ac_k = sum_i x_i x_{i+k}
    # d|ac_k|/dx_i = sign(ac_k) * (x_{i+k} + x_{i-k}) (with boundary handling)
    # For the argmax k*, dM/dx_i = sign(ac_k*) * (x_{i+k*} + x_{i-k*})
    
    k_star = np.argmax(abs_ac)
    sign_ac = np.sign(ac[k_star])
    if sign_ac == 0:
        sign_ac = 1.0
    
    k = k_star
    n = len(x)
    grad_M = np.zeros(n)
    # ac_k = sum_{i=0}^{n-1-k} x[i] * x[i+k] for k >= 0
    # d(ac_k)/dx[i] = x[i+k] if i <= n-1-k, plus x[i-k] if i >= k
    # Actually ac_k = sum_{i} x[i] x[i+k] where valid indices
    # d/dx[i] of ac_k = x[i+k] (if i+k < n) + x[i-k] (if i-k >= 0)
    
    if k == 0:
        # ac_0 = sum x[i]^2, d/dx[i] = 2*x[i]
        grad_M = 2.0 * sign_ac * x
    else:
        # For i from 0 to n-1-k: contribution x[i+k]
        # For i from k to n-1: contribution x[i-k]
        grad_M[:n-k] += sign_ac * x[k:]
        grad_M[k:] += sign_ac * x[:n-k]
    
    ds_dx = 1.0
    grad = grad_M / (s * s) - 2.0 * (np.max(abs_ac) / (s * s * s)) * ds_dx
    return obj, grad

def _local_peak_polish(x, max_iters=500):
    """Iteratively reduce the dominant peak by adjusting nearby elements."""
    best_x = x.copy()
    best_obj, _ = _objective_and_grad(best_x)
    for it in range(max_iters):
        ac = _autoconv(best_x)
        abs_ac = np.abs(ac)
        k_star = np.argmax(abs_ac)
        peak_val = ac[k_star]
        if abs(peak_val) < 1e-10:
            break
        
        # Try to reduce the peak by flipping or adjusting elements
        # For lag k, ac_k = sum x[i]*x[i+k]
        # To reduce |ac_k|, we can flip signs of some x[i] or x[i+k]
        # Try flipping the element that contributes most to the peak
        n = len(best_x)
        k = k_star
        if k == 0:
            # ac_0 = sum x[i]^2, can't reduce by flipping
            break
        
        # Contributions to ac_k: x[i]*x[i+k] for i=0..n-1-k
        contribs = best_x[:n-k] * best_x[k:]
        # The element with largest |contrib| that we can flip
        # Flipping x[i] changes ac_k by -2*x[i]*x[i+k]
        # Flipping x[i+k] changes ac_k by -2*x[i]*x[i+k]
        # So flipping either reduces ac_k by 2*|contrib| if sign matches
        
        # Try flipping the index with largest |contrib| * sign(peak)
        if peak_val > 0:
            # Want to reduce ac_k, so flip where contrib > 0
            target_idx = np.argmax(contribs)
        else:
            # Want to increase ac_k (make less negative), flip where contrib < 0
            target_idx = np.argmin(contribs)
        
        # Try flipping x[target_idx] and x[target_idx+k]
        best_flip_obj = best_obj
        best_flip_x = best_x.copy()
        
        for flip_pos in [target_idx, target_idx + k]:
            if flip_pos < 0 or flip_pos >= n:
                continue
            trial = best_x.copy()
            trial[flip_pos] *= -1
            trial_obj, _ = _objective_and_grad(trial)
            if trial_obj < best_flip_obj:
                best_flip_obj = trial_obj
                best_flip_x = trial
        
        if best_flip_obj >= best_obj - 1e-12:
            # No improvement from single flip, try small perturbations
            improved = False
            for _ in range(10):
                trial = best_x.copy()
                # Randomly perturb a few elements
                idxs = np.random.choice(n, size=min(5, n), replace=False)
                for idx in idxs:
                    if np.random.random() < 0.5:
                        trial[idx] *= -1
                    else:
                        trial[idx] += np.random.uniform(-0.5, 0.5)
                trial_obj, _ = _objective_and_grad(trial)
                if trial_obj < best_obj - 1e-12:
                    best_obj = trial_obj
                    best_x = trial
                    improved = True
                    break
            if not improved:
                break
        else:
            best_obj = best_flip_obj
            best_x = best_flip_x
    
    return best_x, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    best_x = None
    best_obj = np.inf
    
    # Diverse initializations
    inits = []
    # All ones (baseline)
    inits.append(np.ones(N))
    # Random signs
    inits.append(np.random.choice([-1.0, 1.0], size=N))
    # Sparse: mostly zeros with some ones
    for trial in range(5):
        x = np.zeros(N)
        n_ones = N // 4
        idxs = np.random.choice(N, size=n_ones, replace=False)
        x[idxs] = np.random.choice([-1.0, 1.0], size=n_ones)
        inits.append(x)
    # High frequency alternating
    inits.append(np.tile([1.0, -1.0], N // 2))
    # Block patterns
    for block_size in [2, 4, 8, 16]:
        pattern = np.zeros(N)
        for i in range(0, N, block_size * 2):
            pattern[i:i+block_size] = 1.0
            if i + 2*block_size <= N:
                pattern[i+block_size:i+2*block_size] = -1.0
        inits.append(pattern)
    # Cosine-like
    t = np.linspace(0, 4*np.pi, N)
    inits.append(np.sign(np.cos(t)))
    
    for x0 in inits:
        if time.time() > deadline:
            break
        
        # L-BFGS optimization
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 500}
            )
            x_opt = res.x
            obj_opt, _ = _objective_and_grad(x_opt)
            
            # Peak polish
            x_polished, obj_polished = _local_peak_polish(x_opt, max_iters=200)
            
            if obj_polished < best_obj:
                best_obj = obj_polished
                best_x = x_polished.copy()
        except Exception:
            continue
    
    # Final high-intensity polish if time permits
    if best_x is not None and time.time() < deadline - 5:
        x_final, obj_final = _local_peak_polish(best_x, max_iters=1000)
        if obj_final < best_obj:
            best_obj = obj_final
            best_x = x_final
    
    if best_x is None:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
