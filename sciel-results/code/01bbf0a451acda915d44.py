"""Combined program: structured search + adaptive refinement for signed step functions."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115  # safety margin before 120s kill
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _objective_grad(x):
    """Compute objective and its gradient (approximate via finite differences on FFT)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    ac = _autoconv_fft(x)
    max_abs = np.max(np.abs(ac))
    val = max_abs / (s * s)
    
    # Gradient: use subgradient of max(|ac|)
    # d/dx max(|ac|) = sign(ac[k*]) * d(ac[k*])/dx where k* is argmax
    k_star = np.argmax(np.abs(ac))
    sign_ac = np.sign(ac[k_star])
    if ac[k_star] == 0:
        sign_ac = 1.0
    
    # d(ac[k])/dx[j] = x[k-j] if k-j in [0,N-1] else 0
    # This is the cross-correlation of x with a delta at k
    # More precisely: ac[k] = sum_j x[j]*x[k-j]
    # d(ac[k])/dx[j] = x[k-j] + x[j] (if we consider symmetric)
    # Actually ac[k] = sum_{j=0}^{N-1} x[j] * x[k-j] where x[m]=0 for m outside [0,N-1]
    # d(ac[k])/dx[j] = x[k-j] (treating x as independent variables)
    # But since it's autoconvolution, both terms depend on x:
    # ac[k] = sum_j x[j]*x[k-j]
    # d(ac[k])/dx[j] = x[k-j] + x[j] (when k-j != j, i.e., 2j != k)
    # If 2j == k, then d/dx[j] (x[j]^2) = 2*x[j]
    
    # Let's compute gradient numerically for simplicity and correctness
    grad = np.zeros(N)
    eps = 1e-7
    # Only compute gradient at a few points for speed? No, we need full gradient.
    # Alternative: use FFT-based gradient
    # ac = ifft(|fft(x)|^2)
    # d(ac[k])/dx = ? This is complex. Let's use finite differences on a subset or full.
    # Full finite difference is N * cost(fft) which is 400 * ~O(N log N) - feasible.
    
    # Actually, let's just use the analytical subgradient approach more carefully.
    # For the max term: let k* be argmax |ac[k]|
    # g = sign(ac[k*]) * (d(ac[k*])/dx)
    # d(ac[k*])/dx[j] = x[k*-j] + x[j] if k*-j in [0,N-1] and j in [0,N-1]
    # But careful: ac[k] = sum_{j=max(0,k-N+1)}^{min(k,N-1)} x[j]*x[k-j]
    # d(ac[k])/dx[j] = x[k-j] (from the term where first index is j) + x[j] (from the term where second index is j, i.e., k-j'=j => j'=k-j)
    # So if k-j is in [0,N-1], then d(ac[k])/dx[j] = x[k-j] + x[j]
    # Wait, that's double counting when j = k-j.
    # Let's just do: for each j, if 0 <= k-j < N, add x[k-j]. Then if j != k-j and 0 <= k-j < N, add x[j]? No.
    # ac[k] = sum_j x[j] x[k-j]
    # d/dx[j] ac[k] = x[k-j] (derivative of x[j] term) + sum_j' d(x[j']x[k-j'])/dx[j] where j'=j in second factor
    # = x[k-j] + x[j] (if k-j is valid index and j != k-j)
    # = 2*x[j] if j = k-j (i.e., 2j = k)
    # = x[k-j] if k-j invalid? No, if k-j invalid, x[k-j]=0, so derivative is 0 from that term.
    # But the second term: x[j] appears when j' = k-j in the sum, i.e., when j = k-j'.
    # So d(ac[k])/dx[j] = x[k-j] * I(0<=k-j<N) + x[j] * I(0<=k-j<N) where the second term is from the j' such that k-j'=j, i.e., j'=k-j.
    # Wait, this is getting confusing. Let me just use numerical gradient with a small step for the few most important components, or full FD.
    
    # Full FD gradient:
    grad = np.zeros(N)
    for j in range(N):
        x_plus = x.copy()
        x_plus[j] += eps
        ac_plus = _autoconv_fft(x_plus)
        max_plus = np.max(np.abs(ac_plus))
        # This is expensive: N FFTs of size ~1024. 400 * 1024 log 1024 ~ 400 * 1024 * 10 ~ 4M ops. Should be fine.
        # But we also need to handle the max subgradient.
        # For simplicity, let's just compute the value and use scipy's BFGS with numerical gradient.
        pass
    
    # Let's just return value and let scipy compute gradient numerically.
    return val, None

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Try structured initializations
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
    ]
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]:
        initializations.append(r ** np.arange(N))
    for r in [1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    t = np.linspace(0, 1, N)
    initializations.append(1.0 + 0.5 * np.cos(np.pi * t))
    initializations.append(1.0 + np.cos(np.pi * t))
    initializations.append(1.0 + t**2)
    initializations.append(1.0 + 4.0 * (t - 0.5)**2)
    initializations.append(np.exp(2.0 * t))
    initializations.append(np.exp(-2.0 * t))
    initializations.append(1.0 + 0.3 * np.sin(2 * np.pi * t))
    initializations.append(1.0 + 0.5 * np.sin(np.pi * t))
    # Signed variants
    initializations.append(np.ones(N) * np.sign(np.sin(np.pi * t)))
    initializations.append(np.ones(N) * np.sign(np.cos(np.pi * t)))
    initializations.append(1.0 + 0.5 * np.sin(2 * np.pi * t))
    initializations.append(1.0 - 0.5 * np.sin(2 * np.pi * t))
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Local optimization using scipy
    def _obj_func(x):
        return _objective(x)
    
    # Try BFGS from the best initial point
    try:
        result = minimize(_obj_func, best_x, method='BFGS', options={'maxiter': 500, 'gtol': 1e-6})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Random perturbation refinement
    rng = np.random.default_rng(42)
    step_size = 0.05
    stall_count = 0
    max_iters = 200000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        if iteration % 1000 == 0 and (time.time() - START_TIME) > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.9998)
        else:
            stall_count += 1
            if stall_count % 5000 == 0:
                step_size = min(0.5, step_size * 1.05)
                # Occasionally try a larger jump
                if stall_count % 25000 == 0:
                    big_pert = rng.normal(0, step_size * 5, size=N)
                    x_big = best_x + big_pert
                    val_big = _objective(x_big)
                    if val_big < best_val:
                        best_val = val_big
                        best_x = x_big.copy()
    
    return best_x
# EVOLVE-BLOCK-END
