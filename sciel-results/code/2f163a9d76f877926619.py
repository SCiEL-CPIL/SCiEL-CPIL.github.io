"""Suite task: signed equal-width step functions minimizing max|autoconv|/integral^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _check_deadline():
    return (time.time() - START_TIME) > TIME_LIMIT


def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C


def _score(x):
    """Raw objective: max|autoconv| / (sum)^2  (lower is better)."""
    ac = _autoconv_fft(x)
    S = x.sum()
    if abs(S) < 1e-12:
        return float('inf')
    return float(np.max(np.abs(ac)) / (S * S))


def _softmax_objective_and_grad(x, beta):
    """
    Compute soft-max surrogate objective and gradient.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient computation
    # val = soft_max / s^2
    # d(val)/dx[j] = (1/s^2) * d(soft_max)/dx[j] - (2*soft_max/s^3) * 1
    
    # d(soft_max)/dx[j] = sum_n weights[n] * sign(ac[n]) * d(ac[n])/dx[j]
    # d(ac[n])/dx[j] = x[j+n] + x[j-n] (with boundary handling)
    
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1[j] = sum_n g[n] * x[j+n] (correlation)
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # term2[j] = sum_n g[n] * x[j-n]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before the 150s wall-clock kill
    rng = np.random.default_rng(12345)

    # ---- Base flat-top (Hanning) window ----
    base = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N) / N))
    base = base / base.max()
    W = base.sum()

    # ---- Initial sign pattern: all +1 ----
    s = np.ones(N)
    
    # Try multiple initial sign patterns
    init_patterns = [
        np.ones(N),
        np.sign(np.sin(np.pi * np.arange(N) / N)),
        np.sign(np.cos(2 * np.pi * np.arange(N) / N)),
        np.sign(np.sin(2 * np.pi * np.arange(N) / N)),
        np.sign(np.sin(3 * np.pi * np.arange(N) / N)),
        rng.choice([-1, 1], size=N),
        rng.choice([-1, 1], size=N),
    ]
    
    best_x = None
    best_val = float('inf')
    
    for init_s in init_patterns:
        if _check_deadline():
            break
        x0 = init_s * base
        val = _score(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if best_x is None:
        best_x = s * base
        best_val = _score(best_x)
    
    if _check_deadline():
        return best_x
    
    # ---- Gradient-based optimization with smooth surrogate ----
    # Use L-BFGS-B on the continuous relaxation, then project back to sign pattern
    for beta in [50.0, 100.0, 200.0]:
        if _check_deadline():
            break
        
        def smooth_obj(x):
            val, grad = _softmax_objective_and_grad(x, beta)
            return val, grad
        
        try:
            result = minimize(smooth_obj, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-12})
            
            if result.fun < best_val:
                # Project back to sign pattern
                x_opt = result.x
                s_opt = np.sign(x_opt)
                # Keep magnitude as base
                x_projected = s_opt * base
                val_projected = _score(x_projected)
                
                if val_projected < best_val:
                    best_val = val_projected
                    best_x = x_projected.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # ---- Local sign-flip polish (exact objective, accept improving flips) ----
    s = np.sign(best_x)
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        for i in range(N):
            if time.time() > DEADLINE:
                break
            old = s[i]
            s[i] = -old
            x_trial = s * base
            o = _score(x_trial)
            if o < best_val:
                best_val = o
                best_x = x_trial.copy()
                improved = True
            else:
                s[i] = old
        s = np.sign(best_x)
    
    # ---- Return the signed step heights ----
    return best_x


if __name__ == "__main__":
    x = construct()
    print("score:", _score(x))
    print("seed value (all-ones):", 1.0 / N)
