"""
Combined approach: Spectral parameterization + diverse candidates + robust gradient polish.

Strategy:
1. Generate diverse candidates: flat spectrum, chirps, random, block patterns
2. Optimize phases for flat spectrum candidates
3. L-BFGS-B two-pass optimization on smooth objective
4. Final gradient descent polish with backtracking line search
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _objective_smooth(x, temp=1.0):
    """Smooth approximation of the off-diagonal metric for optimization."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv(x)
    if len(c) > 1:
        abs_c = np.abs(c[1:])
    else:
        return 0.0
    max_c = np.max(abs_c)
    if max_c < 1e-15:
        return 0.0
    soft_max = max_c + temp * np.log(np.sum(np.exp((abs_c - max_c) / temp)))
    return soft_max / (s * s)

def _flat_spectrum_candidate(mag_profile, phases):
    """Build a real sequence from magnitude spectrum and phases."""
    n_half = N // 2 + 1
    X = np.zeros(n_half, dtype=complex)
    for k in range(n_half):
        X[k] = mag_profile[k] * np.exp(1j * phases[k])
    x = np.fft.irfft(X, n=N)
    return x

def _optimize_phases_flat(mag_profile, max_iter=150):
    """Optimize phases to minimize off-diagonal autoconvolution."""
    n_half = N // 2 + 1
    rng = np.random.default_rng(123)
    phases = rng.uniform(-np.pi, np.pi, n_half)
    
    def phase_obj(ph):
        x = _flat_spectrum_candidate(mag_profile, ph)
        return _metric_offdiag(x)
    
    best_phases = phases.copy()
    best_val = phase_obj(phases)
    
    try:
        res = minimize(
            phase_obj,
            phases,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_phases = res.x
    except Exception:
        pass
    
    x = _flat_spectrum_candidate(mag_profile, best_phases)
    return x, best_val

def _chirp_sequence(alpha, beta, gamma, phi0):
    """Generate a real-valued sequence from a quadratic chirp phase."""
    n = np.arange(N)
    phase = alpha * n**2 + beta * n + gamma + phi0
    x = np.cos(phase)
    return x

def _gradient_descent_polish(x0, max_iter=1500, initial_lr=0.005, deadline=None):
    """Gradient descent with backtracking line search for final polish."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = initial_lr
    temp = 0.5
    prev_val = best_val
    stall_count = 0
    max_stall = 80
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        exp_vals = np.exp((abs_c - max_c) / temp)
        sum_exp = np.sum(exp_vals)
        soft_max = max_c + temp * np.log(sum_exp)
        
        grad_abs_c = exp_vals / sum_exp
        
        c_off = c[1:]
        sign_c = np.sign(c_off)
        sign_c[abs_c < 1e-15] = 0.0
        grad_c = grad_abs_c * sign_c
        
        w = np.zeros(2 * N - 1)
        w[1:] = grad_c
        
        fsize = 1
        while fsize < 2 * N:
            fsize *= 2
        X = np.fft.rfft(x, fsize)
        W = np.fft.rfft(w, fsize)
        grad_x = np.fft.irfft(W * np.conj(X), fsize)[:N]
        
        s = np.sum(x)
        metric = soft_max / (s * s)
        
        grad_metric = grad_x / (s * s) - 2.0 * soft_max * np.ones(N) / (s ** 3)
        
        # Backtracking line search
        best_x_new = x
        best_val_new = prev_val
        for trial in range(12):
            x_new = x - lr * grad_metric
            s_new = np.sum(x_new)
            if abs(s_new) < 1e-12:
                break
            x_new = x_new / s_new
            val_new = _metric_offdiag(x_new)
            if val_new < prev_val * (1.0 - 1e-4):
                best_x_new = x_new
                best_val_new = val_new
                break
            lr *= 0.5
            if lr < 1e-12:
                break
        
        x = best_x_new
        prev_val = best_val_new
        
        if val_new < best_val:
            best_val = val_new
            best_x = x.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        lr *= 0.998
        temp *= 0.9995
        
        if stall_count > max_stall:
            break
    
    return best_x

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    n_half = N // 2 + 1
    candidates = []
    
    # 1. All ones baseline
    candidates.append(np.ones(N))
    
    # 2. Flat magnitude spectrum with optimized phases
    for trial in range(6):
        if time.time() > deadline - 70:
            break
        mag = np.ones(n_half)
        x, val = _optimize_phases_flat(mag, max_iter=120)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        candidates.append(x)
    
    # 3. Tapered magnitude spectra
    for taper in [0.5, 1.0, 2.0]:
        if time.time() > deadline - 50:
            break
        k = np.arange(n_half)
        mag = 1.0 / (1.0 + (k / n_half) ** taper)
        x, val = _optimize_phases_flat(mag, max_iter=100)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        candidates.append(x)
    
    # 4. Chirp sequences with optimized parameters
    alpha_vals = [0.05, 0.1, 0.15, 0.25, 0.4]
    beta_vals = [0.0, 0.15, 0.3]
    for alpha in alpha_vals:
        if time.time() > deadline - 40:
            break
        for beta in beta_vals:
            if time.time() > deadline - 40:
                break
            try:
                res = minimize(
                    lambda p, a=alpha, b=beta: _objective_smooth(_chirp_sequence(a, b, p[0], p[1]), temp=0.5),
                    [0.0, 0.0],
                    method='L-BFGS-B',
                    options={'maxiter': 25, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                x_chirp = _chirp_sequence(alpha, beta, res.x[0], res.x[1])
                candidates.append(x_chirp)
                val = _metric_offdiag(x_chirp)
                if val < best_val:
                    best_val = val
                    best_x = x_chirp.copy()
            except Exception:
                pass
    
    # 5. Random sequences with non-zero mean
    for _ in range(15):
        if time.time() > deadline - 30:
            break
        r = rng.random(N)
        x = np.where(r > 0.35, 1.0, -1.0)
        if abs(np.sum(x)) > 5:
            candidates.append(x)
    
    # 6. Block patterns
    for block_size in [10, 20, 50]:
        if time.time() > deadline - 20:
            break
        pattern = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                pattern[i] = 1.0
            else:
                pattern[i] = -1.0
        pattern[0] += 0.5
        candidates.append(pattern)
    
    # 7. Sine wave patterns
    for freq in [2, 5, 10]:
        if time.time() > deadline - 15:
            break
        t = np.arange(N)
        pattern = np.sin(2 * np.pi * freq * t / N)
        pattern = np.sign(pattern)
        if abs(np.sum(pattern)) < 1:
            pattern[0] = 1.0
        candidates.append(pattern)
    
    # Add current best
    candidates.append(best_x.copy())
    
    # Score and sort candidates
    scored = []
    for c in candidates:
        val = _metric_offdiag(c)
        scored.append((c, val))
    scored.sort(key=lambda t: t[1])
    
    # Optimize top candidates with two-pass L-BFGS-B
    for x0, cur_val in scored[:12]:
        if time.time() > deadline - 30:
            break
        
        x0 = x0.copy()
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        x0 = x0 / s0
        
        try:
            # First pass: higher temperature
            res1 = minimize(
                lambda x: _objective_smooth(x, temp=1.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_mid = res1.x
            
            # Second pass: lower temperature
            res2 = minimize(
                lambda x: _objective_smooth(x, temp=0.1),
                x_mid,
                method='L-BFGS-B',
                options={'maxiter': 120, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt = res2.x
            
            val = _metric_offdiag(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Final gradient descent polish
    if time.time() < deadline - 5:
        try:
            polished = _gradient_descent_polish(
                best_x,
                max_iter=1200,
                initial_lr=0.005,
                deadline=deadline - 3
            )
            val = _metric_offdiag(polished)
            if val < best_val:
                best_val = val
                best_x = polished
        except Exception:
            pass
    
    return best_x
