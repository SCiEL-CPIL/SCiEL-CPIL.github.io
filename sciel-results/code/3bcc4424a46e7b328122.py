"""Model-and-optimize: spectral phase retrieval + L-BFGS + coordinate descent + peak-targeted polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x, beta=1e6):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if beta < 1e6:
        exps = np.exp(beta * (abs_ac - m))
        log_sum = np.log(np.sum(exps))
        soft_val = m + log_sum / beta
    else:
        soft_val = m
    return soft_val / (s * s)

def _spectral_phase_retrieval(x0, n_iters=200, rng=None):
    """
    Spectral phase retrieval: fix amplitude spectrum, optimize phases
    to minimize peak autoconvolution.
    
    This is a Fienup-style algorithm applied to the autocorrelation problem.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    L_fft = 1
    while L_fft < 2*N-1:
        L_fft *= 2
    
    # Get initial spectrum
    X = np.fft.rfft(x0, n=L_fft)
    # Fix amplitudes
    amps = np.abs(X).copy()
    amps[0] = 0  # DC component handled separately
    
    # Initialize phases randomly
    phases = rng.uniform(0, 2*np.pi, size=len(amps))
    
    best_val = 1e10
    best_x = x0.copy()
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        # Reconstruct with current phases
        X_cur = amps * np.exp(1j * phases)
        x_cur = np.fft.irfft(X_cur, n=L_fft)[:N]
        
        # Compute autoconvolution
        ac = _autoconv_fft(x_cur)
        s = np.sum(x_cur)
        if abs(s) < 1e-10:
            phases = rng.uniform(0, 2*np.pi, size=len(amps))
            continue
        val = np.max(np.abs(ac)) / (s * s)
        
        if val < best_val:
            best_val = val
            best_x = x_cur.copy()
        
        # Find peak lags
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        
        # Fienup-style update: modify phases to reduce peak
        # Error signal: ac - target (target = 0 for flat autocorrelation)
        error = ac
        # Transform error to frequency domain
        E = np.fft.rfft(error, n=L_fft)
        
        # Phase update: shift phases opposite to error direction
        X_m = amps * np.exp(1j * phases)
        with np.errstate(divide='ignore', invalid='ignore'):
            phase_shift = -np.imag(X_m * np.conj(E[:len(amps)])) / (amps**2 + 1e-10)
        
        # Apply step size
        step_size = 0.5 / (1 + it * 0.01)
        phases = (phases + step_size * phase_shift) % (2 * np.pi)
        
        # Occasionally randomize to escape local minima
        if it % 20 == 19 and it > 0:
            # Perturb a random subset of phases
            n_perturb = max(1, len(amps) // 10)
            idx = rng.choice(len(amps), size=n_perturb, replace=False)
            phases[idx] += rng.uniform(-0.5, 0.5, size=n_perturb)
    
    return best_x, best_val

def _peak_targeted_polish(x, n_rounds=100, rng=None):
    """
    Targeted polish: identify dominant autoconvolution lags and apply
    small perturbations to reduce them.
    """
    if rng is None:
        rng = np.random.default_rng(99)
    
    x = x.copy()
    best_val = _objective_only(x)
    
    for round_idx in range(n_rounds):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        current_val = np.max(abs_ac) / (s * s)
        
        # Find top k peak lags
        k = min(5, len(abs_ac))
        top_lags = np.argsort(abs_ac)[-k:][::-1]
        
        improved = False
        for lag in top_lags:
            if _check_deadline():
                break
            
            # For each peak lag, try to reduce it by adjusting x
            for _ in range(20):
                if _check_deadline():
                    break
                
                # Pick a few indices to perturb
                n_perturb = rng.integers(3, 10)
                idx = rng.choice(N, size=n_perturb, replace=False)
                deltas = rng.normal(0, 0.05, size=n_perturb)
                
                x_trial = x.copy()
                x_trial[idx] += deltas
                
                val = _objective_only(x_trial)
                if val < best_val * (1 - 1e-6):
                    best_val = val
                    x = x_trial
                    improved = True
                    break
        
        if not improved:
            # Try larger perturbations
            for _ in range(10):
                if _check_deadline():
                    break
                
                n_perturb = rng.integers(5, 20)
                idx = rng.choice(N, size=n_perturb, replace=False)
                deltas = rng.normal(0, 0.1, size=n_perturb)
                
                x_trial = x.copy()
                x_trial[idx] += deltas
                
                val = _objective_only(x_trial)
                if val < best_val * (1 - 1e-6):
                    best_val = val
                    x = x_trial
                    improved = True
                    break
        
        if not improved:
            break
    
    return x, best_val

def _coordinate_descent(x, beta, max_rounds=200):
    """Coordinate descent refinement."""
    x = x.copy()
    best_val = _objective_only(x, beta)
    
    for round_idx in range(max_rounds):
        if _check_deadline():
            break
        
        improved = False
        for ss in [0.05, 0.02, 0.01]:
            if _check_deadline():
                break
            
            batch_size = 50
            for batch_start in range(0, N, batch_size):
                if _check_deadline():
                    break
                
                batch_end = min(batch_start + batch_size, N)
                for j in range(batch_start, batch_end):
                    if _check_deadline():
                        break
                    
                    for sign in [1, -1]:
                        x_trial = x.copy()
                        x_trial[j] += sign * ss
                        val = _objective_only(x_trial, beta)
                        if val < best_val:
                            best_val = val
                            x = x_trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Spectral phase retrieval from multiple initializations
    t = np.linspace(0, 1, N)
    
    # Structured initializations
    structured_inits = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.0 * t),
        np.exp(-1.0 * t),
        1.0 + 0.5 * (t - 0.5),
        np.maximum(0.3, 1.0 - 2.0 * np.abs(t - 0.5)),
    ]
    
    # Add random initializations
    for _ in range(10):
        scale = rng.uniform(0.5, 1.5)
        structured_inits.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # Run spectral phase retrieval from each init
    for x0 in structured_inits:
        if _check_deadline():
            break
        
        x_sp, val_sp = _spectral_phase_retrieval(x0, n_iters=150, rng=rng)
        if val_sp < best_val:
            best_val = val_sp
            best_x = x_sp.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS refinement from best spectral result
    try:
        def obj_and_grad_func(x):
            ac = _autoconv_fft(x)
            s = np.sum(x)
            if abs(s) < 1e-10:
                return 1e10, np.zeros(N)
            abs_ac = np.abs(ac)
            m = np.max(abs_ac)
            beta = 100.0
            exps = np.exp(beta * (abs_ac - m))
            log_sum = np.log(np.sum(exps))
            soft_val = m + log_sum / beta
            val = soft_val / (s * s)
            
            # Gradient
            weights = exps / np.sum(exps)
            signs = np.sign(ac)
            signs[ac == 0] = 1.0
            grad_ac = weights * signs
            
            g = grad_ac  # length 2N-1
            cumsum_g = np.concatenate([[0], np.cumsum(g)])
            idx_end = np.minimum(np.arange(N) + N - 1, 2*N-2) + 1
            idx_start = np.arange(N)
            window_sums = cumsum_g[idx_end] - cumsum_g[idx_start]
            term2 = x * window_sums
            
            L_corr = 1
            while L_corr < 3 * N:
                L_corr *= 2
            X_f = np.fft.rfft(x, n=L_corr)
            G_f = np.fft.rfft(g, n=L_corr)
            corr = np.fft.irfft(X_f * np.conj(G_f), n=L_corr)
            term1 = corr[:N]
            
            grad_x = term1 + term2
            grad = grad_x / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
            return val, grad
        
        result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                          jac=True, options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10})
        
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
            
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, n_rounds=80, rng=rng)
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Coordinate descent refinement
    best_x, best_val = _coordinate_descent(best_x, beta=100.0, max_rounds=150)
    
    return best_x
# EVOLVE-BLOCK-END
