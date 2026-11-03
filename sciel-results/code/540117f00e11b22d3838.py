"""Model-and-optimize: peak-targeted polish + multi-restart L-BFGS + spectral polish + coordinate descent + high-precision final polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 148

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 2)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _sign_flip_cd(x0, rng, max_iters=5000):
    """
    Coordinate descent with sign flips: for each coordinate, try flipping its sign
    (or negating it) and keep the flip if it improves the objective.
    Exploits the symmetry: if x is a solution, -x is also a solution.
    Also tries partial flips (multiply by -alpha) for smoother transitions.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    no_improve = 0
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2

    for it in range(max_iters):
        if _check_deadline():
            break

        improved_this_round = False
        # Cycle through coordinates in random order
        order = rng.permutation(N)
        for idx in order:
            if _check_deadline():
                return best_x, best_val

            # Try full sign flip
            x_trial = best_x.copy()
            x_trial[idx] = -x_trial[idx]
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial
                improved_this_round = True
                continue

            # Try partial flip: multiply by -0.5
            x_trial = best_x.copy()
            x_trial[idx] = -0.5 * x_trial[idx]
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial
                improved_this_round = True

        if improved_this_round:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                # Try random multi-coordinate flips
                n_flip = rng.integers(2, 8)
                indices = rng.choice(N, size=n_flip, replace=False)
                x_trial = best_x.copy()
                x_trial[indices] *= -1.0
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial
                    no_improve = 0
                else:
                    # Try random sign perturbation
                    x_trial = best_x.copy()
                    x_trial[indices] *= -0.7 + 0.4 * rng.random(n_flip)
                    val = _objective_only(x_trial)
                    if val < best_val:
                        best_val = val
                        best_x = x_trial
                        no_improve = 0
                    else:
                        if no_improve > 200:
                            break

    return best_x, best_val

def _peak_targeted_polish(x0, rng, max_iters=2000, step_size=0.05):
    """
    Vectorized peak-targeted polish: computes the gradient of max|ac| w.r.t. x
    and takes steepest-descent steps toward reducing the dominant lags.
    Much faster than per-lag Python loops.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    current_step = step_size
    no_improve = 0
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2

    for it in range(max_iters):
        if _check_deadline():
            break

        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        s = np.sum(best_x)
        s2 = s * s
        current_obj = np.max(abs_ac) / s2

        # Adaptive beta: increase as we get closer to convergence
        m = np.max(abs_ac)
        beta = min(200.0, 50.0 + 150.0 * it / max(1, max_iters))
        exps = np.exp(beta * (abs_ac - m))
        weights = exps / np.sum(exps)
        signs = np.sign(ac)
        signs[ac == 0] = 1.0
        grad_ac = weights * signs

        # Gradient of soft-max w.r.t. x via FFT cross-correlation
        g_padded = np.zeros(L_fft)
        g_padded[:L] = grad_ac
        X_f = np.fft.rfft(best_x, n=L_fft)
        G_f = np.fft.rfft(g_padded, n=L_fft)
        corr = np.fft.irfft(G_f * np.conj(X_f), n=L_fft)
        grad_x = 2.0 * corr[:N]

        # Full gradient of objective = max|ac|/s^2
        soft_val = m + np.log(np.sum(exps)) / beta
        grad_obj = grad_x / s2 - 2.0 * soft_val * np.ones(N) / (s2 * s)

        # Line search with backtracking
        step = current_step
        improved = False
        for _ in range(10):
            x_trial = best_x - step * grad_obj
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial
                improved = True
                break
            step *= 0.5

        if improved:
            no_improve = 0
            current_step = min(0.15, current_step * 1.03)
        else:
            no_improve += 1
            current_step = max(0.002, current_step * 0.9)

            if no_improve > 150:
                # Global perturbation escape with larger step
                x_trial = best_x + rng.normal(0, 0.05, size=N)
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial
                    no_improve = 0
                    current_step = 0.08

    return best_x, best_val

def _spectral_init(rng, mode=0):
    """Generate initial x via spectral phase retrieval."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(L_fft // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_fft // 2 + 1)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    elif mode == 5:
        # New: bandpass filter
        freqs = np.fft.rfftfreq(L_fft)
        low, high = 0.1, 0.6
        mag = np.where((np.abs(freqs) > low) & (np.abs(freqs) < high), 1.0, 0.05)
    elif mode == 6:
        # New: geometric decay
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-5.0 * np.abs(freqs) * L_fft)
    else:
        mag = np.ones(L_fft // 2 + 1)
    
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _spectral_phase_optimize(x0, rng, max_iters=150, lr=0.01):
    """Optimize the phase of the spectrum while keeping magnitude fixed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    X = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_and_grad(best_x, 50.0)
        G = np.fft.rfft(grad, n=L_fft)
        
        X_curr = mag * np.exp(1j * phase)
        phase_grad = np.imag(X_curr * np.conj(G)) / (mag + 1e-10)
        phase_grad[0] = 0
        
        phase -= lr * phase_grad
        
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
            phase = -phase
            phase[0] = 0
        
        val_trial = _objective_only(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            if it % 20 == 0 and it > 0:
                lr = max(0.001, lr * 0.95)
        else:
            phase += lr * phase_grad
            lr = max(0.001, lr * 0.8)
    
    return best_x, best_val

def _amplitude_optimize(x0, rng, iters=30):
    """Optimize the amplitude spectrum while keeping phase fixed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    phase = np.angle(X)
    phase[0] = 0
    amp = np.abs(X).copy()
    
    best_amp = amp.copy()
    best_val = _objective_only(x0)
    
    for it in range(iters):
        if _check_deadline():
            break
        
        amp_trial = amp * (1 + 0.05 * rng.normal(0, 1, size=n_freq))
        amp_trial = np.maximum(0.01, amp_trial)
        
        spectrum = amp_trial * np.exp(1j * phase)
        spectrum[0] = amp_trial[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            amp = amp_trial.copy()
            best_amp = amp.copy()
    
    spectrum = best_amp * np.exp(1j * phase)
    spectrum[0] = best_amp[0]
    x_best = np.fft.irfft(spectrum, n=L_fft)[:N]
    s = np.sum(x_best)
    if s < 0:
        x_best = -x_best
    
    return x_best, best_val

def _gradient_guided_descent(x0, rng, max_iters=15000, initial_step=0.03):
    """
    Hybrid gradient-guided descent: uses gradient direction when available,
    falls back to structured random perturbations. More efficient than pure random.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    step_size = initial_step
    no_improve = 0
    t_arr = np.arange(N)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        r = rng.random()
        
        if r < 0.4:
            # Gradient-guided step with adaptive beta
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
            s = np.sum(best_x)
            s2 = s * s
            m = np.max(abs_ac)
            beta = min(150.0, 50.0 + 100.0 * no_improve / 50.0)
            exps = np.exp(beta * (abs_ac - m))
            weights = exps / np.sum(exps)
            signs = np.sign(ac)
            signs[ac == 0] = 1.0
            grad_ac = weights * signs
            
            L_fft = 1
            while L_fft < 2 * N:
                L_fft *= 2
            g_padded = np.zeros(L_fft)
            g_padded[:2*N-1] = grad_ac
            X_f = np.fft.rfft(best_x, n=L_fft)
            G_f = np.fft.rfft(g_padded, n=L_fft)
            corr = np.fft.irfft(G_f * np.conj(X_f), n=L_fft)
            grad_x = 2.0 * corr[:N]
            
            soft_val = m + np.log(np.sum(exps)) / beta
            grad_obj = grad_x / s2 - 2.0 * soft_val * np.ones(N) / (s2 * s)
            
            # Normalize gradient for stable steps
            grad_norm = np.linalg.norm(grad_obj)
            if grad_norm > 1e-10:
                x_trial = best_x - step_size * grad_obj / grad_norm
            else:
                x_trial = best_x + rng.normal(0, step_size, size=N)
                
        elif r < 0.55:
            # Small random perturbation
            x_trial = best_x + rng.normal(0, step_size, size=N)
        elif r < 0.7:
            # Sparse perturbation
            n_perturb = rng.integers(5, 20)
            indices = rng.choice(N, size=n_perturb, replace=False)
            x_trial = best_x.copy()
            x_trial[indices] += rng.normal(0, step_size * 2, size=n_perturb)
        elif r < 0.85:
            # Low-frequency sinusoidal
            freq = rng.integers(1, 8)
            phase = rng.uniform(0, 2 * np.pi)
            x_trial = best_x + step_size * 1.5 * np.sin(2 * np.pi * freq * t_arr / N + phase)
        else:
            # Damped sign-flip for escaping local minima
            n_flip = rng.integers(5, 20)
            indices = rng.choice(N, size=n_flip, replace=False)
            x_trial = best_x.copy()
            x_trial[indices] *= -0.5 + 0.5 * rng.random(n_flip)
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial
            step_size = max(0.001, step_size * 0.97)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.08, step_size * 1.01)
        
        if no_improve > 3000 and rng.random() < 0.02:
            x_restart = best_x + rng.normal(0, 0.08, size=N)
            val_restart = _objective_only(x_restart)
            if val_restart < best_val:
                best_val = val_restart
                best_x = x_restart
                no_improve = 0
                step_size = 0.03
    
    return best_x, best_val

def _high_precision_polish(x0, rng, max_iters=500, beta_start=500.0):
    """
    Final high-precision polish using very high beta soft-max and tight tolerances.
    This targets the exact max|ac| rather than a smooth approximation.
    Uses adaptive step sizing and tracks the dominant lag for targeted reduction.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    current_step = 0.005
    no_improve = 0
    grad_norm_history = []

    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        s = np.sum(best_x)
        s2 = s * s
        
        # Very high beta for near-exact max
        m = np.max(abs_ac)
        beta = min(3000.0, beta_start + 200.0 * it)
        exps = np.exp(beta * (abs_ac - m))
        weights = exps / np.sum(exps)
        signs = np.sign(ac)
        signs[ac == 0] = 1.0
        grad_ac = weights * signs
        
        g_padded = np.zeros(L_fft)
        g_padded[:L] = grad_ac
        X_f = np.fft.rfft(best_x, n=L_fft)
        G_f = np.fft.rfft(g_padded, n=L_fft)
        corr = np.fft.irfft(G_f * np.conj(X_f), n=L_fft)
        grad_x = 2.0 * corr[:N]
        
        soft_val = m + np.log(np.sum(exps)) / beta
        grad_obj = grad_x / s2 - 2.0 * soft_val * np.ones(N) / (s2 * s)
        
        grad_norm = np.linalg.norm(grad_obj)
        grad_norm_history.append(grad_norm)
        
        # Adaptive step: scale by gradient norm for consistent displacement
        adaptive_step = current_step / max(grad_norm, 1e-10)
        
        improved = False
        step = adaptive_step
        for _ in range(20):
            x_trial = best_x - step * grad_obj
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial
                improved = True
                break
            step *= 0.4
        
        if improved:
            no_improve = 0
            current_step = min(0.02, current_step * 1.05)
        else:
            no_improve += 1
            current_step = max(0.0005, current_step * 0.85)
            
            if no_improve > 30:
                # Try targeted perturbation on the dominant lag's contributing coordinates
                dominant_lag = np.argmax(abs_ac)
                # Perturb coordinates that most affect this lag
                n_perturb = min(10, N)
                indices = rng.choice(N, size=n_perturb, replace=False)
                x_trial = best_x.copy()
                x_trial[indices] += rng.normal(0, 0.002, size=n_perturb)
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial
                    no_improve = 0
                    current_step = 0.008
                elif no_improve > 100:
                    # Larger escape
                    x_trial = best_x + rng.normal(0, 0.005, size=N)
                    val = _objective_only(x_trial)
                    if val < best_val:
                        best_val = val
                        best_x = x_trial
                        no_improve = 0
                        current_step = 0.01
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        1.0 + 0.4 * np.sin(8 * np.pi * t),
        np.maximum(0.3, 1.0 + 0.6 * np.sin(2 * np.pi * 3 * t)),
        1.0 + 0.3 * np.cos(6 * np.pi * t) * np.sin(2 * np.pi * t),
        np.maximum(0.2, 1.0 + 0.7 * np.sin(2 * np.pi * 5 * t)),
    ]
    
    for mode in range(7):
        for _ in range(2):
            initial_points.append(_spectral_init(rng, mode))
    
    for _ in range(10):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # Phase 1: Multi-restart L-BFGS with adaptive beta
    betas = [40.0, 80.0, 150.0]
    
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        for beta in betas:
            if _check_deadline():
                break
            
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 120, 'gtol': 1e-8, 'ftol': 1e-10, 'maxls': 30})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, rng, max_iters=2500, step_size=0.06)
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Spectral phase optimization
    x_phase, val_phase = _spectral_phase_optimize(best_x, rng, max_iters=100, lr=0.012)
    if val_phase < best_val:
        best_val = val_phase
        best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Amplitude optimization
    for _ in range(3):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=25)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: High-beta L-BFGS polish
    for beta in [300.0, 600.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 100, 'gtol': 1e-9, 'ftol': 1e-11, 'maxls': 25})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5b: Sign-flip coordinate descent
    best_x, best_val = _sign_flip_cd(best_x, rng, max_iters=3000)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Peak-targeted polish again
    best_x, best_val = _peak_targeted_polish(best_x, rng, max_iters=1500, step_size=0.03)
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Gradient-guided descent
    best_x, best_val = _gradient_guided_descent(best_x, rng, max_iters=20000, initial_step=0.025)
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Final peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, rng, max_iters=800, step_size=0.02)
    
    if _check_deadline():
        return best_x
    
    # Phase 8b: Another sign-flip pass after gradient descent
    best_x, best_val = _sign_flip_cd(best_x, rng, max_iters=2000)
    
    if _check_deadline():
        return best_x
    
    # Phase 9: High-precision final polish
    best_x, best_val = _high_precision_polish(best_x, rng, max_iters=400, beta_start=500.0)
    
    if _check_deadline():
        return best_x
    
    # Phase 10: Final L-BFGS with very high beta
    for beta in [800.0, 1200.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 60, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 20})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    # Phase 11: Final sign-flip + high-precision polish
    if not _check_deadline():
        best_x, best_val = _sign_flip_cd(best_x, rng, max_iters=1000)
    
    if not _check_deadline():
        best_x, best_val = _high_precision_polish(best_x, rng, max_iters=200, beta_start=1000.0)
    
    return best_x
# EVOLVE-BLOCK-END
