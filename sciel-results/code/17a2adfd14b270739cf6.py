import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
_DEADLINE = _START + 135.0  # safety margin under 150s kill

def _eval(h):
    h = np.asarray(h, dtype=float)
    if h.size != N:
        return -np.inf
    if not np.all(np.isfinite(h)):
        return -np.inf
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    n1 = np.sum(g) / (m + 1)
    ninf = np.max(g)
    if n1 * ninf < 1e-30:
        return -np.inf
    return l2 / (n1 * ninf)

def _smooth_eval(h, beta=50.0):
    """Smooth surrogate for the objective using log-sum-exp for norm_inf."""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    n1 = np.sum(g) / (m + 1)
    # Smooth max via log-sum-exp
    if np.max(g) <= 0:
        return -1e10
    log_g = np.log(np.maximum(g, 1e-300))
    log_max = np.max(log_g) + np.log(np.sum(np.exp(log_g - np.max(log_g)))) / beta
    # Actually, log-sum-exp: log(sum(exp(g_i))) / beta approximates max(g_i) as beta->inf
    # But we want max(g), not log-sum-exp of g. Let's use: (1/beta)*log(sum(exp(beta*g)))
    # This is the softmax approximation to max.
    scaled = beta * g
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    ninf_smooth = log_sum_exp / beta
    if n1 * ninf_smooth < 1e-30:
        return -1e10
    return l2 / (n1 * ninf_smooth)

def _neg_smooth_eval(h, beta=50.0):
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    v = _smooth_eval(h, beta)
    if v == -1e10:
        return 1e10
    return -v

def _lbfgs_smooth(h0, beta=50.0, max_iter=1000):
    try:
        bounds = [(0, 20)] * N
        res = minimize(lambda x: _neg_smooth_eval(x, beta), h0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _eval(h)
            if np.isfinite(v):
                return h, v
    except Exception:
        pass
    return h0, _eval(h0)

def _lbfgs_exact(h0, max_iter=500):
    try:
        bounds = [(0, 20)] * N
        res = minimize(lambda x: -_eval(np.maximum(x, 0.0)), h0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _eval(h)
            if np.isfinite(v):
                return h, v
    except Exception:
        pass
    return h0, _eval(h0)

def _cd(h0, budget):
    """Coordinate descent with adaptive step sizes."""
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0.0)
    best = _eval(h)
    step = 0.5
    no_improve_count = 0
    
    while time.time() < budget:
        improved = False
        for i in range(N):
            for scale in [1.0, 0.5, 0.2, 0.1, 0.05]:
                s = step * scale
                for d in (s, -s):
                    cand = h.copy()
                    cand[i] = max(0.0, cand[i] + d)
                    v = _eval(cand)
                    if v > best + 1e-12:
                        h = cand
                        best = v
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            step *= 0.6
            no_improve_count += 1
            if step < 1e-8 or no_improve_count > 50:
                break
        else:
            no_improve_count = 0
    return h, best

def _make_initial_guesses():
    t = np.linspace(-0.5, 0.5, N)
    guesses = []
    
    # Uniform
    guesses.append(np.ones(N))
    
    # Gaussian bumps at center with various widths
    for sigma in [0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]:
        guesses.append(np.exp(-t**2 / (2 * sigma**2)))
    
    # Cosine bumps
    for width in [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]:
        guesses.append(np.maximum(np.cos(np.pi * t / width), 0.0))
    
    # Tapered / triangular
    for a in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]:
        guesses.append(np.maximum(1.0 - a * np.abs(t), 0.0))
        guesses.append(np.maximum(1.0 - a * t**2, 0.0))
    
    # Power law
    for p in [0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]:
        guesses.append(np.maximum(1.0 - (2.0 * np.abs(t))**p, 0.0))
    
    # Exponential
    for a in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]:
        guesses.append(np.exp(-a * np.abs(t)))
        guesses.append(np.exp(-a * t**2))
    
    # Lorentzian
    for c in [0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]:
        guesses.append(1.0 / (1.0 + c * t**2))
    
    # Sine-based
    for a in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]:
        guesses.append(np.maximum(np.sin(np.pi * t) ** a, 0.0))
        guesses.append(np.maximum(np.cos(np.pi * t * 0.5) ** a, 0.0))
    
    # Two-bump shapes
    for d in [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]:
        for s in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25]:
            guesses.append(np.exp(-t**2 / (2 * s**2)) + np.exp(-(t-d)**2 / (2 * s**2)))
            guesses.append(np.exp(-t**2 / (2 * s**2)) + np.exp(-(t+d)**2 / (2 * s**2)))
    
    # Three-bump shapes
    for d in [0.08, 0.1, 0.12, 0.15, 0.2]:
        for s in [0.05, 0.08, 0.1, 0.12, 0.15]:
            guesses.append(np.exp(-t**2 / (2 * s**2)) + np.exp(-(t-d)**2 / (2 * s**2)) + np.exp(-(t+d)**2 / (2 * s**2)))
    
    # Plateau shapes
    for w in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        guesses.append(np.where(np.abs(t) < w, 1.0, 0.0))
        guesses.append(np.where(np.abs(t) < w, 1.0, np.exp(-5.0 * (np.abs(t) - w))))
        guesses.append(np.where(np.abs(t) < w, 1.0, np.exp(-10.0 * (np.abs(t) - w))))
    
    # Smoothstep / sigmoid transitions
    for w in [0.08, 0.12, 0.15, 0.2, 0.25, 0.3]:
        x = np.abs(t) / w
        guesses.append(np.where(x < 1, 1 - 3*x**2/2 + x**3/2, 0.0))
        guesses.append(1.0 / (1.0 + np.exp(10 * (np.abs(t) - w))))
    
    # Bessel-like functions
    for a in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        guesses.append(np.maximum(1.0 - a * t**2 * np.log(1 + a * t**2), 0.0))
    
    # Airy-like (approximate)
    for a in [2.0, 4.0, 6.0, 10.0]:
        guesses.append(np.maximum(1.0 - a * np.abs(t)**1.5, 0.0))
    
    # Random smooth guesses (Fourier basis)
    rng = np.random.RandomState(42)
    for _ in range(40):
        n_harmonics = rng.randint(3, 8)
        coeffs = rng.randn(n_harmonics)
        phases = rng.rand(n_harmonics) * 2 * np.pi
        guess = np.zeros(N)
        for k in range(n_harmonics):
            guess += coeffs[k] * np.cos((k + 1) * np.pi * t + phases[k])
        guess = np.maximum(guess, 0.0)
        if np.max(guess) > 1e-6:
            guess /= np.max(guess)
        guesses.append(guess)
    
    # Random with center emphasis
    for seed_offset in range(30):
        rng2 = np.random.RandomState(42 + seed_offset + 100)
        weights = rng2.rand(N)
        sigma = rng2.uniform(0.1, 0.4)
        center_boost = np.exp(-t**2 / (2 * sigma**2))
        guess = weights * center_boost
        guess = np.maximum(guess, 0.0)
        if np.max(guess) > 1e-6:
            guess /= np.max(guess)
        guesses.append(guess)
    
    # Random smooth (spline-like)
    for seed_offset in range(20):
        rng3 = np.random.RandomState(700 + seed_offset)
        n_ctrl = rng3.randint(4, 8)
        ctrl_t = np.sort(rng3.uniform(-0.5, 0.5, n_ctrl))
        ctrl_v = rng3.uniform(0, 1, n_ctrl)
        guess = np.interp(t, ctrl_t, ctrl_v)
        guess = np.maximum(guess, 0.0)
        if np.max(guess) > 1e-6:
            guess /= np.max(guess)
        guesses.append(guess)
    
    return guesses

def construct():
    # Phase 1: Evaluate all initial guesses
    guesses = _make_initial_guesses()
    candidates = []
    
    for g in guesses:
        g = np.asarray(g, dtype=float)
        if not np.all(np.isfinite(g)):
            continue
        g = np.maximum(g, 0.0)
        if np.max(g) < 1e-12:
            continue
        v = _eval(g)
        if np.isfinite(v):
            candidates.append((v, g))
            
    if not candidates:
        return np.ones(N).tolist()
        
    # Sort by value descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 15 for refinement
    top_k = candidates[:15]
    
    best_h = top_k[0][1].copy()
    best_v = top_k[0][0]
    
    # Phase 2: L-BFGS-B with smooth surrogate from top 15 guesses
    refined = []
    for v, h in top_k:
        if time.time() > _DEADLINE - 80:
            break
        h_new, v_new = _lbfgs_smooth(h, beta=50.0, max_iter=800)
        if v_new > best_v:
            best_v = v_new
            best_h = h_new.copy()
        refined.append((v_new, h_new))
        
    # Sort refined
    refined.sort(key=lambda x: x[0], reverse=True)
    top_k_refined = refined[:8]
    
    # Phase 3: L-BFGS-B with exact objective (higher beta) on top refined
    refined2 = []
    for v, h in top_k_refined:
        if time.time() > _DEADLINE - 50:
            break
        h_new, v_new = _lbfgs_smooth(h, beta=200.0, max_iter=600)
        if v_new > best_v:
            best_v = v_new
            best_h = h_new.copy()
        refined2.append((v_new, h_new))
        
    refined2.sort(key=lambda x: x[0], reverse=True)
    top_k_refined2 = refined2[:5]
    
    # Phase 4: Coordinate Descent on top refined candidates
    for v, h in top_k_refined2:
        if time.time() > _DEADLINE - 30:
            break
        budget = min(time.time() + 8.0, _DEADLINE - 25)
        if time.time() < budget:
            h_new, v_new = _cd(h, budget)
            if v_new > best_v:
                best_v = v_new
                best_h = h_new.copy()
    
    # Phase 5: Another round of L-BFGS from the best result with very high beta
    if time.time() < _DEADLINE - 15:
        h_new, v_new = _lbfgs_smooth(best_h, beta=500.0, max_iter=1000)
        if v_new > best_v:
            best_v = v_new
            best_h = h_new.copy()
    
    # Phase 6: Final high-precision Coordinate Descent
    budget = min(time.time() + 12.0, _DEADLINE - 2.0)
    if time.time() < budget:
        h_new, v_new = _cd(best_h, budget)
        if v_new > best_v:
            best_v = v_new
            best_h = h_new.copy()
    
    # Final sanity check
    final = np.maximum(best_h, 0.0)
    final = np.where(np.isfinite(final), final, 0.0)
    if not np.all(np.isfinite(final)) or np.any(final < 0):
        final = np.ones(N)
    
    return final.tolist()
