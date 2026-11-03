"""Riemannian optimization on the unit sphere: minimize max|ac|/S^2 via sphere-constrained L-BFGS."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _fft_len(n):
    L = 1
    while L < n:
        L *= 2
    return L

def _autoconv_fft(x):
    L = 2 * N - 1
    L_fft = _fft_len(L)
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x, beta=1e6):
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

def _sphere_objective_and_grad(x, beta=50.0):
    """Objective and gradient on the unit sphere (x already normalized)."""
    nrm = np.linalg.norm(x)
    if nrm < 1e-10:
        return 1e10, np.zeros(N)
    x = x / nrm
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    val = soft_val / (s * s)
    
    # Gradient computation
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    g_ac = weights * signs  # length 2N-1
    
    # Compute gradient via FFT cross-correlation
    L_c = _fft_len(2 * N - 1)
    G_f2 = np.fft.rfft(g_ac, n=L_c)
    X_f2 = np.fft.rfft(x, n=L_c)
    c = np.fft.irfft(G_f2 * np.conj(X_f2), n=L_c)
    grad_soft = 2.0 * c[:N]
    
    # grad of val = soft_val/s^2
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    
    # Project gradient onto tangent space of sphere
    proj = grad - np.dot(grad, x) * x
    return val, proj

def _riemannian_lbgfs(x0, beta=50.0, max_iter=200):
    """Riemannian L-BFGS on the unit sphere."""
    x = x0 / np.linalg.norm(x0)
    
    def func_and_grad(x_flat):
        return _sphere_objective_and_grad(x_flat, beta)
    
    result = minimize(func_and_grad, x, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-14, 'maxls': 30})
    
    x_out = result.x
    nrm = np.linalg.norm(x_out)
    if nrm > 1e-10:
        x_out = x_out / nrm
    return x_out, result.fun

def _peak_targeting(x, n_rounds=50, beta=100.0):
    """Directly target the dominant autocorrelation lags."""
    x = x.copy()
    nrm = np.linalg.norm(x)
    if nrm < 1e-10:
        return x, _objective_only(x)
    x = x / nrm
    best_val = _objective_only(x)
    rng = np.random.default_rng(77)
    
    for r in range(n_rounds):
        if _check_deadline():
            break
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        cur_val = np.max(abs_ac) / (s * s)
        
        # Find top lag
        peak_idx = np.argmax(abs_ac)
        
        improved = False
        for attempt in range(30):
            if _check_deadline():
                break
            # Random perturbation
            n_pert = rng.integers(5, 20)
            idx = rng.choice(N, size=n_pert, replace=False)
            deltas = rng.normal(0, 0.03, size=n_pert)
            x_trial = x.copy()
            x_trial[idx] += deltas
            nrm_t = np.linalg.norm(x_trial)
            if nrm_t < 1e-10:
                continue
            x_trial = x_trial / nrm_t
            val = _objective_only(x_trial)
            if val < best_val * (1 - 1e-7):
                best_val = val
                x = x_trial
                improved = True
                break
        
        if not improved:
            # Larger perturbation
            for attempt in range(15):
                if _check_deadline():
                    break
                n_pert = rng.integers(10, 40)
                idx = rng.choice(N, size=n_pert, replace=False)
                deltas = rng.normal(0, 0.1, size=n_pert)
                x_trial = x.copy()
                x_trial[idx] += deltas
                nrm_t = np.linalg.norm(x_trial)
                if nrm_t < 1e-10:
                    continue
                x_trial = x_trial / nrm_t
                val = _objective_only(x_trial)
                if val < best_val * (1 - 1e-7):
                    best_val = val
                    x = x_trial
                    improved = True
                    break
        if not improved:
            break
    
    return x, best_val

def _coordinate_descent_sphere(x, max_rounds=100):
    """Coordinate descent on the sphere (renormalize after each step)."""
    x = x.copy()
    nrm = np.linalg.norm(x)
    if nrm < 1e-10:
        return x, _objective_only(x)
    x = x / nrm
    best_val = _objective_only(x)
    rng = np.random.default_rng(55)
    
    for r in range(max_rounds):
        if _check_deadline():
            break
        improved = False
        for ss in [0.02, 0.01, 0.005]:
            if _check_deadline():
                break
            for j in range(N):
                if _check_deadline():
                    break
                for sign in [1, -1]:
                    x_trial = x.copy()
                    x_trial[j] += sign * ss
                    nrm_t = np.linalg.norm(x_trial)
                    if nrm_t < 1e-10:
                        continue
                    x_trial = x_trial / nrm_t
                    val = _objective_only(x_trial)
                    if val < best_val * (1 - 1e-8):
                        best_val = val
                        x = x_trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return x, best_val

def _make_initializations(rng, n_structured=12, n_random=15):
    """Generate diverse initializations on the unit sphere."""
    t = np.linspace(0, 1, N)
    inits = []
    
    structured = [
        np.ones(N),
        1.0 + 0.5 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.5 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        1.0 + 0.3 * np.sin(8 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        1.0 + 0.8 * (t - 0.5),
        np.maximum(0.3, 1.0 - 2.0 * np.abs(t - 0.5)),
        1.0 + 0.5 * np.sin(2 * np.pi * 3 * t),
        1.0 + 0.5 * np.cos(2 * np.pi * 5 * t),
    ]
    for s in structured[:n_structured]:
        nrm = np.linalg.norm(s)
        if nrm > 1e-10:
            inits.append(s / nrm)
    
    for _ in range(n_random):
        # Random with varying sparsity and structure
        scale = rng.uniform(0.5, 2.0)
        x_rand = scale * rng.normal(1.0, 0.3, size=N)
        # Occasionally make it sparse
        if rng.random() < 0.3:
            mask = rng.random(N) < 0.5
            x_rand = np.where(mask, x_rand, 0.0)
        nrm = np.linalg.norm(x_rand)
        if nrm > 1e-10:
            inits.append(x_rand / nrm)
    
    return inits

def _spectral_phase_init(rng, n_freqs=20):
    """Generate initialization from random spectral phase."""
    # Random amplitudes concentrated at low frequencies
    freqs = np.arange(N // 2 + 1)
    amplitudes = np.exp(-freqs / (N // 4)) * rng.uniform(0.5, 1.5, size=N // 2 + 1)
    phases = rng.uniform(0, 2 * np.pi, size=N // 2 + 1)
    
    # Reconstruct signal from spectral representation
    X = amplitudes * np.exp(1j * phases)
    x = np.fft.irfft(X, n=N)
    
    # Add small random noise to break symmetry
    x += rng.normal(0, 0.01, size=N)
    
    nrm = np.linalg.norm(x)
    if nrm > 1e-10:
        x = x / nrm
    return x

def _spectral_phase_optimize(x0, max_iter=100, beta=50.0):
    """Optimize spectral phase while keeping amplitudes fixed."""
    x = x0.copy()
    nrm = np.linalg.norm(x)
    if nrm < 1e-10:
        return x, _objective_only(x)
    x = x / nrm
    
    best_val = _objective_only(x)
    rng = np.random.default_rng(99)
    
    # Get initial spectral representation
    X = np.fft.rfft(x, n=N)
    amplitudes = np.abs(X)
    
    for it in range(max_iter):
        if _check_deadline():
            break
        
        # Random phase perturbation
        phase_pert = rng.normal(0, 0.1, size=len(amplitudes))
        phases = np.angle(X) + phase_pert
        X_trial = amplitudes * np.exp(1j * phases)
        x_trial = np.fft.irfft(X_trial, n=N)
        
        nrm_t = np.linalg.norm(x_trial)
        if nrm_t < 1e-10:
            continue
        x_trial = x_trial / nrm_t
        
        val = _objective_only(x_trial)
        if val < best_val * (1 - 1e-7):
            best_val = val
            x = x_trial
            X = X_trial
    
    return x, best_val

def _amplitude_optimize(x0, max_iter=100, beta=50.0):
    """Optimize spectral amplitudes while keeping phases fixed."""
    x = x0.copy()
    nrm = np.linalg.norm(x)
    if nrm < 1e-10:
        return x, _objective_only(x)
    x = x / nrm
    
    best_val = _objective_only(x)
    rng = np.random.default_rng(88)
    
    # Get initial spectral representation
    X = np.fft.rfft(x, n=N)
    phases = np.angle(X)
    
    for it in range(max_iter):
        if _check_deadline():
            break
        
        # Random amplitude perturbation
        amplitudes = np.abs(X)
        amp_pert = rng.normal(0, 0.05, size=len(amplitudes))
        amplitudes_trial = np.maximum(0, amplitudes + amp_pert)
        X_trial = amplitudes_trial * np.exp(1j * phases)
        x_trial = np.fft.irfft(X_trial, n=N)
        
        nrm_t = np.linalg.norm(x_trial)
        if nrm_t < 1e-10:
            continue
        x_trial = x_trial / nrm_t
        
        val = _objective_only(x_trial)
        if val < best_val * (1 - 1e-7):
            best_val = val
            x = x_trial
            X = X_trial
    
    return x, best_val

def construct():
    best_x = np.ones(N)
    best_x = best_x / np.linalg.norm(best_x)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(12345)
    inits = _make_initializations(rng)
    
    # Add spectral phase initializations
    for _ in range(8):
        x_spec = _spectral_phase_init(rng)
        inits.append(x_spec)
    
    # Phase 1: Riemannian L-BFGS from each initialization
    for x0 in inits:
        if _check_deadline():
            break
        x_opt, val_opt = _riemannian_lbgfs(x0, beta=50.0, max_iter=150)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Peak targeting
    best_x, best_val = _peak_targeting(best_x, n_rounds=60, beta=100.0)
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Coordinate descent on sphere
    best_x, best_val = _coordinate_descent_sphere(best_x, max_rounds=80)
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Spectral phase optimization
    x_spec, val_spec = _spectral_phase_optimize(best_x, max_iter=80, beta=50.0)
    if val_spec < best_val:
        best_val = val_spec
        best_x = x_spec.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Amplitude optimization
    x_amp, val_amp = _amplitude_optimize(best_x, max_iter=80, beta=50.0)
    if val_amp < best_val:
        best_val = val_amp
        best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Another round of Riemannian L-BFGS from the polished point
    x_opt, val_opt = _riemannian_lbgfs(best_x, beta=100.0, max_iter=100)
    if val_opt < best_val:
        best_val = val_opt
        best_x = x_opt.copy()
    
    # Phase 7: Final peak targeting
    best_x, best_val = _peak_targeting(best_x, n_rounds=30, beta=150.0)
    
    return best_x
# EVOLVE-BLOCK-END
