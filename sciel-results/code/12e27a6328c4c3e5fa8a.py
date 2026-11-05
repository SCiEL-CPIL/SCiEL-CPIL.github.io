import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.fftpack import fft, ifft

N = 400

def _compute_value(h):
    """Compute the exact evaluation value."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s ** 2))

def _chirp_harmonics(params):
    """
    Construct h from parameters:
    params = [phi0, alpha, beta, h1_amp, h1_phase, h2_amp, h2_phase, ...]
    Chirp: cos(phi0 + 0.5*alpha*n^2 + beta*n)
    Plus DFT harmonics at k=1,2,...
    """
    n = np.arange(N)
    h = np.cos(params[0] + 0.5 * params[1] * n**2 + params[2] * n)
    # Add harmonics starting from index 3
    for i in range(3, len(params), 2):
        amp = params[i]
        phase = params[i+1]
        k = (i - 3) // 2 + 1
        h += amp * np.cos(2.0 * np.pi * k * n / N + phase)
    return h

def _objective_params(params):
    h = _chirp_harmonics(params)
    return _compute_value(h)

def _nm_polish(params, deadline):
    """Polish parameters using Nelder-Mead."""
    try:
        res = minimize(
            _objective_params,
            params,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.fun < _objective_params(params):
            return res.x
    except Exception:
        pass
    return params

def _zadoff_chu_sequence(N, root=1):
    """Generate a Zadoff-Chu sequence."""
    n = np.arange(N)
    # Zadoff-Chu: exp(-j * pi * root * n * (n+1) / N) for even N, adjust for odd
    # Standard ZC: exp(-j * pi * u * n * (n+1) / N)
    # For autocorrelation, we want real parts or complex? We need real h.
    # Let's use the real part of the complex ZC sequence, or just the cosine form.
    # Actually, ZC sequences are complex. We need real step heights.
    # Let's use cos(pi * u * n * (n+1) / N)
    # This is a chirp with specific parameters.
    # alpha = pi * u / N, beta = pi * u / (2N)
    # Let's just generate a few chirps with specific roots.
    pass

def _generate_zc_candidate(root):
    n = np.arange(N)
    # Real part of ZC: cos(pi * root * n * (n+1) / N)
    # This is cos(pi*root/N * n^2 + pi*root/N * n)
    # So alpha = pi*root/N, beta = pi*root/N, phi0 = 0
    # But we can also add harmonics.
    # Let's just return the base chirp parameters.
    alpha = np.pi * root / N
    beta = np.pi * root / N
    phi0 = 0.0
    return [phi0, alpha, beta, 0.0, 0.0, 0.0, 0.0] # 7 params: chirp + 2 harmonics (amp=0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Define parameter bounds
    # phi0: [0, 2pi]
    # alpha: [-5, 5]
    # beta: [-5, 5]
    # h1_amp: [0, 1]
    # h1_phase: [0, 2pi]
    # h2_amp: [0, 1]
    # h2_phase: [0, 2pi]
    
    bounds = [
        (0, 2*np.pi),
        (-5, 5),
        (-5, 5),
        (0, 1),
        (0, 2*np.pi),
        (0, 1),
        (0, 2*np.pi)
    ]

    best_params = None
    best_val = np.inf
    best_h = np.ones(N)

    # Phase 1: Differential Evolution on parametric space
    # Use a smaller population and iterations to fit within time
    try:
        de_res = differential_evolution(
            _objective_params,
            bounds,
            maxiter=30,
            popsize=15,
            seed=42,
            tol=1e-5,
            polish=False,
            updating='immediate'
        )
        if de_res.fun < best_val:
            best_val = de_res.fun
            best_params = de_res.x
            best_h = _chirp_harmonics(best_params)
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 60:
        # If we are running out of time, skip further polishing and random starts
        if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
            return np.ones(N)
        return best_h

    # Phase 2: Nelder-Mead Polish on the best DE result
    if best_params is not None:
        polished_params = _nm_polish(best_params, deadline)
        polished_val = _objective_params(polished_params)
        if polished_val < best_val:
            best_val = polished_val
            best_params = polished_params
            best_h = _chirp_harmonics(best_params)

    # Phase 3: Try Zadoff-Chu inspired initializations
    # ZC sequences correspond to specific chirp parameters.
    # Let's try a few roots.
    for root in [1, 3, 5, 7, 9]:
        if time.time() > deadline - 30:
            break
        zc_params = _generate_zc_candidate(root)
        # Add small random noise to escape exact local minima
        rng = np.random.default_rng(42 + root)
        zc_params_noisy = np.array(zc_params)
        zc_params_noisy[3:] += rng.normal(0, 0.1, size=4) # Perturb harmonics
        
        zc_val = _objective_params(zc_params_noisy)
        if zc_val < best_val:
            best_val = zc_val
            best_params = zc_params_noisy
            best_h = _chirp_harmonics(best_params)
            
            # Polish this new best
            if time.time() < deadline - 20:
                polished_params = _nm_polish(best_params, deadline)
                polished_val = _objective_params(polished_params)
                if polished_val < best_val:
                    best_val = polished_val
                    best_params = polished_params
                    best_h = _chirp_harmonics(best_params)

    # Phase 4: A few random restarts in parametric space
    for i in range(5):
        if time.time() > deadline - 20:
            break
        rng = np.random.default_rng(100 + i)
        rand_params = rng.uniform(bounds[0][0], bounds[0][1], size=len(bounds))
        # Adjust bounds for non-uniform ranges
        rand_params[0] = rng.uniform(0, 2*np.pi)
        rand_params[1] = rng.uniform(-5, 5)
        rand_params[2] = rng.uniform(-5, 5)
        rand_params[3] = rng.uniform(0, 1)
        rand_params[4] = rng.uniform(0, 2*np.pi)
        rand_params[5] = rng.uniform(0, 1)
        rand_params[6] = rng.uniform(0, 2*np.pi)
        
        rand_val = _objective_params(rand_params)
        if rand_val < best_val:
            best_val = rand_val
            best_params = rand_params
            best_h = _chirp_harmonics(best_params)
            
            if time.time() < deadline - 10:
                polished_params = _nm_polish(best_params, deadline)
                polished_val = _objective_params(polished_params)
                if polished_val < best_val:
                    best_val = polished_val
                    best_params = polished_params
                    best_h = _chirp_harmonics(best_params)

    # Final validation
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h
