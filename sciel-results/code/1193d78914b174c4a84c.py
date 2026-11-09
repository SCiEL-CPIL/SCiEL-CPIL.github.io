"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_c3(x):
    """Compute the C3 upper bound from a signed step function.
    C3 is the maximum absolute autocorrelation sidelobe normalized by N.
    """
    n = len(x)
    # Autocorrelation via FFT
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # Sidelobes: lags 1..n-1
    sidelobes = np.abs(acf[1:])
    return np.max(sidelobes) / n

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 20s safety margin before 150s kill
    
    # Best result so far
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    # Strategy 1: Sinusoidal parameterization with gradient optimization
    # x_i = sign(sin(2*pi*f*t_i + phi)), t_i in [-1/4, 1/4]
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    def _sinusoidal_c3(params):
        f, phi = params
        x = np.sign(np.sin(2 * np.pi * f * t + phi))
        # Handle zeros: assign +1
        x[x == 0] = 1.0
        return _compute_c3(x)
    
    # Try multiple initial points for sinusoidal optimization
    best_sinusoidal_c3 = best_c3
    best_sinusoidal_params = (1.0, 0.0)
    
    for f0 in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
        for phi0 in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            if time.time() > deadline - 20:
                break
            try:
                res = minimize(_sinusoidal_c3, [f0, phi0], method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-8})
                if res.fun < best_sinusoidal_c3:
                    best_sinusoidal_c3 = res.fun
                    best_sinusoidal_params = (res.x[0], res.x[1])
            except Exception:
                pass
    
    if best_sinusoidal_c3 < best_c3:
        f_opt, phi_opt = best_sinusoidal_params
        best_x = np.sign(np.sin(2 * np.pi * f_opt * t + phi_opt))
        best_x[best_x == 0] = 1.0
        best_c3 = best_sinusoidal_c3
    
    # Strategy 2: Local search on binary sequence
    # Try flipping individual bits and small neighborhoods
    x = best_x.copy()
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(N):
            if time.time() > deadline - 10:
                break
            x[i] *= -1
            c3 = _compute_c3(x)
            if c3 < best_c3:
                best_c3 = c3
                best_x = x.copy()
                improved = True
            else:
                x[i] *= -1  # revert
    
    # Strategy 3: Random restarts with local search
    rng = np.random.default_rng(42)
    for _ in range(50):
        if time.time() > deadline - 15:
            break
        x_rand = rng.choice([-1.0, 1.0], size=N)
        c3_rand = _compute_c3(x_rand)
        if c3_rand < best_c3:
            best_c3 = c3_rand
            best_x = x_rand.copy()
        # Local search from this point
        x = best_x.copy()
        for _iter in range(200):
            if time.time() > deadline - 15:
                break
            i = rng.integers(0, N)
            x[i] *= -1
            c3 = _compute_c3(x)
            if c3 < best_c3:
                best_c3 = c3
                best_x = x.copy()
            else:
                x[i] *= -1
    
    return best_x
# EVOLVE-BLOCK-END
