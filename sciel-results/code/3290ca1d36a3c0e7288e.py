import numpy as np
from scipy import optimize
import time

_start = time.time()
_DEADLINE = 200.0

def _hermite(n, x):
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2*x
    H0 = np.ones_like(x)
    H1 = 2*x
    for k in range(1, n):
        H2 = 2*x*H1 - 2*k*H0
        H0, H1 = H1, H2
    return H1

# Precompute
_N = 10000
_X = np.linspace(-8, 8, _N)
_DX = _X[1] - _X[0]
_H0 = _hermite(0, _X)
_H4 = _hermite(4, _X)
_H8 = _hermite(8, _X)
_H12 = _hermite(12, _X)
_EXP = np.exp(-_X**2/2)
_X4 = _X**4
_XI = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(_N, d=_DX))
_XI4 = _XI**4

def _objective(coeffs):
    a0, a4, a8 = coeffs
    H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 1680.0, 665280.0
    a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0
    
    g = (a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12) * _EXP
    I1 = np.sum(_X4 * g**2) * _DX
    
    g_hat = np.fft.fftshift(np.fft.fft(g)) * _DX
    I2 = np.sum(_XI4 * np.abs(g_hat)**2) * _DX
    
    return max(I1, I2)

def construct():
    global _start, _DEADLINE
    
    seed = np.array([-113/100, 1/25, 1/3240], dtype=float)
    best_val = _objective(seed)
    best_x = seed.copy()
    
    rng = np.random.default_rng(12345)
    starts = [seed.copy()]
    # Add perturbations around seed
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(5):
            starts.append(seed + scale * rng.standard_normal(3))
    # Random starts
    for _ in range(10):
        starts.append(rng.uniform(-10, 10, size=3))
    
    for x0 in starts:
        if time.time() - _start > _DEADLINE - 15:
            break
        try:
            res = optimize.minimize(_objective, x0, method='Nelder-Mead',
                                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_x, dtype=float)
