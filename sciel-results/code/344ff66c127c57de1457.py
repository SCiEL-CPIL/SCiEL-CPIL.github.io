# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _phys_hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) via recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _build_g(coeffs, x):
    """Return g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) * exp(-x^2/2).

    a12 is chosen so that g(0) = 0.
    """
    a0, a4, a8 = coeffs
    H0 = _phys_hermite(0, x)
    H4 = _phys_hermite(4, x)
    H8 = _phys_hermite(8, x)
    H12 = _phys_hermite(12, x)
    H0_0, H4_0, H8_0, H12_0 = 1.0, 3.0, 105.0, 10395.0
    # g(0) = (a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0) * 1 = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    return poly * np.exp(-0.5 * x * x)


def _objective(coeffs):
    """Uncertainty concentration proxy: A(g) * A(Fourier(g)) using FFT."""
    # Use a large grid to minimize edge effects. 
    # Hermite-Gaussians decay as exp(-x^2/2). 
    # At x=10, exp(-50) is ~1.9e-22. So 10 is plenty.
    # Let's use range [-12, 12] with 16384 points.
    N = 16384
    x = np.linspace(-12.0, 12.0, N)
    dx = x[1] - x[0]
    
    g = _build_g(coeffs, x)
    
    # L1 mass of g
    A_g = np.trapz(np.abs(g), x)
    
    # Compute Fourier Transform using FFT
    # Convention: hat_g(xi) = int g(x) exp(-2 pi i x xi) dx
    # FFT computes sum_j g_j exp(-2 pi i j k / N)
    # Relationship: hat_g(xi_k) ~ dx * G_k
    G = np.fft.fft(g)
    # Shift so that zero frequency is in the center
    G = np.fft.fftshift(G)
    
    # Frequency axis
    # The FFT frequencies are k / (N dx) for k = 0...N/2, -N/2...-1
    # After fftshift, the axis is -N/2...N/2
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    # Scale to match continuous FT definition
    hat_g = G * dx
    
    # L1 mass of hat_g
    A_ghat = np.trapz(np.abs(hat_g), freq)
    
    return A_g * A_ghat


def construct():
    # Goncalves et al. (2017) seed
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    try:
        base_val = _objective(seed)
        best = seed.copy()
        best_val = base_val
        
        # Multi-start optimization
        rng = np.random.default_rng(42)
        
        # Define bounds to prevent coefficients from blowing up
        # H12 is large, so a12 will be small if a0..a8 are O(1).
        # Let's allow a wide range.
        bounds = [(-100, 100), (-100, 100), (-100, 100)]
        
        for attempt in range(5):
            if attempt == 0:
                x0 = seed
            else:
                # Perturb seed
                x0 = seed + rng.normal(0, 0.1, size=3)
            
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best = res.x
        
        return best
    except Exception:
        return seed
# EVOLVE-BLOCK-END
