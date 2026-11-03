"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def construct():
    """Find optimal coefficients c0, c4, c8 to minimize the bound."""
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 180s kill
    
    # Hermite polynomial values at 0 for the constraint P(0)=0
    # Using physicists' Hermite polynomials
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 1680
    # H12(0) = 1032192
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 1032192.0
    
    def get_c12(c0, c4, c8):
        """Compute c12 from constraint P(0) = 0."""
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Precompute grid
    x = np.linspace(-12, 12, 20001)
    dx = x[1] - x[0]
    
    # Precompute Hermite polynomials on the grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)
    
    def compute_bound(c0, c4, c8):
        """Compute the upper bound C4 for given coefficients."""
        c12 = get_c12(c0, c4, c8)
        
        # Evaluate polynomial part P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        
        # g(x) = P(x) * exp(-x^2/2)
        g = P * gauss
        
        # Compute L2 norm squared
        norm_g_sq = np.trapz(g**2, x)
        
        if norm_g_sq < 1e-30:
            return 1e10
        
        # Normalize to unit L2 norm
        g_norm = g / np.sqrt(norm_g_sq)
        
        # For the uncertainty principle, we want to minimize the product
        # of some measure of spread in x-space and frequency space.
        # A common formulation involves the L1 norm or the second moment.
        
        # Let's compute the L1 norm of the normalized function
        l1_norm = np.trapz(np.abs(g_norm), x)
        
        # Also compute the second moment (variance) as a measure of spread
        # Var(x) = E[x^2] - (E[x])^2
        E_x = np.trapz(x * g_norm**2, x)
        E_x2 = np.trapz(x**2 * g_norm**2, x)
        var_x = E_x2 - E_x**2
        
        # For the Fourier transform of a Hermite-Gaussian combination,
        # since all powers are even, the Fourier transform has the same form
        # (up to a phase factor of i^n, which is 1 for even n in this case
        # if we use the convention that gives real transforms for even functions).
        # Actually, F[Hn(x)exp(-x^2/2)] = i^n Hn(k) exp(-k^2/2)
        # For even n, i^n is real (1 or -1).
        # Since our coefficients are real and we have only even powers,
        # the Fourier transform will also be a real function of the same form.
        
        # Let's compute the Fourier transform numerically to be safe,
        # or use the property that for these specific functions, the variance
        # in frequency space is related to the variance in x-space.
        
        # For a normalized Gaussian exp(-x^2/2), Var(x) = 1/2, Var(k) = 1/2.
        # The product of standard deviations is 1/2.
        # The uncertainty principle says sigma_x * sigma_k >= 1/2.
        
        # Let's compute the Fourier transform using FFT for accuracy
        # But first, let's try a simpler approach: compute the second moment
        # of the Fourier transform analytically if possible.
        
        # For now, let's use the L1 norm as a proxy for the "width"
        # and assume the frequency space width is similar due to the
        # self-similar nature of the Hermite-Gaussian functions.
        
        # A better measure: the ratio of L1 to L2 norm, which is related
        # to the "peakiness" or concentration.
        
        # Actually, let's compute the Fourier transform properly.
        # Use FFT with appropriate scaling.
        
        # The grid is uniform, so we can use FFT.
        # g_hat(k) = integral g(x) exp(-ikx) dx
        # Using FFT: sum g[n] exp(-2*pi*i*n*k/N) * dx
        # We need to match the frequency axis.
        
        N = len(x)
        G = np.fft.fftshift(np.fft.fft(g_norm)) * dx
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        
        # Normalize the Fourier transform to unit L2 norm
        norm_g_hat_sq = np.trapz(np.abs(G)**2, freqs)
        
        if norm_g_hat_sq < 1e-30:
            return 1e10
        
        G_norm = G / np.sqrt(norm_g_hat_sq)
        
        # Compute L1 norm in frequency space
        l1_norm_hat = np.trapz(np.abs(G_norm), freqs)
        
        # The uncertainty product can be measured as l1_norm * l1_norm_hat
        # or using variances. Let's use the product of L1 norms as a
        # measure of the "width" in both spaces.
        
        # Actually, a more standard measure is the product of standard deviations.
        # Let's compute that.
        
        # Variance in x-space
        E_x = np.trapz(x * g_norm**2, x)
        E_x2 = np.trapz(x**2 * g_norm**2, x)
        var_x = E_x2 - E_x**2
        sigma_x = np.sqrt(max(var_x, 0))
        
        # Variance in frequency space
        E_k = np.trapz(freqs * np.abs(G_norm)**2, freqs)
        E_k2 = np.trapz(freqs**2 * np.abs(G_norm)**2, freqs)
        var_k = E_k2 - E_k**2
        sigma_k = np.sqrt(max(var_k, 0))
        
        # The uncertainty product is sigma_x * sigma_k
        # The lower bound is 1/2, so we want to minimize this product.
        
        return sigma_x * sigma_k
    
    # Initial guess from the seed program
    c0_init, c4_init, c8_init = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    
    # Define objective function with time guard
    def objective(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        c0, c4, c8 = coeffs
        try:
            return compute_bound(c0, c4, c8)
        except:
            return 1e10
    
    # Try multiple starting points
    best_result = None
    best_obj = np.inf
    
    start_points = [
        [c0_init, c4_init, c8_init],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [-0.5, 0.1, 0.01],
        [-0.1, 0.01, 0.001],
    ]
    
    for sp in start_points:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            break
        
        try:
            result = minimize(objective, sp, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except:
            continue
    
    if best_result is not None:
        c0, c4, c8 = best_result.x
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed values
        return np.array([c0_init, c4_init, c8_init])
# EVOLVE-BLOCK-END
