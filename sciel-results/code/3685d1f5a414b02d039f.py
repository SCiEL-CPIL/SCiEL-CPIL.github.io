"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite

def construct():
    # Use a wider grid with more points for accurate integration
    x = np.linspace(-20, 20, 12000)
    dx = x[1] - x[0]
    
    # Hermite polynomials at x
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty ratio for given coefficients."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x)
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)
            
            # Compute norms using trapezoidal rule
            norm2_sq = np.trapz(g**2, x)
            if norm2_sq < 1e-15:
                return 1e10
                
            norm4 = np.trapz(np.abs(g)**4, x)
            norm4_val = norm4 ** 0.25
            
            # Compute Fourier transform using FFT with correct scaling
            # For continuous FT: F(k) = integral f(x) exp(-2*pi*i*k*x) dx
            # FFT gives sum f[n] exp(-2*pi*i*m*n/N), so we multiply by dx
            # And the frequency spacing is 1/(N*dx)
            n = len(g)
            nfft = 1
            while nfft < 4 * n:
                nfft *= 2
            
            # Zero-pad and compute FFT
            G_fft = np.fft.fft(g, n=nfft) * dx
            
            # Parseval: integral |f|^2 dx = integral |F|^2 dk
            # With our scaling, sum |G_fft|^2 * (2*pi/nfft) approximates integral |F|^2 dk
            # But we need to be careful with the 2*pi factor
            # Actually: F(k) = sum f(x_j) exp(-2*pi*i*k*x_j) dx
            # The FFT computes sum f(x_j) exp(-2*pi*i*m*j/N)
            # So F(k_m) = G_fft[m] * dx where k_m = m/(N*dx) for m < N/2
            # and k_m = (m-N)/(N*dx) for m >= N/2
            # integral |F(k)|^2 dk = sum |F(k_m)|^2 * dk = sum |G_fft[m]|^2 * dx^2 * (1/(N*dx))
            # = (dx/N) * sum |G_fft[m]|^2
            dk = 1.0 / (nfft * dx)
            
            norm2_hat_sq = np.sum(np.abs(G_fft)**2) * dk
            norm4_hat = np.sum(np.abs(G_fft)**4) * dk
            norm4_hat_val = norm4_hat ** 0.25
            
            if norm2_hat_sq < 1e-15:
                return 1e10
                
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Use tanh for smooth sign transition
            sign0 = np.tanh(s0)
            sign4 = np.tanh(s4)
            sign8 = np.tanh(s8)
            
            # Ensure signs are not too close to zero to avoid numerical issues
            eps = 1e-6
            if abs(sign0) < eps:
                sign0 = eps
            if abs(sign4) < eps:
                sign4 = eps
            if abs(sign8) < eps:
                sign8 = eps
            
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Define bounds for differential evolution
    # m_i: log-magnitude, range from -5 to 4 (coefficients from ~0.007 to ~55)
    # s_i: sign parameter, range from -3 to 3 (tanh gives -1 to 1)
    bounds = [(-5, 4), (-5, 4), (-5, 4), (-3, 3), (-3, 3), (-3, 3)]
    
    # Run differential evolution with multiple seeds for robustness
    best_ratio = 1e10
    best_params = None
    
    for seed in [42, 123, 456, 789]:
        result = differential_evolution(
            objective_from_params, bounds, 
            seed=seed, maxiter=200, popsize=25,
            tol=1e-10, polish=False,
            init='sobol',
            mutation=(0.3, 1.0),
            recombination=0.9
        )
        
        if result.fun < best_ratio:
            best_ratio = result.fun
            best_params = result.x
    
    if best_params is not None:
        m0, m4, m8, s0, s4, s8 = best_params
        sign0 = np.tanh(s0)
        sign4 = np.tanh(s4)
        sign8 = np.tanh(s8)
        
        eps = 1e-6
        if abs(sign0) < eps:
            sign0 = eps
        if abs(sign4) < eps:
            sign4 = eps
        if abs(sign8) < eps:
            sign8 = eps
            
        c0 = sign0 * np.exp(m0)
        c4 = sign4 * np.exp(m4)
        c8 = sign8 * np.exp(m8)
        
        # Polish with local optimization using direct coefficients
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        local_result = minimize(
            local_obj, 
            [c0, c4, c8],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if local_result.fun < best_ratio:
            c0, c4, c8 = local_result.x
        
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
