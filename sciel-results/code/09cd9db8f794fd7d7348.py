"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We use the seed value as a fallback
_SEED_COEFFS = np.array([-113 / 100, 1 / 25, 1 / 3240])

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s limit
    
    # The task mentions:
    # - Fourier-invariant Hermite-polynomial times Gaussian test function
    # - Three free coefficients multiply H0, H4, H8
    # - H12 coefficient derived to impose P(0)=0 (g(0)=0)
    # - Score is upper bound on C4 (A(f)*A(Fourier(f)))
    
    # The function g(x) = (a0*H0 + a1*H4 + a2*H8 + a3*H12) * exp(-x^2/2)
    # where a3 is determined by g(0) = 0
    
    # Hermite polynomials (physicists' convention):
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12(x) = 4096x^12 - 76800x^10 + 645120x^8 - 2611200x^6 + 5160960x^4 - 4536000x^2 + 960960
    
    # g(0) = (a0 + a1*12 + a2*1680 + a3*960960) * exp(0) = 0
    # So a3 = -(a0 + 12*a1 + 1680*a2) / 960960
    
    # The uncertainty product A(f)*A(Fourier(f)) involves:
    # A(f) = sqrt(∫|f|^2 / ∫|f|^4) or similar concentration measure
    # Actually, for the uncertainty principle, A(f) is typically the standard deviation
    # or a related width measure.
    
    # Let me think about this more carefully. The task says "upper bound on A(f)*A(Fourier(f))"
    # and the score is "upper bound on C4".
    
    # For a function g(x) = P(x) * exp(-x^2/2), where P is a polynomial:
    # The L2 norm, and moments can be computed analytically using Gaussian integrals.
    
    # Let's define the objective. The uncertainty product for a normalized function is:
    # Δx * Δp where Δx^2 = <x^2> - <x>^2 and Δp^2 = <p^2> - <p>^2
    # with p = -i d/dx
    
    # For g(x) = P(x) exp(-x^2/2):
    # <x^n> = ∫ x^n |g|^2 dx / ∫ |g|^2 dx
    # |g|^2 = P(x)^2 exp(-x^2)
    
    # Using the fact that ∫ x^{2k} exp(-x^2) dx = (2k-1)!! * sqrt(pi) / 2^k
    
    # Let me compute the objective numerically for given coefficients.
    
    def _hermite(n, x):
        """Compute physicists' Hermite polynomial H_n(x)."""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 * x
        elif n == 2:
            return 4 * x**2 - 2
        elif n == 3:
            return 8 * x**3 - 12 * x
        elif n == 4:
            return 16 * x**4 - 48 * x**2 + 12
        elif n == 5:
            return 32 * x**5 - 160 * x**3 + 120 * x
        elif n == 6:
            return 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
        elif n == 7:
            return 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
        elif n == 8:
            return 256 * x**8 - 3072 * x**6 + 11520 * x**4 - 13440 * x**2 + 1680
        elif n == 9:
            return 512 * x**9 - 7680 * x**7 + 40320 * x**5 - 80640 * x**3 + 30240 * x
        elif n == 10:
            return 1024 * x**10 - 19456 * x**8 + 133120 * x**6 - 376320 * x**4 + 302400 * x**2 - 30240
        elif n == 11:
            return 2048 * x**11 - 43008 * x**9 + 344064 * x**7 - 1209600 * x**5 + 1663200 * x**3 - 665280 * x
        elif n == 12:
            return 4096 * x**12 - 76800 * x**10 + 645120 * x**8 - 2611200 * x**6 + 5160960 * x**4 - 4536000 * x**2 + 960960
        else:
            raise ValueError(f"Hermite polynomial of order {n} not implemented")
    
    def _compute_a3(a0, a1, a2):
        """Compute a3 such that g(0) = 0."""
        # H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = 960960
        return -(a0 + 12 * a1 + 1680 * a2) / 960960
    
    def _objective(a0, a1, a2):
        """Compute the uncertainty product A(f)*A(Fourier(f)) for g(x) = (a0*H0 + a1*H4 + a2*H8 + a3*H12) * exp(-x^2/2)."""
        a3 = _compute_a3(a0, a1, a2)
        
        # We need to compute:
        # <x^2> = ∫ x^2 P(x)^2 exp(-x^2) dx / ∫ P(x)^2 exp(-x^2) dx
        # <p^2> = ∫ (P'(x))^2 exp(-x^2) dx / ∫ P(x)^2 exp(-x^2) dx
        #         (since p = -i d/dx, and for real functions, <p^2> = ∫ |g'|^2 dx / ∫ |g|^2 dx)
        
        # Actually, Δx^2 = <x^2> - <x>^2 and Δp^2 = <p^2> - <p>^2
        # But for even functions, <x> = 0 and <p> = 0, so Δx^2 = <x^2> and Δp^2 = <p^2>
        
        # P(x) is even (all Hermite polynomials of even degree are even), so g is even.
        # Thus <x> = 0 and <p> = 0.
        
        # Let's use numerical integration with a fine grid.
        x = np.linspace(-10, 10, 20001)
        dx = x[1] - x[0]
        
        H0 = _hermite(0, x)
        H4 = _hermite(4, x)
        H8 = _hermite(8, x)
        H12 = _hermite(12, x)
        
        P = a0 * H0 + a1 * H4 + a2 * H8 + a3 * H12
        g = P * np.exp(-x**2 / 2)
        g2 = g**2
        
        # Normalize
        norm = np.trapz(g2, x)
        if norm < 1e-30:
            return np.inf
        
        # <x^2>
        x2 = np.trapz(x**2 * g2, x) / norm
        
        # <p^2> = ∫ (dg/dx)^2 dx / ∫ g^2 dx
        # dg/dx = P' * exp(-x^2/2) + P * (-x) * exp(-x^2/2)
        #       = (P' - x*P) * exp(-x^2/2)
        
        # Compute P'(x) numerically or analytically
        # H0' = 0, H4' = 64x^3 - 96x, H8' = 2048x^7 - 18432x^5 + 46080x^3 - 26880x, H12' = ...
        
        # Let's use numerical differentiation
        dP = np.gradient(P, x)
        dg = (dP - x * P) * np.exp(-x**2 / 2)
        p2 = np.trapz(dg**2, x) / norm
        
        # Uncertainty product
        return np.sqrt(x2 * p2)
    
    # Try the seed value
    best_score = _objective(*_SEED_COEFFS)
    best_coeffs = _SEED_COEFFS.copy()
    
    # Now do a local optimization using Nelder-Mead or similar
    from scipy.optimize import minimize
    
    def _obj_vec(coeffs):
        return _objective(coeffs[0], coeffs[1], coeffs[2])
    
    # Try a few random starts and local optimizations
    rng = np.random.RandomState(42)
    
    for i in range(20):
        if time.time() > deadline - 30:
            break
        
        # Random start
        if i == 0:
            start = _SEED_COEFFS.copy()
        else:
            start = rng.normal(0, 1, 3)
        
        try:
            result = minimize(_obj_vec, start, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Check deadline
    if time.time() > deadline - 10:
        return best_coeffs
    
    # Return the best coefficients found
    return best_coeffs
# EVOLVE-BLOCK-END
