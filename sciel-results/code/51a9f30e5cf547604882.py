"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


def _hermite_phys(n, x):
    """
    Physicists' Hermite polynomials H_n(x).
    Defined by H_n(x) = (-1)^n exp(x^2) d^n/dx^n exp(-x^2).
    """
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    if n == 2:
        return 4 * x**2 - 2
    if n == 3:
        return 8 * x**3 - 12 * x
    if n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    if n == 5:
        return 32 * x**5 - 160 * x**3 + 120 * x
    if n == 6:
        return 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    if n == 7:
        return 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    if n == 8:
        return 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    if n == 9:
        return 512 * x**9 - 10752 * x**7 + 60480 * x**5 - 120960 * x**3 + 60480 * x
    if n == 10:
        return 1024 * x**10 - 32256 * x**8 + 290304 * x**6 - 1016064 * x**4 + 1209600 * x**2 - 30240
    if n == 11:
        return 2048 * x**11 - 80640 * x**9 + 873152 * x**7 - 4365696 * x**5 + 8731584 * x**3 - 665280 * x
    if n == 12:
        return 4096 * x**12 - 204800 * x**10 + 3276800 * x**8 - 23887872 * x**6 + 79626240 * x**4 - 119439360 * x**2 + 4656960
    # Fallback: use scipy for higher n
    from scipy.special import eval_hermite
    return eval_hermite(n, x)


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f) * A(Fourier(f)) for the test function
    g(x) = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    Returns the bound value.
    """
    # Determine c12 so that g(0) = 0
    # g(0) = (c0 * H0(0) + c4 * H4(0) + c8 * H8(0) + c12 * H12(0)) * exp(0)
    # H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = 4656960
    H0_0 = 1
    H4_0 = 12
    H8_0 = 1680
    H12_0 = 4656960
    
    if abs(H12_0) < 1e-15:
        return np.inf
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Numerically compute A(g) and A(ĝ)
    # We need to find the smallest interval where |g(x)| is concentrated
    # A(f) is the essential support in a specific sense related to uncertainty
    
    # For the uncertainty principle bound, we compute:
    # The Fourier transform of H_n(x) exp(-x^2/2) is (i)^n H_n(x) exp(-x^2/2) * sqrt(2*pi)
    # Wait, let me be more careful. The Fourier transform convention matters.
    
    # Using the convention: F(f)(k) = integral f(x) exp(-2*pi*i*x*k) dx
    # For Gaussian exp(-x^2/2), F[exp(-x^2/2)] = sqrt(2*pi) * exp(-2*pi^2*k^2)
    # This is not symmetric. Let me use the unitary convention or the physics convention.
    
    # Actually, for the uncertainty principle in the context of this problem,
    # the standard approach uses the Fourier transform as a unitary operator.
    # Let's use: F(f)(k) = (1/sqrt(2*pi)) * integral f(x) exp(-i*x*k) dx
    # Then F[exp(-x^2/2)] = exp(-k^2/2)
    # And F[H_n(x) exp(-x^2/2)] = (i)^n H_n(x) exp(-x^2/2)
    
    # With this convention, the Hermite functions are eigenfunctions of F.
    
    # So g(x) = sum_{n in {0,4,8,12}} c_n H_n(x) exp(-x^2/2)
    # F(g)(k) = sum_{n in {0,4,8,12}} c_n (i)^n H_n(k) exp(-k^2/2)
    
    # Now A(f) is related to the "concentration" of f. In the context of 
    # uncertainty inequalities for Fourier-invariant test functions, A(f) typically
    # refers to the width or support of f.
    
    # For a function that is a sum of Hermite functions, the "uncertainty" 
    # is often measured by the second moment or by finding the minimal 
    # annulus/interval containing most of the mass.
    
    # Given the task says "upper bound on A(f)*A(Fourier(f))", and this is 
    # related to the Heisenberg uncertainty principle, let's compute:
    # The variance of x and the variance of k (Fourier variable).
    
    # For a normalized function psi(x), 
    # Delta_x = sqrt(<x^2> - <x>^2)
    # Delta_k = sqrt(<k^2> - <k>^2)
    # The product Delta_x * Delta_k >= 1/2
    
    # But A(f) might be something different. Let me think about this differently.
    
    # The task mentions "tune a Fourier-invariant Hermite-polynomial times Gaussian 
    # test function to reduce the upper bound on A(f)*A(Fourier(f))".
    # 
    # In the context of uncertainty inequalities, particularly for the 
    # uncertainty principle, A(f) could refer to the "annulus" or the 
    # concentration of f in physical space and Fourier space.
    
    # One common formulation: For a function f, define the "uncertainty" 
    # as the smallest R such that integral_{|x|>R} |f(x)|^2 dx <= epsilon
    # and similarly for Fourier space. But this depends on epsilon.
    
    # Another approach: Use the second moment. For a normalized f,
    # A(f) = sqrt(<x^2> / ||f||^2) where <x^2> = integral x^2 |f(x)|^2 dx
    
    # Let's compute the normalized second moments.
    
    # First, compute g(x) and its norm
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    
    H0 = _hermite_phys(0, x)
    H4 = _hermite_phys(4, x)
    H8 = _hermite_phys(8, x)
    H12 = _hermite_phys(12, x)
    
    gauss = np.exp(-x**2 / 2)
    
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # Normalize
    norm_sq = np.trapz(g**2, x)
    if norm_sq < 1e-15:
        return np.inf
    
    g_norm = g / np.sqrt(norm_sq)
    
    # Compute <x^2>
    x2_moment = np.trapz(x**2 * g_norm**2, x)
    
    # Compute the Fourier transform analytically
    # F(g)(k) = (1/sqrt(2*pi)) * integral g(x) exp(-i*x*k) dx
    # Since g(x) = sum c_n H_n(x) exp(-x^2/2), 
    # F(g)(k) = (1/sqrt(2*pi)) * sum c_n (i)^n H_n(k) exp(-k^2/2) * (2*pi) / sqrt(2*pi)
    
    # Wait, let me be more careful. The Fourier transform of exp(-x^2/2) is:
    # integral exp(-x^2/2) exp(-i*x*k) dx = sqrt(2*pi) * exp(-k^2/2)
    # So F[exp(-x^2/2)] = (1/sqrt(2*pi)) * sqrt(2*pi) * exp(-k^2/2) = exp(-k^2/2)
    
    # For H_n(x) exp(-x^2/2), the Fourier transform is:
    # F[H_n(x) exp(-x^2/2)] = (i)^n H_n(k) exp(-k^2/2)
    # This is a well-known result.
    
    # So F(g)(k) = (1/sqrt(2*pi)) * integral [sum c_n H_n(x) exp(-x^2/2)] exp(-i*x*k) dx
    # = sum c_n (i)^n H_n(k) exp(-k^2/2) / sqrt(2*pi) * ... 
    
    # Actually, let me just state: if we define the Fourier transform as
    # (Ff)(k) = (1/sqrt(2*pi)) int f(x) e^{-ixk} dx
    # then F[H_n(x) e^{-x^2/2}] = i^n H_n(k) e^{-k^2/2}
    
    # This is because the Hermite functions psi_n(x) = (1/sqrt(2^n n! sqrt(pi))) H_n(x) e^{-x^2/2}
    # are eigenfunctions of F with eigenvalues i^n.
    
    # So for g(x) = sum c_n H_n(x) e^{-x^2/2},
    # F(g)(k) = sum c_n i^n H_n(k) e^{-k^2/2}
    
    # But we need to be careful about normalization. Let's just compute it directly.
    
    # Actually, the exact form of A(f) matters. Let me assume A(f) is related to 
    # the second moment (variance) of the probability density |f|^2.
    
    # For the Fourier side, we compute the second moment of |F(g)(k)|^2.
    
    k = x  # Same grid
    Fg = (c0 * (1j)**0 * H0 + c4 * (1j)**4 * H4 + c8 * (1j)**8 * H8 + c12 * (1j)**12 * H12) * gauss
    # Note: (i)^4 = 1, (i)^8 = 1, (i)^12 = -1
    
    # Normalize Fg
    Fg_norm_sq = np.trapz(np.abs(Fg)**2, k)
    if Fg_norm_sq < 1e-15:
        return np.inf
    
    Fg_norm = Fg / np.sqrt(Fg_norm_sq)
    
    k2_moment = np.trapz(k**2 * np.abs(Fg_norm)**2, k)
    
    # The uncertainty product
    delta_x = np.sqrt(max(0, x2_moment))  # Since <x> = 0 by symmetry
    delta_k = np.sqrt(max(0, k2_moment))  # Since <k> = 0 by symmetry
    
    return delta_x * delta_k


def construct():
    """
    Find coefficients c0, c4, c8 that minimize the uncertainty bound.
    The degree-12 coefficient c12 is determined by g(0) = 0.
    """
    # Use differential evolution for global optimization
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # First try a grid search to find a good region
    best_result = None
    best_bound = np.inf
    
    # Try the initial values
    init = np.array([-113 / 100, 1 / 25, 1 / 3240])
    bound_init = _compute_bound(*init)
    best_bound = bound_init
    best_result = init.copy()
    
    # Local optimization from the initial point
    try:
        result = minimize(_compute_bound, init, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_bound:
            best_bound = result.fun
            best_result = result.x
    except Exception:
        pass
    
    # Try differential evolution for a more global search
    try:
        de_result = differential_evolution(_compute_bound, bounds, seed=42,
                                           maxiter=100, tol=1e-8, polish=True)
        if de_result.fun < best_bound:
            best_bound = de_result.fun
            best_result = de_result.x
    except Exception:
        pass
    
    # Try a few random restarts
    rng = np.random.RandomState(123)
    for _ in range(20):
        x0 = rng.uniform(-5, 5, size=3)
        try:
            res = minimize(_compute_bound, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x
        except Exception:
            pass
    
    return np.array(best_result)
# EVOLVE-BLOCK-END
