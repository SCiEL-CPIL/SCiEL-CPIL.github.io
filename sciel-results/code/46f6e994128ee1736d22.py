"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import root
import time

_start_time = None
deadline = 175.0

def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial Hn(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def _build_matrices():
    """Build the Gram matrices M, Mx, Mf for the constrained basis."""
    # Use a sufficiently large grid for accurate integration
    N = 16384
    # Symmetric grid around 0
    x = np.linspace(-30, 30, N)
    dx = x[1] - x[0]
    
    # Frequencies for Fourier domain
    # We need the Fourier transform of the basis functions.
    # The basis functions are g_k(x) = H_k(x) exp(-x^2/2) adjusted for g(0)=0.
    # Let's define the raw basis functions first.
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2)
    
    # Constraint g(0) = 0.
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=40320.
    # c12 = -(c0 + 12*c4 + 1680*c8) / 40320.
    
    # We form the effective basis vectors v0, v4, v8 such that
    # g(x) = c0*v0(x) + c4*v4(x) + c8*v8(x)
    # where v0 = (H0 - H12/40320)*gauss
    #       v4 = (H4 - 12*H12/40320)*gauss
    #       v8 = (H8 - 1680*H12/40320)*gauss
    
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss
    
    # M: Norm matrix (c^T M c = ||g||^2)
    M = np.array([
        [np.sum(v0**2) * dx, np.sum(v0 * v4) * dx, np.sum(v0 * v8) * dx],
        [np.sum(v0 * v4) * dx, np.sum(v4**2) * dx, np.sum(v4 * v8) * dx],
        [np.sum(v0 * v8) * dx, np.sum(v4 * v8) * dx, np.sum(v8**2) * dx]
    ])
    
    # Mx: Second moment matrix (c^T Mx c = <x^2> * ||g||^2)
    Mx = np.array([
        [np.sum(x**2 * v0**2) * dx, np.sum(x**2 * v0 * v4) * dx, np.sum(x**2 * v0 * v8) * dx],
        [np.sum(x**2 * v0 * v4) * dx, np.sum(x**2 * v4**2) * dx, np.sum(x**2 * v4 * v8) * dx],
        [np.sum(x**2 * v0 * v8) * dx, np.sum(x**2 * v4 * v8) * dx, np.sum(x**2 * v8**2) * dx]
    ])
    
    # Mf: Fourier second moment matrix (c^T Mf c = <k^2> * ||g||^2)
    # We need the Fourier transform of v0, v4, v8.
    # Use FFT.
    # Note: The grid x is not necessarily centered at 0 for FFT if N is even? 
    # np.linspace(-30, 30, N) is symmetric.
    # We should use ifftshift/fftshift to handle the origin correctly for the frequency axis.
    
    G0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v0))) * dx
    G4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v4))) * dx
    G8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v8))) * dx
    
    # Frequency axis
    # d=dx, N points.
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    # Mf_ij = sum(k^2 * G_i * conj(G_j)) * dx
    Mf = np.array([
        [np.sum(freqs**2 * np.abs(G0)**2) * dx, np.sum(freqs**2 * np.real(G0 * np.conj(G4))) * dx, np.sum(freqs**2 * np.real(G0 * np.conj(G8))) * dx],
        [np.sum(freqs**2 * np.real(G0 * np.conj(G4))) * dx, np.sum(freqs**2 * np.abs(G4)**2) * dx, np.sum(freqs**2 * np.real(G4 * np.conj(G8))) * dx],
        [np.sum(freqs**2 * np.real(G0 * np.conj(G8))) * dx, np.sum(freqs**2 * np.real(G4 * np.conj(G8))) * dx, np.sum(freqs**2 * np.abs(G8)**2) * dx]
    ])
    
    return M, Mx, Mf

def _compute_bound(c, M, Mx, Mf):
    """Compute the uncertainty product bound for coefficient vector c."""
    c = np.array(c, dtype=float)
    norm2 = c @ M @ c
    if norm2 <= 0 or not np.isfinite(norm2):
        return np.inf
    x2_norm = c @ Mx @ c
    freq2_norm = c @ Mf @ c
    if not np.isfinite(x2_norm) or x2_norm < 0 or not np.isfinite(freq2_norm) or freq2_norm < 0:
        return np.inf
    # The bound is sqrt(<x^2><k^2>) = sqrt( (x2_norm/norm2) * (freq2_norm/norm2) )
    return np.sqrt(x2_norm * freq2_norm / (norm2 * norm2))

def _solve_critical_points(M, Mx, Mf):
    """
    Find all critical points of the Rayleigh quotient ratio.
    We solve the system:
    B D (Mx c)_i - A D (Mf c)_i - 2 A B (M c)_i = 0 for i=0,1,2
    where A = c^T Mx c, B = c^T Mf c, D = c^T M c.
    """
    solutions = []
    
    # Define the system of equations
    def equations(c):
        A = c @ Mx @ c
        B = c @ Mf @ c
        D = c @ M @ c
        
        # If any are zero or non-finite, return large values to avoid division by zero in gradient or similar
        if A <= 0 or B <= 0 or D <= 0 or not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(D):
            return np.array([1e10, 1e10, 1e10])
            
        Mx_c = Mx @ c
        Mf_c = Mf @ c
        M_c = M @ c
        
        # The condition is: B*D*Mx_c - A*D*Mf_c - 2*A*B*M_c = 0
        # This is a vector equation.
        return B * D * Mx_c - A * D * Mf_c - 2 * A * B * M_c

    # We use multiple initial guesses to find different roots.
    # The solution space is projective (scale invariant).
    # We can fix the norm ||c||_2 = 1 or c^T M c = 1.
    # Let's just use random vectors and let fsolve find the roots.
    # Since the equations are homogeneous of degree 3, if c is a root, -c is a root, and k*c is a root.
    # We should normalize the found roots to avoid duplicates.
    
    rng = np.random.default_rng(42)
    
    # Generate many random starting points
    # We sample from a wide range to ensure we cover the space.
    # Since the coefficients can be of varying magnitudes, we try different scales.
    initial_guesses = []
    for _ in range(100):
        # Random vector
        c0 = rng.normal(0, 1)
        c1 = rng.normal(0, 1)
        c2 = rng.normal(0, 1)
        # Scale randomly to explore different regions
        scale = 10**rng.uniform(-3, 3)
        initial_guesses.append(scale * np.array([c0, c1, c2]))
        
    # Also add some structured guesses
    for scale in [0.01, 0.1, 1, 10, 100]:
        for angle in np.linspace(0, 2*np.pi, 10):
            c0 = scale * np.cos(angle)
            c1 = scale * np.sin(angle)
            c2 = scale * 0.5
            initial_guesses.append(np.array([c0, c1, c2]))
            # And variations
            c0 = scale * 0.5
            c1 = scale * np.cos(angle)
            c2 = scale * np.sin(angle)
            initial_guesses.append(np.array([c0, c1, c2]))
            
    best_bound = np.inf
    best_c = None
    
    for c_init in initial_guesses:
        if time.time() > deadline - 5:
            break
            
        try:
            sol = root(equations, c_init, method='hybr', tol=1e-10)
            if sol.success:
                c_sol = sol.x
                # Check if it's a valid solution (A,B,D > 0)
                A = c_sol @ Mx @ c_sol
                B = c_sol @ Mf @ c_sol
                D = c_sol @ M @ c_sol
                if A > 0 and B > 0 and D > 0:
                    # Calculate bound
                    bound = _compute_bound(c_sol, M, Mx, Mf)
                    if bound < best_bound:
                        best_bound = bound
                        best_c = c_sol.copy()
        except Exception:
            pass
            
    # If no solution found, fall back to a simple optimization
    if best_c is None:
        # Use the seed solution
        best_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
        best_bound = _compute_bound(best_c, M, Mx, Mf)
        
    return best_c, best_bound

def construct():
    global _start_time, deadline
    _start_time = time.time()
    deadline = 175.0
    
    M, Mx, Mf = _build_matrices()
    
    # Try the exact critical point approach
    try:
        c_best, best_bound = _solve_critical_points(M, Mx, Mf)
    except Exception:
        c_best = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
        best_bound = _compute_bound(c_best, M, Mx, Mf)
        
    # Verify and possibly refine
    # We can do a local polish with Nelder-Mead or BFGS on the best candidate
    # to ensure we are at the precise minimum, as fsolve might stop slightly off.
    
    from scipy.optimize import minimize
    
    def bound_from_c(c):
        return _compute_bound(c, M, Mx, Mf)
        
    if time.time() < deadline - 10:
        # Polish the best solution found
        try:
            res = minimize(
                bound_from_c,
                x0=c_best,
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 10000, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                c_best = res.x
        except Exception:
            pass
    
    # Final check against seed
    c_seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_bound = _compute_bound(c_seed, M, Mx, Mf)
    
    if seed_bound < best_bound:
        c_best = c_seed
        best_bound = seed_bound
        
    return c_best

# EVOLVE-BLOCK-END
