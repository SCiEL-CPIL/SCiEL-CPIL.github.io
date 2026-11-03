"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


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
    N = 8192
    x = np.linspace(-25, 25, N)
    dx = x[1] - x[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2)

    # Effective basis vectors with g(0)=0 constraint: c12 = -(c0+12*c4+1680*c8)/40320
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss

    M = np.array([
        [np.sum(v0**2) * dx, np.sum(v0 * v4) * dx, np.sum(v0 * v8) * dx],
        [np.sum(v0 * v4) * dx, np.sum(v4**2) * dx, np.sum(v4 * v8) * dx],
        [np.sum(v0 * v8) * dx, np.sum(v4 * v8) * dx, np.sum(v8**2) * dx]
    ])

    Mx = np.array([
        [np.sum(x**2 * v0**2) * dx, np.sum(x**2 * v0 * v4) * dx, np.sum(x**2 * v0 * v8) * dx],
        [np.sum(x**2 * v0 * v4) * dx, np.sum(x**2 * v4**2) * dx, np.sum(x**2 * v4 * v8) * dx],
        [np.sum(x**2 * v0 * v8) * dx, np.sum(x**2 * v4 * v8) * dx, np.sum(x**2 * v8**2) * dx]
    ])

    G0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v0))) * dx
    G4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v4))) * dx
    G8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v8))) * dx

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
    return np.sqrt(x2_norm * freq2_norm / (norm2 * norm2))


def _solve_critical_points(M, Mx, Mf):
    """
    Find all critical points of the Rayleigh quotient ratio.
    
    We minimize f(c) = (c^T Mx c)(c^T Mf c) / (c^T M c)^2
    subject to c^T M c = 1 (normalization).
    
    The critical point condition: gradient of log(f) = 0
    => 2*Mx*c/(c^T Mx c) + 2*Mf*c/(c^T Mf c) - 4*M*c/(c^T M c) = 0
    
    Let a = c^T Mx c, b = c^T Mf c, d = c^T M c = 1.
    Then: Mx*c/a + Mf*c/b - 2*M*c = 0
    
    This is: (1/a)*Mx*c + (1/b)*Mf*c = 2*M*c
    
    Let lambda = 1/a, mu = 1/b. Then:
    lambda*Mx*c + mu*Mf*c - 2*M*c = 0
    
    This is a generalized eigenvalue-like problem with two parameters.
    We can eliminate mu by taking the cross product approach or by 
    parameterizing. Since we have 3 unknowns (c0,c1,c2) and 2 
    parameters (lambda, mu), we can fix one coefficient and solve.
    
    Alternative: use the fact that at a critical point, c must be 
    orthogonal (in some sense) to variations. We can set up the 
    system by eliminating c using the constraint.
    
    Better approach: parameterize c = (1, u, v) in affine chart, then 
    the objective becomes a rational function in (u,v). Its critical 
    points satisfy two polynomial equations.
    """
    # Work in affine coordinates: c = (1, u, v)
    # f(u,v) = [(1,u,v)^T Mx (1,u,v) * (1,u,v)^T Mf (1,u,v)] / [(1,u,v)^T M (1,u,v)]^2
    
    # Extract components
    Mx_00, Mx_01, Mx_02 = Mx[0,0], Mx[0,1], Mx[0,2]
    Mx_10, Mx_11, Mx_12 = Mx[1,0], Mx[1,1], Mx[1,2]
    Mx_20, Mx_21, Mx_22 = Mx[2,0], Mx[2,1], Mx[2,2]
    
    Mf_00, Mf_01, Mf_02 = Mf[0,0], Mf[0,1], Mf[0,2]
    Mf_10, Mf_11, Mf_12 = Mf[1,0], Mf[1,1], Mf[1,2]
    Mf_20, Mf_21, Mf_22 = Mf[2,0], Mf[2,1], Mf[2,2]
    
    M_00, M_01, M_02 = M[0,0], M[0,1], M[0,2]
    M_10, M_11, M_12 = M[1,0], M[1,1], M[1,2]
    M_20, M_21, M_22 = M[2,0], M[2,1], M[2,2]
    
    # (1,u,v)^T Mx (1,u,v) = Mx_00 + 2*u*Mx_10 + 2*v*Mx_20 + u^2*Mx_11 + 2*u*v*Mx_12 + v^2*Mx_22
    # Let's define:
    # A(u,v) = (1,u,v)^T Mx (1,u,v)
    # B(u,v) = (1,u,v)^T Mf (1,u,v)
    # D(u,v) = (1,u,v)^T M (1,u,v)
    
    # f = A*B / D^2
    # Critical points: df/du = 0, df/dv = 0
    # => (A'*B - A*B') * D^2 - A*B * 2*D*D' = 0
    # => A'*B*D - A*B'*D - 2*A*B*D' = 0
    # Similarly for v.
    
    # This gives two polynomial equations in u, v.
    # Let's compute them symbolically by evaluating on a grid and 
    # using resultant, or better: directly compute the polynomial coefficients.
    
    # A(u,v) = Mx_00 + 2*Mx_10*u + 2*Mx_20*v + Mx_11*u^2 + 2*Mx_12*u*v + Mx_22*v^2
    # B(u,v) = Mf_00 + 2*Mf_10*u + 2*Mf_20*v + Mf_11*u^2 + 2*Mf_12*u*v + Mf_22*v^2
    # D(u,v) = M_00 + 2*M_10*u + 2*M_20*v + M_11*u^2 + 2*M_12*u*v + M_22*v^2
    
    # Partial derivatives:
    # A_u = 2*Mx_10 + 2*Mx_11*u + 2*Mx_12*v
    # A_v = 2*Mx_20 + 2*Mx_12*u + 2*Mx_22*v
    # B_u = 2*Mf_10 + 2*Mf_11*u + 2*Mf_12*v
    # B_v = 2*Mf_20 + 2*Mf_12*u + 2*Mf_22*v
    # D_u = 2*M_10 + 2*M_11*u + 2*M_12*v
    # D_v = 2*M_20 + 2*M_12*u + 2*M_22*v
    
    # Equation 1: A_u*B*D - A*B_u*D - 2*A*B*D_u = 0
    # Equation 2: A_v*B*D - A*B_v*D - 2*A*B*D_v = 0
    
    # These are polynomials of degree 1+2+2 = 5 in (u,v).
    # We can solve this system using Groebner basis or by elimination.
    
    # For simplicity, let's use a numerical approach: sample on a grid 
    # and find zeros, then refine with Newton's method.
    
    # Actually, let's use a more direct approach: use scipy's fsolve 
    # from multiple initial guesses, combined with a global search.
    
    def gradient_uv(u, v):
        """Compute the gradient of log(f) with respect to (u,v)."""
        c = np.array([1.0, u, v])
        A = c @ Mx @ c
        B = c @ Mf @ c
        D = c @ M @ c
        
        if A <= 0 or B <= 0 or D <= 0:
            return np.array([np.nan, np.nan])
        
        A_u = 2 * (Mx[0,1] + Mx[1,1]*u + Mx[1,2]*v)
        A_v = 2 * (Mx[0,2] + Mx[1,2]*u + Mx[2,2]*v)
        B_u = 2 * (Mf[0,1] + Mf[1,1]*u + Mf[1,2]*v)
        B_v = 2 * (Mf[0,2] + Mf[1,2]*u + Mf[2,2]*v)
        D_u = 2 * (M[0,1] + M[1,1]*u + M[1,2]*v)
        D_v = 2 * (M[0,2] + M[1,2]*u + M[2,2]*v)
        
        # d(log f)/du = A_u/A - B_u/B - 2*D_u/D
        # d(log f)/dv = A_v/A - B_v/B - 2*D_v/D
        g_u = A_u/A - B_u/B - 2*D_u/D
        g_v = A_v/A - B_v/B - 2*D_v/D
        return np.array([g_u, g_v])
    
    # Use differential evolution to find all local minima of the bound
    # in the (u,v) plane, then verify they are critical points.
    
    def bound_uv(u, v):
        c = np.array([1.0, u, v])
        return _compute_bound(c, M, Mx, Mf)
    
    # Global search with DE
    de_result = differential_evolution(
        bound_uv,
        bounds=[(-5, 5), (-5, 5)],
        seed=42,
        maxiter=200,
        tol=1e-10,
        popsize=20,
        polish=True
    )
    
    best_bound = de_result.fun
    best_uv = de_result.x
    
    # Also try the other affine chart: c = (u, 1, v)
    def bound_uv2(u, v):
        c = np.array([u, 1.0, v])
        return _compute_bound(c, M, Mx, Mf)
    
    de_result2 = differential_evolution(
        bound_uv2,
        bounds=[(-5, 5), (-5, 5)],
        seed=42,
        maxiter=200,
        tol=1e-10,
        popsize=20,
        polish=True
    )
    
    if de_result2.fun < best_bound:
        best_bound = de_result2.fun
        best_uv = de_result2.x
        # Convert back: c = (u, 1, v)
        c_best = np.array([de_result2.x[0], 1.0, de_result2.x[1]])
    else:
        c_best = np.array([1.0, de_result.x[0], de_result.x[1]])
    
    # Also try c = (u, v, 1)
    def bound_uv3(u, v):
        c = np.array([u, v, 1.0])
        return _compute_bound(c, M, Mx, Mf)
    
    de_result3 = differential_evolution(
        bound_uv3,
        bounds=[(-5, 5), (-5, 5)],
        seed=42,
        maxiter=200,
        tol=1e-10,
        popsize=20,
        polish=True
    )
    
    if de_result3.fun < best_bound:
        best_bound = de_result3.fun
        c_best = np.array([de_result3.x[0], de_result3.x[1], 1.0])
    
    # Final polish with Nelder-Mead on the best solution
    if time.time() - _start_time < deadline - 10:
        def bound_from_c(c):
            return _compute_bound(c, M, Mx, Mf)
        
        res = minimize(
            bound_from_c,
            x0=c_best,
            method='Nelder-Mead',
            options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 30000, 'adaptive': True}
        )
        if res.fun < best_bound:
            best_bound = res.fun
            c_best = res.x
    
    return c_best, best_bound


_start_time = None
deadline = None

def construct():
    global _start_time, deadline
    _start_time = time.time()
    deadline = 175
    
    M, Mx, Mf = _build_matrices()
    
    # Try the exact critical point approach first
    try:
        c_best, best_bound = _solve_critical_points(M, Mx, Mf)
    except Exception:
        # Fallback to seed
        c_best = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
        best_bound = _compute_bound(c_best, M, Mx, Mf)
    
    # Verify and possibly refine
    final_bound = _compute_bound(c_best, M, Mx, Mf)
    
    # If the solution is in a bad region, try the seed
    c_seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_bound = _compute_bound(c_seed, M, Mx, Mf)
    
    if seed_bound < final_bound:
        c_best = c_seed
        final_bound = seed_bound
    
    return c_best

# EVOLVE-BLOCK-END
