import numpy as np
from scipy.special import lpmv, factorial
import time

def _build_harmonics():
    """Precompute real orthonormal degree-10 spherical harmonics on the quadrature grid."""
    x, w = np.polynomial.legendre.leggauss(70)
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    dphi = 2 * np.pi / 130
    weights_2d = np.outer(w, np.ones(130)) * dphi
    
    P = {}
    for m in range(0, 11):
        P[m] = lpmv(m, 10, cos_theta)
    
    N = {}
    for m in range(0, 11):
        N[m] = np.sqrt(21.0 / (4 * np.pi) * factorial(10 - m) / factorial(10 + m))
    
    Y = np.zeros((21, 70, 130))
    for idx, m in enumerate(range(-10, 11)):
        if m == 0:
            Y[idx] = N[0] * P[0]
        elif m > 0:
            Y[idx] = np.sqrt(2) * N[m] * P[m] * np.cos(m * phi_grid)
        else:
            am = abs(m)
            Y[idx] = np.sqrt(2) * N[am] * P[am] * np.sin(am * phi_grid)
    
    # Precompute weighted basis for fast evaluation
    # We need to compute g = Y^T (f^3 * w)
    # Let W be the 2D weight array.
    # g_m = sum_{i,j} Y[m,i,j] * f[i,j]^3 * W[i,j]
    # This is equivalent to: g = np.einsum('mij,ij,ij->m', Y, f**3, W)
    
    return Y, weights_2d

def _compute_f(c, Y):
    return np.tensordot(c, Y, axes=(0, 0))

def _objective(c, Y, weights_2d):
    """Compute ||f||_4 / ||f||_2."""
    f = _compute_f(c, Y)
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    return norm4 / norm2

def _power_iteration_step(c, Y, w, damping=1.0):
    """
    One step of the fixed-point iteration for minimizing ||f||_4 / ||f||_2.
    
    The Euler-Lagrange condition for minimizing ||f||_4 subject to ||f||_2 = 1 is:
    f^3 = lambda * f  (in the sense of the variational derivative)
    Actually, minimizing ||f||_4^4 / ||f||_2^4.
    Let R = ||f||_4 / ||f||_2. We want to minimize R.
    At the minimum, the gradient of ||f||_4^4 is parallel to f (in the coefficient space? No, in function space).
    
    Let's look at the gradient of J(c) = ||f||_4^4 / ||f||_2^4.
    This is a Rayleigh quotient for the p=4 case.
    The stationary points satisfy:
    (Y^T (f^3 * w)) / ||f||_4^3  is proportional to c ?
    No, let's derive it properly.
    
    Minimize F(c) = (sum (Y c)^4 w)^{1/4} / (sum (Y c)^2 w)^{1/2}
    Subject to ||c||_2 = 1 (since the ratio is scale invariant, we can fix the norm of c).
    
    Let N4 = (sum f^4 w)^{1/4}, N2 = (sum f^2 w)^{1/2}.
    We want to minimize R = N4 / N2.
    Equivalent to minimizing R^4 = N4^4 / N2^4 = sum(f^4 w) / (sum(f^2 w))^2.
    
    Let S4 = sum(f^4 w) and S2 = sum(f^2 w).
    Minimize Q(c) = S4 / S2^2.
    Gradient of Q:
    dS4/dc = 4 sum(f^3 Y w) = 4 g
    dS2/dc = 2 sum(f Y w) = 2 c  (since Y is orthonormal, sum Y_i Y_j w = delta_ij, so sum f Y_m w = c_m)
    dQ/dc = (dS4 * S2^2 - S4 * d(S2^2)) / S2^4
    d(S2^2)/dc = 2 S2 * dS2/dc = 2 S2 * 2 c = 4 S2 c
    dQ/dc = (4 g S2^2 - S4 * 4 S2 c) / S2^4
           = 4 (g S2 - S4 c / S2) / S2^2  ... wait.
           = 4/S2^2 * (g - (S4/S2) c)
    
    At the optimum, the gradient projected onto the tangent plane of the sphere ||c||=1 must be zero.
    So, proj_tangent(dQ/dc) = 0.
    This implies that (g - (S4/S2) c) is parallel to c.
    Let g = Y^T (f^3 w).
    Then g = mu * c + (S4/S2) c = (mu + S4/S2) c.
    So g is parallel to c.
    
    This means the fixed point iteration is:
    c_new = normalize( g )
    where g = Y^T (f^3 w).
    
    Let's verify. If c is the eigenvector, g = lambda c.
    Then f = Y c.
    g = Y^T (f^3 w).
    Is g parallel to c?
    Yes, if c minimizes the ratio.
    
    So the iteration is simply:
    1. f = Y c
    2. g = Y^T (f^3 w)
    3. c_new = normalize( (1-damping)*c + damping * g )
    
    This is a nonlinear power iteration.
    """
    f = _compute_f(c, Y)
    f3 = f**3
    # g = Y^T (f^3 * w)
    # g_m = sum_{i,j} Y[m,i,j] * f3[i,j] * w[i,j]
    g = np.einsum('mij,ij,ij->m', Y, f3, w)
    
    # Check if g is zero
    norm_g = np.linalg.norm(g)
    if norm_g < 1e-15:
        return c.copy()
    
    # Damped update
    if damping >= 1.0:
        c_new = g / norm_g
    else:
        c_new = (1 - damping) * c + damping * (g / norm_g)
        norm_new = np.linalg.norm(c_new)
        if norm_new < 1e-15:
            return c.copy()
        c_new = c_new / norm_new
        
    return c_new

def _solve_fixed_point(c0, Y, w, max_iter=1000, tol=1e-12, time_limit=10.0):
    """
    Solve the fixed point equation using damped iteration.
    Starts with damping=1.0 (pure power iteration) and reduces damping if it oscillates.
    """
    c_curr = c0 / np.linalg.norm(c0)
    start_time = time.time()
    
    # Adaptive damping
    damping = 1.0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        c_next = _power_iteration_step(c_curr, Y, w, damping)
        
        # Check convergence
        diff = np.linalg.norm(c_next - c_curr)
        if diff < tol:
            c_curr = c_next
            break
            
        # If oscillating (diff doesn't decrease or increases), reduce damping
        if it > 10:
            # Heuristic: if we haven't converged, try damping
            # We can track the objective, but diff is cheaper.
            # If diff is large, we are far. If diff is small but not converging, we might be oscillating.
            # Let's just reduce damping slowly if we are stuck.
            if diff > 1e-6:
                 pass # Still moving
            else:
                 damping = max(0.1, damping * 0.9)
        
        c_curr = c_next
        
    return c_curr

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio
    using nonlinear power iteration (fixed-point) with symmetry-aware multi-restart.
    """
    start_time = time.time()
    deadline = 610.0  # safety margin before 630s limit
    
    Y, weights_2d = _build_harmonics()
    w = weights_2d
    
    best_coeffs = np.ones(21) / np.sqrt(21)
    best_score = _objective(best_coeffs, Y, w)
    
    # Generate initial points
    # The problem has symmetries.
    # 1. Real harmonics.
    # 2. Parity: Y_m(-r) = (-1)^m Y_m(r).
    #    For L4/L2, f^4 and f^2 are even. So the ratio is invariant under f -> -f.
    #    Also, mixing m and -m?
    #    Y_m (cos) and Y_-m (sin).
    #    If we replace f by f rotated in phi, the norm doesn't change.
    #    However, the basis is fixed.
    
    # Strategy:
    # 1. Try the "flat" distribution (all m=0? No, m=0 is P_10^0 which is peaked at poles).
    #    To minimize L4/L2, we want f to be as "flat" as possible.
    #    The "flattest" function on the sphere is a constant, but degree 10 cannot be constant.
    #    We want to spread the energy over as many m's as possible and avoid high peaks.
    #    High |m| harmonics are concentrated at the equator.
    #    Low |m| harmonics are concentrated at the poles.
    #    A mix of all m's might be flattest.
    
    initial_points = []
    
    # 1. All cosines (m >= 0)
    c = np.zeros(21)
    for m in range(0, 11):
        c[10 + m] = 1.0
    initial_points.append(c)
    
    # 2. All sines (m < 0)
    c = np.zeros(21)
    for m in range(-10, 0):
        c[10 + m] = 1.0
    initial_points.append(c)
    
    # 3. All m (cos and sin)
    c = np.ones(21)
    initial_points.append(c)
    
    # 4. Only m=0
    c = np.zeros(21)
    c[10] = 1.0
    initial_points.append(c)
    
    # 5. Only high m (equatorial)
    c = np.zeros(21)
    for m in [8, 9, 10]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    initial_points.append(c)
    
    # 6. Only low m (polar)
    c = np.zeros(21)
    for m in [0, 1, 2]:
        c[10 + m] = 1.0
        if m > 0: c[10 - m] = 1.0
    initial_points.append(c)
    
    # 7. Random with fixed seed
    np.random.seed(42)
    for _ in range(50):
        c = np.random.randn(21)
        initial_points.append(c)
        
    # 8. Sparse random
    for _ in range(20):
        c = np.random.randn(21)
        mask = np.random.random(21) < 0.5
        c *= mask
        if np.linalg.norm(c) < 1e-10:
            c = np.random.randn(21)
        initial_points.append(c)
        
    # Run optimization
    for idx, c0 in enumerate(initial_points):
        if time.time() - start_time > deadline - 60:
            break
            
        c0 = c0 / np.linalg.norm(c0)
        
        try:
            # Use a shorter time limit per restart to allow more restarts
            # The fixed point iteration is very fast (just matrix-vector products essentially)
            c_opt = _solve_fixed_point(c0, Y, w, max_iter=500, tol=1e-14, time_limit=5.0)
            
            # Refine with a longer run if it looks good
            score = _objective(c_opt, Y, w)
            if score < best_score and np.isfinite(score):
                # Check if we should refine
                if time.time() - start_time < deadline - 30:
                    c_opt = _solve_fixed_point(c_opt, Y, w, max_iter=2000, tol=1e-15, time_limit=10.0)
                    score = _objective(c_opt, Y, w)
                
                if score < best_score and np.isfinite(score):
                    best_score = score
                    best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Final refinement on the best solution
    if time.time() - start_time < deadline - 10:
        try:
            c_opt = _solve_fixed_point(best_coeffs, Y, w, max_iter=5000, tol=1e-16, time_limit=20.0)
            score = _objective(c_opt, Y, w)
            if score < best_score and np.isfinite(score):
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Validate
    if len(best_coeffs) != 21 or not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs < 1e-3:
            best_coeffs *= 1.0 / max_abs if max_abs > 0 else 1.0
        elif max_abs > 1e5:
            best_coeffs *= 1e5 / max_abs
        if not np.all(np.isfinite(best_coeffs)):
            return np.ones(21)
    
    return best_coeffs.astype(np.float64)
