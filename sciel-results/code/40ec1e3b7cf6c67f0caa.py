# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve

def _hermite_powers(k):
    # H_k(x) = (-1)^k e^{x^2/2} d^k/dx^k e^{-x^2/2}
    # H0=1, H2=2x^2-1, H4=4x^4-12x^2+3, H6=8x^6-60x^4+90x^2-15
    # H8=16x^8-320x^6+1680x^4-2400x^2+105
    # H12=32x^12-4800x^10+40320x^8-144000x^6+226800x^4-151200x^2+10395
    # (physicists' convention, matching the seed's c12 = (12c0+128c4+384c8)/10395)
    return np.array([
        1.0,
        2.0*x**2 - 1.0,
        4.0*x**4 - 12.0*x**2 + 3.0,
        8.0*x**6 - 60.0*x**4 + 90.0*x**2 - 15.0,
        16.0*x**8 - 320.0*x**6 + 1680.0*x**4 - 2400.0*x**2 + 105.0,
        32.0*x**12 - 4800.0*x**10 + 40320.0*x**8 - 144000.0*x**6 + 226800.0*x**4 - 151200.0*x**2 + 10395.0,
    ])

def _compute_bound(c):
    # c = (c0, c4, c8); c12 fixed by g(0)=0
    c0, c4, c8 = c
    c12 = (12.0*c0 + 128.0*c4 + 384.0*c8) / 10395.0
    coeff = np.array([c0, c4, c8, c12])
    x = np.linspace(-30.0, 30.0, 200001)
    H = _hermite_powers(x)  # (6, N)
    g = coeff[0]*H[0] + coeff[1]*H[2] + coeff[2]*H[4] + coeff[3]*H[6] + coeff[4]*H[8] + coeff[5]*H[10]
    g2 = g*g
    dx = x[1] - x[0]
    norm = np.trapz(g2, dx=dx)
    if not np.isfinite(norm) or norm <= 0:
        return np.inf
    norm2 = norm*norm
    x2 = np.trapz(x*x*g2, dx=dx) / norm
    # g' = sum coeff_k * H_k' ; H_k' = k H_{k-1}
    gp = (coeff[2]*2.0*H[1]
          + coeff[3]*4.0*(4.0*x**3 - 12.0*x)
          + coeff[4]*8.0*(8.0*x**5 - 60.0*x**3 + 90.0*x)
          + coeff[5]*12.0*(16.0*x**7 - 320.0*x**5 + 1680.0*x**3 - 2400.0*x))
    p2 = np.trapz(gp*gp, dx=dx) / norm
    return x2 * p2

def construct():
    t0 = time.time()
    DEADLINE = 210.0
    MARGIN = 10.0

    # ---- Build the two symmetric quadratic forms in (c0, c4, c8) ----
    # g(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12,  c12 = (12c0+128c4+384c8)/10395
    # Define basis in coefficient space:
    #   b0 = H0 + 12/10395 H12
    #   b4 = H4 + 128/10395 H12
    #   b8 = H8 + 384/10395 H12
    # Then g = c0*b0 + c4*b4 + c8*b8,  norm = <b,b>, x2 = <x^2 b,b>, p2 = <b',b'>
    def _basis_funcs(x):
        H0 = np.ones_like(x)
        H4 = 4.0*x**4 - 12.0*x**2 + 3.0
        H8 = 16.0*x**8 - 320.0*x**6 + 1680.0*x**4 - 2400.0*x**2 + 105.0
        H12 = (32.0*x**12 - 4800.0*x**10 + 40320.0*x**8 - 144000.0*x**6
               + 226800.0*x**4 - 151200.0*x**2 + 10395.0)
        b0 = H0 + 12.0/10395.0*H12
        b4 = H4 + 128.0/10395.0*H12
        b8 = H8 + 384.0/10395.0*H12
        return b0, b4, b8

    def _deriv_funcs(x):
        # b_k' = H_k' + (coeff/10395) H12'
        # H4' = 4*(4x^3 - 12x); H8' = 8*(8x^5 - 60x^3 + 90x)
        # H12' = 12*(16x^7 - 320x^5 + 1680x^3 - 2400x)
        H4p = 4.0*(4.0*x**3 - 12.0*x)
        H8p = 8.0*(8.0*x**5 - 60.0*x**3 + 90.0*x)
        H12p = 12.0*(16.0*x**7 - 320.0*x**5 + 1680.0*x**3 - 2400.0*x)
        b0p = 12.0/10395.0*H12p
        b4p = H4p + 128.0/10395.0*H12p
        b8p = H8p + 384.0/10395.0*H12p
        return b0p, b4p, b8p

    # Numerical integration setup
    N = 200001
    x = np.linspace(-30.0, 30.0, N)
    dx = x[1] - x[0]
    w = np.ones(N) * dx
    w[0] = 0.5 * dx
    w[-1] = 0.5 * dx
    xw = x * w

    b0, b4, b8 = _basis_funcs(x)
    bp0, bp4, bp8 = _deriv_funcs(x)
    B = np.array([b0, b4, b8])      # (3, N)
    BP = np.array([bp0, bp4, bp8])  # (3, N)

    # Gram matrix (norm), x^2-weighted, momentum
    G = B @ (B * w) @ np.eye(1)  # placeholder
    # Properly: G[i,j] = sum_k B[i,k] B[j,k] w_k
    G = (B * w) @ B.T
    X2 = (B * (x*x) * w) @ B.T
    P2 = (BP * w) @ BP.T

    # Quadratic forms: norm = c^T G c, x2 = c^T X2 c / c^T G c, p2 = c^T P2 c / c^T G c
    # Product = (c^T X2 c)(c^T P2 c) / (c^T G c)^2
    # Homogeneous degree 0 in c. Let's find the shape that minimizes this.

    # ---- Generalized eigenvalue approach ----
    # Minimize (c^T X2 c)(c^T P2 c) subject to c^T G c = 1.
    # Using the trick: for the product of two positive forms under a norm constraint,
    # the stationary points satisfy a generalized eigenvalue problem.
    # Specifically, minimize f = (c^T A c)(c^T B c)/(c^T G c)^2.
    # Let u = G^{1/2} c / ||G^{1/2} c|| so that ||u|| = 1.
    # Then f = (u^T A' u)(u^T B' u) where A' = G^{-1/2} A G^{-1/2}, etc.
    # The minimum of (u^T A' u)(u^T B' u) over ||u||=1 is achieved at an eigenvector
    # of A' B' (or equivalently of the generalized system).
    # Actually, the product (u^T A u)(u^T B u) with ||u||=1: the minimum is
    # the smallest eigenvalue of the problem (A+B)u = lambda(A-B)u when A-B is invertible.
    # More robustly: minimize (u^T A u)(u^T B u) over ||u||=1.
    # By Lagrange: gradient = 2(A u (u^T B u) + B u (u^T A u)) = 2 lambda u.
    # This is a nonlinear eigenvalue problem. But we can use the substitution:
    # Let t = u^T B u / u^T A u. Then minimize t * (u^T A u)^2 ... 
    # Alternative: the minimum of (u^T A u)(u^T B u) over ||u||=1 equals
    # the smallest eigenvalue of the generalized eigenvalue problem:
    #   (A + B) v = lambda (A - B) v  ... no, this isn't standard.

    # Let me use a cleaner approach: minimize (c^T X2 c)(c^T P2 c) subject to c^T G c = 1.
    # Scale invariance: let c = s * v where v is a unit vector in G-norm.
    # Then f = (v^T X2 v)(v^T P2 v) / (v^T G v)^2 * (s^4/s^2) = (v^T X2 v)(v^T P2 v)/(v^T G v)^2 * s^2.
    # Wait, that's not right. Let me recompute.
    # norm = c^T G c = s^2 v^T G v
    # x2 = c^T X2 c / norm = s^2 v^T X2 v / (s^2 v^T G v) = v^T X2 v / (v^T G v)
    # p2 = v^T P2 v / (v^T G v)
    # product = (v^T X2 v)(v^T P2 v) / (v^T G v)^2
    # This is scale-invariant! So the shape v determines the product, and scale s is free.
    # But wait, the actual uncertainty product for the scaled function g(x/s) changes with s.
    # The task says the coefficients are the three Hermite coefficients, so the scale is
    # encoded in the coefficients themselves. The product (x2*p2) as computed above IS the
    # uncertainty product for the function g(x) with those coefficients.
    # So I just need to minimize f(v) = (v^T X2 v)(v^T P2 v) / (v^T G v)^2 over v (projective).

    # To find the minimum, use the generalized eigenvalue approach:
    # Let A = G^{-1/2} X2 G^{-1/2}, B = G^{-1/2} P2 G^{-1/2}.
    # Minimize (u^T A u)(u^T B u) over ||u|| = 1.
    # The stationary points satisfy: A u (u^T B u) + B u (u^T A u) = lambda u.
    # Let alpha = u^T A u, beta = u^T B u. Then (alpha B + beta A) u = lambda u.
    # Taking u^T: alpha beta + beta alpha = lambda, so lambda = 2 alpha beta.
    # This is the nonlinear eigenvalue problem. A practical approach:
    # The minimum of (u^T A u)(u^T B u) over ||u||=1 is the smallest eigenvalue of
    # the matrix product A B (when A, B are SPD and commute) or more generally found
    # via the generalized eigenvalue problem.

    # Actually, a cleaner formulation: minimize (u^T A u)(u^T B u) s.t. ||u||^2 = 1.
    # This equals the minimum of t where t >= (u^T A u)(u^T B u), ||u||=1.
    # By AM-GM or direct: the minimum is the smallest eigenvalue of the problem
    #   A B u = lambda u  ... no.
    
    # Let me just use the fact that for the product of two quadratic forms,
    # the minimum over the unit sphere is found by:
    #   min_{||u||=1} (u^T A u)(u^T B u) = min eigenvalue of (A+B) in the metric of (A-B)
    # when A and B are such that A-B is invertible.
    # Specifically, the minimum of (u^T A u)(u^T B u) / (u^T u) over u != 0
    # equals the smallest eigenvalue of the generalized eigenvalue problem:
    #   (A + B) v = lambda (A - B) v
    # This is a known result (see e.g. the analysis of the uncertainty principle).

    # Let me verify: if u^T A u = a, u^T B u = b, then (a+b)(u^T u) = u^T(A+B)u,
    # and (a-b)(u^T u) = u^T(A-B)u. If (A+B)v = lambda(A-B)v, then
    # v^T(A+B)v = lambda v^T(A-B)v, so (a+b) = lambda(a-b), giving a/b = (lambda+1)/(lambda-1).
    # The product ab = (a-b)^2 * (lambda+1)/(lambda-1) * 1/(lambda-1)^2 ... 
    # This is getting complicated. Let me just do a direct numerical optimization.

    # ---- Direct approach: optimize over the projective plane using scipy ----
    # Parameterize v = (1, r, s) / norm, or use spherical coordinates.
    # Actually, let me use the eigenvalue approach properly.
    
    # The minimum of (u^T A u)(u^T B u) over ||u||=1:
    # Define the generalized eigenvalue problem: A u = mu B u.
    # The eigenvalues mu give the ratio (u^T A u)/(u^T B u) at stationary points.
    # At the minimum of the product, we need a specific relation.
    
    # Let me just use scipy.optimize on the 2-parameter spherical representation.
    from scipy.optimize import minimize

    def _objective(theta_phi):
        theta, phi = theta_phi
        # v on unit sphere: v = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
        v = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        va = v @ A @ v
        vb = v @ B @ v
        if va <= 0 or vb <= 0:
            return 1e10
        return va * vb

    # Compute A and B (whitened)
    # G should be SPD (it's a Gram matrix of linearly independent functions)
    try:
        G_chol = np.linalg.cholesky(G)
        G_inv_half = np.linalg.inv(G_chol)
        G_inv_half = G_inv_half.T  # G^{-1/2} = (chol(G))^{-T}
    except np.linalg.LinAlgError:
        # Fallback: use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.maximum(eigvals, 1e-15)
        G_inv_half = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T

    A = G_inv_half @ X2 @ G_inv_half
    B = G_inv_half @ P2 @ G_inv_half

    # Symmetrize
    A = 0.5*(A + A.T)
    B = 0.5*(B + B.T)

    # Try the generalized eigenvalue approach:
    # The minimum of (u^T A u)(u^T B u) over ||u||=1 is the smallest eigenvalue
    # of the matrix M = A @ B (when A, B are SPD and we look at the product).
    # Actually, for SPD A, B, the minimum of (u^T A u)(u^T B u) over ||u||=1
    # equals the smallest eigenvalue of A^{1/2} B A^{1/2} ... no.
    # Let me just use the nonlinear eigenvalue / direct optimization.

    # Multiple starts for robustness
    best_val = np.inf
    best_v = None

    # Initial guesses: the seed direction and a few others
    seed_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_c_norm = seed_c / np.sqrt(seed_c @ G @ seed_c)
    u_seed = G_inv_half @ seed_c_norm
    u_seed = u_seed / np.linalg.norm(u_seed)

    # Convert u to theta, phi
    def _u_to_tp(u):
        z = u[2]
        r = np.sqrt(u[0]**2 + u[1]**2)
        if r < 1e-15:
            return (0.0, 0.0)
        theta = np.arctan2(r, z)
        phi = np.arctan2(u[1], u[0])
        return (theta, phi)

    starts = [_u_to_tp(u_seed)]
    # Add a few random starts
    rng = np.random.default_rng(42)
    for _ in range(8):
        u_rand = rng.standard_normal(3)
        u_rand /= np.linalg.norm(u_rand)
        starts.append(_u_to_tp(u_rand))

    for tp in starts:
        res = minimize(_objective, tp, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-15})
        if res.fun < best_val:
            best_val = res.fun
            best_v = np.array([np.sin(res.x[0])*np.cos(res.x[1]),
                                np.sin(res.x[0])*np.sin(res.x[1]),
                                np.cos(res.x[0])])

    # Also try the generalized eigenvalue approach directly
    # The minimum of (u^T A u)(u^T B u) over ||u||=1:
    # Let C = A + B, D = A - B. If D is invertible, the stationary points of
    # (u^T A u)(u^T B u) / (u^T u) satisfy (A+B)u = lambda(A-B)u.
    # The product at such a point: if u^T A u = a, u^T B u = b, then
    # (a+b) = lambda(a-b), so a/b = (lambda+1)/(lambda-1).
    # Product ab = a^2 * b/a = a^2 * (lambda-1)/(lambda+1).
    # And a = u^T A u, b = u^T B u.
    # This is a standard generalized eigenvalue problem.

    try:
        C = A + B
        D = A - B
        # Check if D is invertible
        if np.linalg.cond(D) < 1e12:
            eigvals_g, eigvecs_g = eigh(C, D)
            for i in range(len(eigvals_g)):
                lam = eigvals_g[i]
                if abs(lam - 1.0) < 1e-10:
                    continue
                v_g = eigvecs_g[:, i]
                v_g = v_g / np.linalg.norm(v_g)
                a = v_g @ A @ v_g
                b = v_g @ B @ v_g
                if a > 0 and b > 0:
                    prod = a * b
                    if prod < best_val:
                        best_val = prod
                        best_v = v_g
    except Exception:
        pass

    # Convert best_v back to coefficients
    # u = G_inv_half @ c_norm, so c_norm = G^{1/2} @ u = G_chol @ u
    c_shape = G_chol @ best_v
    c_shape = c_shape / np.linalg.norm(c_shape)

    # Now optimize the scale. The product for coefficients s*c_shape:
    # norm = s^2 * (c_shape^T G c_shape) = s^2
    # x2 = s^2 * (c_shape^T X2 c_shape) / s^2 = c_shape^T X2 c_shape (scale invariant!)
    # Wait, that means the product is scale-invariant, which can't be right.
    # Let me recompute. For g(x) = c^T H(x):
    # norm = integral |g|^2 = c^T G c
    # x2 = integral x^2 |g|^2 / norm = c^T X2 c / (c^T G c)
    # p2 = integral |g'|^2 / norm = c^T P2 c / (c^T G c)
    # product = (c^T X2 c)(c^T P2 c) / (c^T G c)^2
    # If c -> s c: numerator -> s^4, denominator -> s^4. So product is scale-invariant!
    # This means the uncertainty product does NOT depend on the overall scale of the
    # coefficients. That makes sense because scaling g just scales the function, and
    # the uncertainty product (variance of x * variance of p) is scale-invariant
    # under g -> a*g (it's a shape property, not a magnitude property).
    # So the three coefficients only determine the shape, and the product is
    # determined by the projective class [c0:c4:c8].
    # The scale is irrelevant! So I just need to return the shape.

    # But wait, the seed returns specific values. Let me check if the product
    # really is scale-invariant by computing it for the seed.
    seed_prod = _compute_bound(seed_c)
    # And for a scaled version:
    scaled_prod = _compute_bound(2.0 * seed_c)
    # These should be equal.

    # So the task is to find the shape [c0:c4:c8] that minimizes the product.
    # The returned coefficients can be any representative of the projective class.
    # I'll normalize so that c^T G c = 1 (unit norm) for a canonical choice.

    c_return = c_shape / np.sqrt(c_shape @ G @ c_shape)

    # Verify with the direct computation
    final_prod = _compute_bound(c_return)
    if not np.isfinite(final_prod) or final_prod > seed_prod:
        # Fallback to seed
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])

    return c_return
# EVOLVE-BLOCK-END
