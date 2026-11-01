"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

# EVOLVE-BLOCK-START

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_grad(x, grad_euclidean):
    """Project Euclidean gradient to tangent space."""
    return grad_euclidean - np.sum(grad_euclidean * x, axis=1, keepdims=True) * x

def _exp_map_sphere(x, v):
    """Exponential map on unit sphere."""
    theta = np.linalg.norm(v, axis=1, keepdims=True)
    safe_theta = np.where(theta < 1e-10, 1.0, theta)
    safe_v = np.where(theta < 1e-10, v, v / safe_theta)
    x_new = np.cos(theta) * x + np.sin(safe_theta) * safe_v
    zero_mask = (theta < 1e-10)[:, 0]
    x_new[zero_mask] = x[zero_mask]
    x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
    return x_new

def _spherical_harmonics(l, m, theta, phi):
    """
    Compute Y_l^m(theta, phi).
    theta: polar angle [0, pi]
    phi: azimuthal angle [0, 2pi]
    """
    # Associated Legendre polynomial P_l^m(cos(theta))
    # scipy.special.lpmv(m, l, x) computes P_l^m(x)
    # Note: lpmv includes Condon-Shortley phase (-1)^m
    p_lm = lpmv(m, l, np.cos(theta))
    
    # Normalization constant
    # N_lm = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! )
    # Note: lpmv in scipy does not include the normalization, but it does include the Condon-Shortley phase.
    # The standard Y_lm includes the phase.
    # Let's verify normalization.
    # Y_l^m = (-1)^m sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! ) P_l^m(cos theta) e^{im phi}
    # scipy's lpmv(m, l, x) is P_l^m(x) * (-1)^m ? No, scipy docs say:
    # "lpmv(m, v, x) ... The Condon-Shortley phase (-1)^m is included."
    # So P_l^m_scipy = (-1)^m P_l^m_standard.
    # Thus Y_l^m = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! ) P_l^m_scipy e^{im phi}
    
    norm = np.sqrt((2*l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
    
    # Handle complex exponentials
    # e^{im phi}
    exp_term = np.exp(1j * m * phi)
    
    return norm * p_lm * exp_term

def _spectral_energy_and_grad(coeffs, L_max):
    """
    Compute energy of a band-limited function defined by spherical harmonic coefficients.
    coeffs: array of shape (L_max+1, 2*L_max+1) for m in [-L, L].
    Actually, let's use a simpler representation: real and imaginary parts for each l, m.
    Or just use the potential energy of the density.
    
    Energy of a density rho(x) on the sphere:
    E = 1/2 ∫∫ rho(x) rho(y) log(1/|x-y|) dx dy
    
    For a discrete set of points, this is the sum.
    For a continuous density, we can expand log(1/|x-y|) in spherical harmonics.
    The kernel K(x,y) = log(1/|x-y|) has expansion:
    log(1/|x-y|) = -log(2) + sum_{l=0}^∞ (1/(l+0.5)) P_l(cos gamma)  (approx, need exact)
    
    Actually, the expansion of the logarithmic potential on the sphere is:
    log(1/|x-y|) = -log(2) + sum_{l=0}^∞ \frac{1}{l+1/2} P_l(\cos \gamma_{xy})
    Wait, the exact expansion is:
    \frac{1}{|x-y|} = \sum_{l=0}^\infty \frac{2l+1}{|x-y|^{l+1}} P_l(\cos \gamma) for |x|>|y| etc.
    For log, it is:
    \log \frac{1}{|x-y|} = -\log 2 + \sum_{l=0}^\infty \frac{1}{l+1/2} P_l(\cos \gamma_{xy})
    
    Let's verify for l=0: P_0 = 1. Term = 1/0.5 = 2.
    Sum = -log 2 + 2 + ...
    This seems correct. The coefficients are c_l = 1/(l+0.5).
    
    So E = 1/2 ∫∫ rho(x) rho(y) [ -log 2 + sum_l c_l P_l(cos gamma) ] dx dy
    E = -log 2 / 2 * (∫ rho)^2 + 1/2 sum_l c_l ∫∫ rho(x) rho(y) P_l(cos gamma) dx dy
    
    Using addition theorem: P_l(cos gamma) = \frac{4\pi}{2l+1} \sum_{m=-l}^l Y_l^m(x) \overline{Y_l^m(y)}
    
    Then ∫∫ rho(x) rho(y) P_l(cos gamma) dx dy = \frac{4\pi}{2l+1} \sum_{m=-l}^l \left( \int rho(x) Y_l^m(x) dx \right) \left( \int rho(y) \overline{Y_l^m(y)} dy \right)
    = \frac{4\pi}{2l+1} \sum_{m=-l}^l |a_l^m|^2
    where a_l^m = \int rho(x) Y_l^m(x) dx.
    
    So E = -log 2 / 2 * (a_0^0 \sqrt{4\pi})^2 + 1/2 sum_l c_l \frac{4\pi}{2l+1} \sum_{m=-l}^l |a_l^m|^2
    Note a_0^0 = \int rho Y_0^0 dx = \int rho \frac{1}{\sqrt{4\pi}} dx = \frac{1}{\sqrt{4\pi}} \int rho dx.
    So (a_0^0 \sqrt{4\pi})^2 = (\int rho dx)^2.
    
    E = -log 2 / 2 * (M^2) + 2\pi \sum_{l=0}^\infty \frac{c_l}{2l+1} \sum_{m=-l}^l |a_l^m|^2
    
    where M = \int rho dx.
    
    We want to minimize E.
    We represent rho(x) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_l^m Y_l^m(x).
    Then a_l^m = c_l^m (since Y are orthonormal).
    And M = c_0^0 \sqrt{4\pi}.
    
    So E(c) = -log 2 / 2 * (c_0^0 \sqrt{4\pi})^2 + 2\pi \sum_{l=0}^{L} \frac{c_l}{2l+1} \sum_{m=-l}^l |c_l^m|^2
    
    This is a quadratic form in the coefficients!
    E(c) = -2\pi \log 2 (c_0^0)^2 + 2\pi \sum_{l=0}^{L} \frac{1}{(l+0.5)(2l+1)} \sum_{m=-l}^l |c_l^m|^2
    
    Wait, c_l (kernel coeff) = 1/(l+0.5).
    So term is 2\pi \frac{1}{(l+0.5)(2l+1)} |c_l^m|^2.
    
    For l=0: term is 2\pi \frac{1}{0.5 \cdot 1} |c_0^0|^2 = 4\pi |c_0^0|^2.
    First term: -2\pi \log 2 (c_0^0)^2.
    Total for l=0: (4\pi - 2\pi \log 2) (c_0^0)^2.
    Since 4\pi - 2\pi \log 2 > 0, this is positive.
    
    So the spectral energy is convex in the coefficients?
    If so, we can minimize it easily.
    But we also need to ensure rho(x) >= 0.
    And we want to map this to 150 points.
    
    Strategy:
    1. Minimize spectral energy E(c) subject to \int rho = 1 and rho >= 0?
    Minimizing E(c) with fixed mass M=1.
    E(c) = const + sum_{l>=1} ... |c_l^m|^2.
    The l=0 term is fixed if M=1.
    So minimizing E(c) is equivalent to minimizing \sum_{l>=1} |c_l^m|^2.
    This means we want the density to be as "flat" as possible (only l=0 term).
    But a flat density rho=1/4pi gives energy E_flat.
    Is E_flat the minimum?
    Yes, for positive densities, the minimum of the logarithmic energy is achieved by the uniform distribution?
    No, for finite N, the minimum is lower than the continuum limit?
    Actually, for logarithmic potential, the continuum minimum is indeed uniform.
    But we are approximating a discrete sum.
    
    The spectral approach helps by finding a "good" smooth density that approximates the discrete optimum.
    However, the discrete optimum for N=150 is not uniform; it has fluctuations.
    
    Alternative: Use the spectral representation to generate a good initial guess for the 150 points.
    1. Start with uniform density (c_0^0 = 1/sqrt(4pi)).
    2. Perturb the coefficients to create a density with some structure.
    3. Sample 150 points from this density.
    4. Optimize the 150 points.
    
    How to sample from a spherical harmonic density?
    We can use rejection sampling.
    1. Generate uniform points on sphere.
    2. Accept with probability proportional to rho(x).
    3. Need to know max(rho).
    
    Let's try this.
    
    """
    # This function is a placeholder for now, we will implement the sampling in construct
    pass

def _spectral_to_points(coeffs, L, N, rng):
    """
    Sample N points from the density defined by spherical harmonic coefficients.
    coeffs: dict or array of coefficients c_l^m.
    """
    # Rejection sampling
    # 1. Compute max density
    # To do this, we need to evaluate rho(x) on a grid.
    # Grid: 360 x 180 points?
    
    # Let's define the density function
    def rho(x):
        # x: (N_x, 3)
        # Compute theta, phi
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        z = x[:, 2]
        xy = np.sqrt(np.maximum(0, 1 - z*z))
        theta = np.arccos(z)
        phi = np.arctan2(x[:, 1], x[:, 0])
        
        val = np.zeros(len(x))
        for l in range(L + 1):
            for m in range(-l, l + 1):
                c = coeffs.get((l, m), 0.0)
                if c == 0: continue
                y_lm = _spherical_harmonics(l, m, theta, phi)
                val += c * y_lm
        
        # val is complex? Y_lm are complex.
        # We must ensure rho is real.
        # If coeffs are chosen such that c_{l,-m} = (-1)^m conj(c_{l,m}), then rho is real.
        # Let's assume we use real spherical harmonics or enforce this symmetry.
        # For simplicity, let's use real-valued basis?
        # Or just take real part and ensure it's positive.
        
        return np.real(val)
    
    # Evaluate on grid
    n_grid = 360
    theta_grid = np.linspace(0, np.pi, n_grid)
    phi_grid = np.linspace(0, 2*np.pi, n_grid, endpoint=False)
    T, P = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    x_grid = np.column_stack([
        np.sin(T) * np.cos(P),
        np.sin(T) * np.sin(P),
        np.cos(T)
    ]).reshape(-1, 3)
    
    rho_grid = rho(x_grid)
    max_rho = np.max(rho_grid)
    if max_rho <= 0:
        # Fallback to uniform
        x_pts = rng.normal(size=(N, 3))
        x_pts /= np.linalg.norm(x_pts, axis=1, keepdims=True)
        return x_pts
    
    # Rejection sampling
    pts = []
    while len(pts) < N:
        # Generate uniform point
        x_try = rng.normal(size=(1, 3))
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        
        # Evaluate rho
        r_try = rho(x_try)[0]
        
        if r_try < 0:
            continue
            
        u = rng.random()
        if u * max_rho < r_try:
            pts.append(x_try[0])
            
    return np.array(pts)

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _riemannian_cg(x0, max_iter=200, tol=1e-10):
    """Riemannian Conjugate Gradient method for minimizing energy on the sphere."""
    x = x0.copy()
    E, grad_eucl = _log_energy_and_grad_sphere(x)
    g = _riemannian_grad(x, grad_eucl)
    
    p = -g
    p_norm = np.linalg.norm(p, axis=1, keepdims=True)
    p = np.where(p_norm < 1e-15, p, p / p_norm)
    
    best_energy = E
    best_x = x.copy()
    
    for k in range(max_iter):
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        alpha = 1.0
        E_trial = E
        x_trial = x
        grad_eucl_trial = grad_eucl
        for ls_iter in range(30):
            v = alpha * p
            x_trial = _exp_map_sphere(x, v)
            E_trial, grad_eucl_trial = _log_energy_and_grad_sphere(x_trial)
            if E_trial < E:
                break
            alpha *= 0.5
        
        x = x_trial
        g_trial = _riemannian_grad(x, grad_eucl_trial)
        
        if E_trial < best_energy:
            best_energy = E_trial
            best_x = x.copy()
        
        # Parallel transport p to new tangent space
        p = p - np.sum(p * x, axis=1, keepdims=True) * x
        
        # Polak-Ribiere update
        g_diff = g_trial - g
        beta = max(0.0, np.sum(g_trial * g_diff) / (np.sum(g * g) + 1e-15))
        
        g = g_trial
        p = -g + beta * p
        E = E_trial
    
    return best_x, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_energy = float('inf')
    best_x = None
    
    # Phase 1: Spectral Initialization
    # Try to find a good density using spectral coefficients
    # We will try a few random perturbations of the uniform density
    # and sample points from them.
    
    rng = np.random.default_rng(42)
    
    # Uniform density coefficient
    c00 = 1.0 / np.sqrt(4 * np.pi)
    
    # Try a few spectral initializations
    spectral_inits = []
    
    # 1. Pure uniform
    coeffs_uniform = {(0, 0): c00}
    pts_uniform = _spectral_to_points(coeffs_uniform, 0, N, rng)
    spectral_inits.append(pts_uniform)
    
    # 2. Perturbed uniform (low L)
    for trial in range(5):
        coeffs = {(0, 0): c00}
        # Add small random coefficients for l=1,2,3
        for l in range(1, 4):
            for m in range(-l, l+1):
                if m == 0:
                    coeffs[(l, m)] = rng.normal(0, 0.1)
                else:
                    # Enforce reality: c_{l,-m} = (-1)^m conj(c_{l,m})
                    # We only generate for m > 0 and set the conjugate
                    if m > 0:
                        c = rng.normal(0, 0.1) + 1j * rng.normal(0, 0.1)
                        coeffs[(l, m)] = c
                        coeffs[(l, -m)] = (-1)**m * np.conj(c)
        
        pts_spec = _spectral_to_points(coeffs, 3, N, rng)
        spectral_inits.append(pts_spec)
        
    # 3. Fibonacci (as a strong baseline)
    spectral_inits.append(_fibonacci_sphere(N))
    
    # 4. Random
    pts_rand = rng.normal(size=(N, 3))
    pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
    spectral_inits.append(pts_rand)
    
    # Phase 2: Optimize each initialization
    for i, x_init in enumerate(spectral_inits):
        if time.time() > deadline - 20:
            break
        
        try:
            # Fast CG polish
            x_opt, E_opt = _riemannian_cg(x_init, max_iter=100, tol=1e-8)
            
            if E_opt < best_energy:
                best_energy = E_opt
                best_x = x_opt.copy()
                
        except Exception:
            E_init, _ = _log_energy_and_grad_sphere(x_init)
            if E_init < best_energy:
                best_energy = E_init
                best_x = x_init.copy()
                
    if best_x is None:
        best_x = _fibonacci_sphere(N)
        best_energy = _log_energy_and_grad_sphere(best_x)[0]
        
    # Phase 3: Basin Hopping
    if time.time() < deadline - 15:
        for sigma in [0.1, 0.05, 0.02]:
            if time.time() > deadline - 10:
                break
            try:
                v = rng.normal(size=(N, 3))
                v = v - np.sum(v * best_x, axis=1, keepdims=True) * best_x
                v *= sigma
                x_pert = _exp_map_sphere(best_x, v)
                x_opt, E_opt = _riemannian_cg(x_pert, max_iter=50, tol=1e-8)
                
                if E_opt < best_energy:
                    best_energy = E_opt
                    best_x = x_opt.copy()
            except Exception:
                pass
                
    # Final polish
    if time.time() < deadline - 5:
        try:
            x_final, E_final = _riemannian_cg(best_x, max_iter=300, tol=1e-12)
            if E_final < best_energy:
                best_x = x_final
                best_energy = E_final
        except Exception:
            pass
            
    # Ensure non-zero and normalized
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_x = best_x / norms
    
    return best_x

# EVOLVE-BLOCK-END
