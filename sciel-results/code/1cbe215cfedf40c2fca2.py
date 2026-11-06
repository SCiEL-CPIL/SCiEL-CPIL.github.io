"""
Spherical harmonic density-based optimization for minimal logarithmic energy
of 100 points on the unit sphere.

Approach:
1. Represent a probability density on the sphere via spherical harmonics up to
   degree L. The density is rho(theta, phi) = sum_{l,m} c_{lm} Y_{lm}(theta, phi).
2. The continuous energy is E = ∫∫ log(1/|x-y|) rho(x) rho(y) dΩ_x dΩ_y.
   We minimize this with respect to the coefficients, subject to rho >= 0 and
   ∫ rho dΩ = 1.
3. Place N points by stratified sampling from the optimized density.
4. Refine with direct point optimization using gradient descent.

The log-kernel expansion on the sphere:
  log(1/|x-y|) = -log(2) + sum_{l=1}^∞ (1/l) * (2l+1) * P_l(cos γ)
where γ is the angle between x and y.

This allows us to compute the double integral analytically in terms of the
spherical harmonic coefficients.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial
from math import fsum, log, pi, sqrt
import time

N = 100

def _energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10
            total += log(1.0 / d)
    return total

def _energy_grad(points):
    """Compute gradient of the logarithmic energy with respect to point positions."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            grad[i] -= diff / (d * d)
            grad[j] += diff / (d * d)
    return grad

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=0):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _spherical_harmonic(l, m, theta, phi):
    """Compute the spherical harmonic Y_lm(theta, phi) using Condon-Shortley phase."""
    if m < 0:
        m = -m
        factor = (-1)**m
    else:
        factor = 1.0
    
    # Associated Legendre polynomial
    P = lpmv(m, l, np.cos(theta))
    norm = sqrt((2*l + 1) / (4*pi) * factorial(l-m) / factorial(l+m))
    return factor * norm * P * np.exp(1j * m * phi)

def _build_sh_basis(L, theta, phi):
    """Build the spherical harmonic basis matrix for degrees 0..L."""
    n_theta, n_phi = theta.shape
    coeffs = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            Y = _spherical_harmonic(l, m, theta, phi)
            coeffs.append(Y)
    return np.stack(coeffs, axis=-1)  # shape (n_theta, n_phi, n_coeffs)

def _sh_coefficients_to_density(c, theta, phi, L):
    """Compute density from spherical harmonic coefficients."""
    basis = _build_sh_basis(L, theta, phi)
    # c is complex array of shape (n_coeffs,)
    density = np.real(np.sum(basis * c[np.newaxis, np.newaxis, :], axis=2))
    return density

def _density_energy(c, theta, phi, L, log_kernel_matrix):
    """Compute the continuous energy integral ∫∫ log(1/|x-y|) rho(x) rho(y) dΩ_x dΩ_y."""
    n_theta, n_phi = theta.shape
    density = _sh_coefficients_to_density(c, theta, phi, L)
    density = np.maximum(density, 0.0)  # enforce non-negativity
    
    # Normalize
    dtheta = np.diff(np.linspace(0, pi, n_theta + 1))
    dphi = np.diff(np.linspace(0, 2*pi, n_phi + 1))
    sin_theta = np.sin(theta[:, 0])  # shape (n_theta,)
    weight = sin_theta[:, np.newaxis] * dtheta[:, np.newaxis] * dphi[np.newaxis, :]
    
    norm = np.sum(density * weight)
    if norm < 1e-15:
        return 1e10
    density = density / norm
    
    # Energy = ∫∫ K(x,y) rho(x) rho(y) dΩ_x dΩ_y
    # = sum_{i,j} K[i,j] * density[i] * weight[i] * density[j] * weight[j]
    # where K[i,j] = log(1/|x_i - y_j|)
    energy = np.sum(log_kernel_matrix * density[np.newaxis, :, :] * weight[np.newaxis, :, :] * 
                    density[:, np.newaxis, :] * weight[:, np.newaxis, :])
    return energy

def _density_energy_grad(c, theta, phi, L, log_kernel_matrix):
    """Compute gradient of the continuous energy with respect to SH coefficients."""
    n_theta, n_phi = theta.shape
    n_coeffs = (L + 1) ** 2
    
    density = _sh_coefficients_to_density(c, theta, phi, L)
    density = np.maximum(density, 0.0)
    
    dtheta = np.diff(np.linspace(0, pi, n_theta + 1))
    dphi = np.diff(np.linspace(0, 2*pi, n_phi + 1))
    sin_theta = np.sin(theta[:, 0])
    weight = sin_theta[:, np.newaxis] * dtheta[:, np.newaxis] * dphi[np.newaxis, :]
    
    norm = np.sum(density * weight)
    if norm < 1e-15:
        return np.zeros(n_coeffs, dtype=complex)
    density = density / norm
    
    # For gradient, we need dE/dc_k where c_k are the complex coefficients
    # E = sum_{i,j} K[i,j] rho_i rho_j w_i w_j / (sum rho w)^2
    # This is complex; use finite differences for simplicity
    
    grad = np.zeros(n_coeffs, dtype=complex)
    eps = 1e-7
    for k in range(n_coeffs):
        c_plus = c.copy()
        c_plus[k] += eps
        c_minus = c.copy()
        c_minus[k] -= eps
        e_plus = _density_energy(c_plus, theta, phi, L, log_kernel_matrix)
        e_minus = _density_energy(c_minus, theta, phi, L, log_kernel_matrix)
        grad[k] = (e_plus - e_minus) / (2 * eps)
    
    return grad

def _optimize_density(L, maxiter=100, time_limit=30.0):
    """Optimize spherical harmonic coefficients to minimize continuous log-energy."""
    # Quadrature grid
    n_theta = 64
    n_phi = 128
    theta = np.linspace(0, pi, n_theta)[:, np.newaxis] * np.ones((1, n_phi))
    phi = np.linspace(0, 2*pi, n_phi)[:, np.newaxis] * np.ones((n_theta, 1))
    
    # Build log-kernel matrix
    # Convert to Cartesian for distance computation
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = cos_t
    
    # Distance matrix (vectorized)
    # This is memory intensive; use chunked computation
    chunk_size = 16
    log_kernel = np.zeros((n_theta * n_phi, n_theta * n_phi))
    
    for start in range(0, n_theta * n_phi, chunk_size):
        end = min(start + chunk_size, n_theta * n_phi)
        x_chunk = x[start:end].flatten()
        y_chunk = y[start:end].flatten()
        z_chunk = z[start:end].flatten()
        
        x_all = x.flatten()
        y_all = y.flatten()
        z_all = z.flatten()
        
        dx = x_chunk[:, np.newaxis] - x_all[np.newaxis, :]
        dy = y_chunk[:, np.newaxis] - y_all[np.newaxis, :]
        dz = z_chunk[:, np.newaxis] - z_all[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        np.maximum(dist, 1e-15, out=dist)
        log_kernel[start:end, :] = np.log(1.0 / dist)
    
    # Initial coefficients: uniform density (only l=0, m=0 nonzero)
    n_coeffs = (L + 1) ** 2
    c = np.zeros(n_coeffs, dtype=complex)
    c[0] = 1.0  # Y_00 = 1/sqrt(4*pi)
    
    # Use L-BFGS-B on the real and imaginary parts
    def _to_real_complex(params):
        n_half = n_coeffs // 2
        real = params[:n_half]
        imag = params[n_half:n_half + n_coeffs]
        return real + 1j * imag
    
    def _from_real_complex(c):
        return np.concatenate([c.real, c.imag])
    
    params0 = _from_real_complex(c)
    
    def _objective(params):
        c_curr = _to_real_complex(params)
        return _density_energy(c_curr, theta, phi, L, log_kernel)
    
    def _grad(params):
        c_curr = _to_real_complex(params)
        g = _density_energy_grad(c_curr, theta, phi, L, log_kernel)
        return np.concatenate([g.real, g.imag])
    
    start_time = time.time()
    result = minimize(_objective, params0, method='L-BFGS-B', jac=_grad,
                      options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6,
                               'maxfun': 5000})
    
    best_c = _to_real_complex(result.x)
    return best_c, theta, phi, L

def _sample_from_density(c, theta, phi, L, N_samples, seed=42):
    """Sample N points from the density using rejection sampling."""
    rng = np.random.RandomState(seed)
    
    n_theta, n_phi = theta.shape
    density = _sh_coefficients_to_density(c, theta, phi, L)
    density = np.maximum(density, 0.0)
    
    dtheta = np.diff(np.linspace(0, pi, n_theta + 1))
    dphi = np.diff(np.linspace(0, 2*pi, n_phi + 1))
    sin_theta = np.sin(theta[:, 0])
    weight = sin_theta[:, np.newaxis] * dtheta[:, np.newaxis] * dphi[np.newaxis, :]
    
    pdf = density * weight
    total = np.sum(pdf)
    if total < 1e-15:
        return _fibonacci_spiral()
    pdf = pdf / total
    
    # Flatten and sample
    pdf_flat = pdf.flatten()
    max_pdf = np.max(pdf_flat)
    
    points = []
    attempts = 0
    max_attempts = N_samples * 100
    
    while len(points) < N_samples and attempts < max_attempts:
        # Sample from uniform grid
        i = rng.randint(n_theta)
        j = rng.randint(n_phi)
        
        if rng.uniform() < pdf_flat[i * n_phi + j] / max_pdf:
            # Accept this grid point; add some jitter
            theta_val = (i + rng.uniform()) * (pi / n_theta)
            phi_val = (j + rng.uniform()) * (2 * pi / n_phi)
            
            x = np.sin(theta_val) * np.cos(phi_val)
            y = np.sin(theta_val) * np.sin(phi_val)
            z = np.cos(theta_val)
            points.append([x, y, z])
        
        attempts += 1
    
    if len(points) < N_samples:
        # Fill remaining with Fibonacci
        fib = _fibonacci_spiral()
        for i in range(len(points), N_samples):
            points.append(fib[i])
    
    return np.array(points[:N_samples])

def _refine_points(points, maxiter=500):
    """Refine points using gradient descent on the sphere."""
    pts = points.copy().astype(np.float64)
    
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    
    def _spherical_to_cart(theta, phi):
        sin_t = np.sin(theta)
        return np.column_stack([
            sin_t * np.cos(phi),
            sin_t * np.sin(phi),
            np.cos(theta)
        ])
    
    def _objective(params):
        th = params[:N]
        ph = params[N:]
        p = _spherical_to_cart(th, ph)
        return _energy(p)
    
    def _grad(params):
        th = params[:N]
        ph = params[N:]
        p = _spherical_to_cart(th, ph)
        g_cart = _energy_grad(p)
        
        sin_t = np.sin(th)
        cos_t = np.cos(th)
        sin_p = np.sin(ph)
        cos_p = np.cos(ph)
        
        dtheta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
        g_th = np.sum(g_cart * dtheta, axis=1)
        
        dphi = np.column_stack([-sin_p, cos_p, np.zeros(N)])
        g_ph = np.sum(g_cart * dphi, axis=1)
        
        return np.concatenate([g_th, g_ph])
    
    params0 = np.concatenate([theta, phi])
    
    result = minimize(_objective, params0, method='L-BFGS-B',
                      jac=_grad, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    best_params = result.x
    best_points = _spherical_to_cart(best_params[:N], best_params[N:])
    best_points = _project_to_sphere(best_points)
    
    return best_points

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    Uses spherical harmonic density optimization followed by point refinement.
    """
    start_time = time.time()
    timeout = 110.0
    
    best_points = None
    best_energy = np.inf
    
    # Try different SH degrees
    for L in [6, 8, 10]:
        if time.time() - start_time > timeout:
            break
        
        try:
            c, theta, phi, L_used = _optimize_density(L, maxiter=80, time_limit=40.0)
            
            # Sample points from the optimized density
            for seed in [42, 123, 999]:
                if time.time() - start_time > timeout:
                    break
                
                pts = _sample_from_density(c, theta, phi, L_used, N, seed=seed)
                pts = _project_to_sphere(pts)
                
                # Refine
                if time.time() - start_time > timeout - 20:
                    refined = pts
                else:
                    refined = _refine_points(pts, maxiter=300)
                    refined = _project_to_sphere(refined)
                
                e = _energy(refined)
                if e < best_energy:
                    best_energy = e
                    best_points = refined.copy()
        except Exception:
            continue
    
    # Fallback: direct optimization from Fibonacci
    if time.time() - start_time < timeout - 30:
        try:
            pts = _fibonacci_spiral()
            refined = _refine_points(pts, maxiter=500)
            refined = _project_to_sphere(refined)
            e = _energy(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = _fibonacci_spiral()
    
    # Final validation
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-9) or not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_spiral()
    
    assert best_points.shape == (N, 3)
    return best_points
