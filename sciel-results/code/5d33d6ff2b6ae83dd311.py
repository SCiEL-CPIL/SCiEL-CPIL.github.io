"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _log_energy_and_grad(points):
    """
    Compute the logarithmic energy and its gradient.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    dists_sq_inv = 1.0 / dists_sq
    # Gradient of E w.r.t x_i is sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    # Note: d/dx_i (-log|x_i-x_j|) = -(x_i-x_j)/|x_i-x_j|^2
    # Wait. E = sum -log(d). dE/dx_i = sum - (1/d) * (x_i-x_j)/d = -sum (x_i-x_j)/d^2.
    # Let's check signs.
    # If points are close, d is small. We want to push them apart.
    # Force on i from j should be in direction (x_i - x_j).
    # Gradient of -log(d) is -(x_i-x_j)/d^2. This points from i to j? No.
    # x_i - x_j points from j to i.
    # So gradient points from j to i. Minimizing moves i in direction -grad = (x_j-x_i)/d^2.
    # This moves i towards j. That's wrong.
    # Energy is -log(d). As d increases, -log(d) decreases. So we want to increase d.
    # Gradient of -log(d) w.r.t x_i:
    # d/dx_i log(|x_i - x_j|) = (x_i - x_j)/|x_i - x_j|^2.
    # So d/dx_i (-log|...|) = -(x_i - x_j)/|x_i - x_j|^2.
    # This vector points from i to j? No, (x_i - x_j) points from j to i.
    # So the negative points from i to j.
    # Minimizing moves in direction of negative gradient.
    # -grad = (x_i - x_j)/d^2. This points from j to i.
    # So i moves away from j. Correct.
    
    grad_raw = -np.sum(diff * dists_sq_inv[:, :, np.newaxis], axis=1)
    
    iu = np.triu_indices(n, k=1)
    log_dists_sq = np.log(dists_sq[iu])
    energy = -0.5 * np.sum(log_dists_sq)
    
    return energy, grad_raw

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, seed=0.0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _sph_harm(l, m, theta, phi):
    """
    Compute spherical harmonic Y_l^m(theta, phi).
    theta: polar angle [0, pi]
    phi: azimuthal angle [0, 2pi]
    """
    if m < 0:
        return 0.0 # We only use m >= 0 for real combinations or complex basis
    # Condon-Shortley phase (-1)^m is included in lpmv
    # Normalization factor
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
    val = norm * lpmv(m, l, np.cos(theta)) * np.exp(1j * m * phi)
    return val

def _get_basis_functions(l_max):
    """
    Generate a set of real spherical harmonic basis functions up to degree l_max.
    We use real harmonics to keep the parameter space real.
    Y_l^m_real = (1/sqrt(2)) * ( (-1)^m Y_l^{-m} + Y_l^m ) for m > 0
    Y_l^0_real = Y_l^0
    Y_l^m_imag = (1/(i sqrt(2))) * ( (-1)^m Y_l^{-m} - Y_l^m ) for m > 0
    """
    # We will construct the matrix of basis values for the N points.
    # We need theta and phi for each point.
    # But the points move! So we can't precompute this easily if we deform the grid.
    # Strategy:
    # 1. Start with a fixed reference grid (Fibonacci).
    # 2. Define a displacement field on the sphere using spherical harmonics.
    # 3. The position of point i is x_i = normalize( x_ref_i + alpha * sum_k c_k Y_k(theta_i, phi_i) )
    #    where Y_k is a vector field? Or scalar field?
    #    If we use a scalar field, we displace radially? No, points must stay on sphere.
    #    We can use a vector field constructed from spherical harmonics.
    #    Simplest: Displace along the normal (radial) and then normalize?
    #    x_new = normalize( x_ref + eps * f(theta, phi) * x_ref )
    #    This is equivalent to scaling the radius by (1 + eps f) and normalizing.
    #    This changes the angular position slightly? No, it stays on the same ray.
    #    This is NOT a good deformation. It doesn't change angles.
    
    # Better: Use a vector field.
    # A divergence-free vector field on the sphere can be constructed from gradients of scalar harmonics.
    # v = grad_theta f, grad_phi f.
    # Or simply: Displace points in the tangent plane.
    # x_new = normalize( x_ref + sum_k c_k v_k(x_ref) )
    # Where v_k are basis vector fields.
    
    # Let's use a simpler approach for the "Spectral Basis" idea:
    # Parameterize the points directly as a perturbation of the Fibonacci grid.
    # But the prompt asks for "spectral basis ... to parameterize point positions as low-frequency deformations".
    # This implies we should optimize the coefficients of the deformation.
    
    # Let's define the deformation as:
    # x_i(c) = normalize( x_ref_i + sum_{l=1}^{L} sum_{m=-l}^{l} c_{lm} Y_{lm}(theta_i, phi_i) * x_ref_i )
    # Wait, multiplying a scalar harmonic by the position vector creates a radial displacement.
    # To get tangential displacement, we need vector harmonics.
    
    # Alternative:
    # Just use the complex spherical harmonics as real and imaginary parts to create 3 sets of basis functions?
    # Actually, let's stick to the simplest effective spectral deformation:
    # We define a map from the reference grid to the sphere.
    # Let's use the fact that any smooth function on the sphere can be approximated by harmonics.
    # Let the displacement of point i be d_i = sum_{k} c_k Y_k(theta_i, phi_i).
    # We add this to the position? No, d_i is a scalar.
    # We need a vector.
    # Let's use 3 scalar fields f_x, f_y, f_z, each expanded in harmonics.
    # x_i(c) = normalize( x_ref_i + (f_x(c, theta_i, phi_i), f_y(...), f_z(...)) )
    # This allows full 3D deformation.
    # The number of coefficients is 3 * (L+1)^2.
    # For L=3, (4)^2 = 16. 3*16 = 48 coefficients.
    # For L=5, (6)^2 = 36. 3*36 = 108 coefficients.
    # This is a manageable dimension for L-BFGS-B.
    
    pass

def _construct_basis_matrix(theta, phi, l_max):
    """
    Construct the matrix of real spherical harmonic values for given theta, phi.
    Returns a matrix of shape (N, 3*(l_max+1)^2).
    Columns correspond to (l, m, component) where component is 0,1,2 for x,y,z displacement fields.
    """
    N = len(theta)
    num_coeffs = 3 * (l_max + 1) ** 2
    basis = np.zeros((N, num_coeffs))
    
    idx = 0
    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            # Calculate complex Y_l^m
            # We need real basis.
            # For m=0: Y_l^0 is real.
            # For m>0: Re(Y_l^m), Im(Y_l^m)
            # For m<0: related to m>0.
            
            # Let's just compute Y for all m and use real/imag parts for m>=0.
            # To avoid redundancy, we generate a basis of real functions.
            # Standard real spherical harmonics:
            # Y_{lm}^c = sqrt(2) (-1)^m Re(Y_l^m) for m > 0
            # Y_{l0}^c = Y_l^0
            # Y_{lm}^s = sqrt(2) (-1)^m Im(Y_l^m) for m > 0
            
            # However, we just need a complete basis. We can use the complex ones and take real/imag separately for m>=0.
            # Let's iterate m from 0 to l.
            pass
            
    # This implementation is getting complex. Let's simplify.
    # We can just use the first few non-zero modes of the complex harmonics, treating real and imaginary parts as separate basis functions.
    # Or, even simpler:
    # Use a Fourier-like expansion in theta and phi? No, spherical harmonics are better.
    
    # Let's implement a robust version.
    # We will generate basis vectors for the displacement field.
    # We'll use l up to l_max.
    # For each l, m in [0, l]:
    #   If m == 0: basis_func = Y_l^0 (real)
    #   If m > 0: basis_func_1 = Re(Y_l^m), basis_func_2 = Im(Y_l^m)
    
    # We need 3 such fields (for x, y, z components of displacement).
    
    col_idx = 0
    for l in range(1, l_max + 1):
        for m in range(0, l + 1):
            # Compute Y_l^m
            # theta, phi are arrays of shape (N,)
            # Y_l^m = N_l * P_l^m(cos(theta)) * e^{i m phi}
            
            # P_l^m(cos(theta))
            p_lm = lpmv(m, l, np.cos(theta))
            
            if m == 0:
                # Real part only
                # Normalization for real harmonics?
                # Let's just use the complex Y and take real/imag.
                # Y = C * p_lm * exp(i*m*phi)
                # For m=0, exp(0)=1, so Y is real (times constant).
                # We need the normalization constant C.
                # C = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
                norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
                y_real = norm * p_lm * np.cos(0) # cos(0)=1
                y_imag = norm * p_lm * np.sin(0) # sin(0)=0
                
                # 3 components (x, y, z displacement)
                basis[:, col_idx] = y_real
                basis[:, col_idx + 1] = y_real
                basis[:, col_idx + 2] = y_real
                col_idx += 3
                
            else:
                # m > 0
                # We use Re(Y) and Im(Y)
                norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
                # Condon-Shortley phase is in lpmv
                y_complex = norm * p_lm * np.exp(1j * m * phi)
                y_real = np.real(y_complex)
                y_imag = np.imag(y_complex)
                
                # For Re(Y): 3 components
                basis[:, col_idx] = y_real
                basis[:, col_idx + 1] = y_real
                basis[:, col_idx + 2] = y_real
                col_idx += 3
                
                # For Im(Y): 3 components
                basis[:, col_idx] = y_imag
                basis[:, col_idx + 1] = y_imag
                basis[:, col_idx + 2] = y_imag
                col_idx += 3
                
    return basis, col_idx

def _spectral_optimize(l_max, deadline, rng):
    """
    Optimize the spectral coefficients to deform the Fibonacci grid.
    """
    # Reference grid
    ref_points = _fibonacci_sphere(N)
    theta = np.arccos(ref_points[:, 2])
    phi = np.arctan2(ref_points[:, 1], ref_points[:, 0])
    
    # Build basis matrix
    basis, num_coeffs = _construct_basis_matrix(theta, phi, l_max)
    
    # Initial coefficients: zero (start from Fibonacci)
    c0 = np.zeros(num_coeffs)
    
    def objective(c):
        # Displacement field
        disp = basis @ c  # shape (N, 3)
        # New points
        pts = ref_points + disp
        pts = _normalize(pts)
        
        # Energy
        energy, grad_euc = _log_energy_and_grad(pts)
        
        # Gradient w.r.t c
        # dE/dc = dE/dpts * dpts/dc
        # dpts/dc is (N, 3, num_coeffs)
        # This is expensive to compute explicitly.
        # We can use finite differences or analytical chain rule.
        # dpts_i/dc_k = d/ dc_k normalize( ref_i + sum_j basis_ij c_j )
        # Let u_i = ref_i + disp_i. pts_i = u_i / |u_i|.
        # d(pts_i)/dc_k = (I - pts_i pts_i^T) * basis_i^k
        # So grad_c = sum_i (grad_pts_i^T * (I - pts_i pts_i^T) * basis_i^k)
        # grad_pts_i is the i-th row of grad_euc.
        
        # Vectorized:
        # grad_euc: (N, 3)
        # basis: (N, num_coeffs)
        # We need to compute for each i: proj_i = (I - pts_i pts_i^T)
        # term_i = proj_i @ grad_euc_i  (vector in tangent space)
        # grad_c[k] = sum_i term_i[k] * basis_i[k]? No.
        # grad_c[k] = sum_i (term_i . basis_i[:, k])
        
        # Compute projection of gradient onto tangent space
        # proj_grad = grad_euc - (grad_euc * pts) * pts
        # (grad_euc * pts) is dot product (N,)
        dots = np.sum(grad_euc * pts, axis=1)
        proj_grad = grad_euc - dots[:, np.newaxis] * pts
        
        # grad_c = basis.T @ proj_grad
        grad_c = basis.T @ proj_grad
        
        return energy, grad_c

    try:
        res = minimize(
            objective,
            c0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if time.time() > deadline:
            return _fibonacci_sphere(N)
            
        c_opt = res.x
        disp = basis @ c_opt
        pts = ref_points + disp
        pts = _normalize(pts)
        
        # Check validity
        min_dist = np.min(np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1)[np.triu_indices(N, k=1)])
        if min_dist < 1e-9:
            return _fibonacci_sphere(N)
            
        return pts
        
    except Exception:
        return _fibonacci_sphere(N)

def _local_minimize(points, deadline):
    """
    Perform local minimization using L-BFGS-B with a penalty method for the sphere constraint.
    """
    n = len(points)
    x0 = points.flatten()
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(n, 3)
        energy, grad_euc = _log_energy_and_grad(x)
        
        # Penalty for deviation from unit sphere
        norms_sq = np.sum(x * x, axis=1)
        penalty = np.sum((norms_sq - 1.0) ** 2)
        grad_penalty = 4.0 * x * (norms_sq - 1.0)[:, np.newaxis]
        
        mu = 50.0
        total_energy = energy + mu * penalty
        total_grad = grad_euc + mu * grad_penalty
        
        return total_energy, total_grad.flatten()

    try:
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 200}
        )
        
        if time.time() > deadline:
            return points

        x_opt = res.x.reshape(n, 3)
        x_opt = _normalize(x_opt)
        
        min_dist = np.min(np.linalg.norm(x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :], axis=-1)[np.triu_indices(n, k=1)])
        if min_dist < 1e-9:
            return points
            
        return x_opt
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    rng = np.random.default_rng(12345)
    
    best_points = None
    best_energy = np.inf
    
    # Try different spectral orders
    for l_max in [3, 5, 7]:
        if time.time() > deadline - 30.0:
            break
            
        # Spectral optimization
        pts_spectral = _spectral_optimize(l_max, deadline, rng)
        
        # Local polish
        pts_polished = _local_minimize(pts_spectral, deadline)
        
        # Evaluate
        try:
            energy, _ = _log_energy_and_grad(pts_polished)
            min_dist = np.min(np.linalg.norm(pts_polished[:, np.newaxis, :] - pts_polished[np.newaxis, :, :], axis=-1)[np.triu_indices(N, k=1)])
            if min_dist >= 1e-9:
                if energy < best_energy:
                    best_energy = energy
                    best_points = pts_polished
        except Exception:
            pass

    # If spectral didn't yield anything good, fallback to standard basin hopping
    if best_points is None or time.time() < deadline - 60.0:
        # Quick basin hopping
        current_points = _fibonacci_sphere(N)
        current_points = _local_minimize(current_points, deadline)
        e, _ = _log_energy_and_grad(current_points)
        
        if e < best_energy:
            best_energy = e
            best_points = current_points
            
        kick_size = 0.2
        while time.time() < deadline - 10.0:
            kicked = current_points + rng.standard_normal((N, 3)) * kick_size
            kicked = _normalize(kicked)
            new_pts = _local_minimize(kicked, deadline)
            
            try:
                new_e, _ = _log_energy_and_grad(new_pts)
                min_dist = np.min(np.linalg.norm(new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :], axis=-1)[np.triu_indices(N, k=1)])
                if min_dist >= 1e-9 and new_e < best_energy:
                    best_energy = new_e
                    best_points = new_pts
                    current_points = new_pts
                    kick_size *= 0.9
                else:
                    kick_size *= 1.1
            except Exception:
                pass

    # Final polish
    if best_points is not None and time.time() < deadline - 5.0:
        final_pts = _local_minimize(best_points, deadline)
        try:
            final_e, _ = _log_energy_and_grad(final_pts)
            min_dist = np.min(np.linalg.norm(final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :], axis=-1)[np.triu_indices(N, k=1)])
            if min_dist >= 1e-9 and final_e < best_energy:
                best_points = final_pts
        except Exception:
            pass

    if best_points is None:
        best_points = _fibonacci_sphere(N)

    return _normalize(best_points)

# EVOLVE-BLOCK-END
