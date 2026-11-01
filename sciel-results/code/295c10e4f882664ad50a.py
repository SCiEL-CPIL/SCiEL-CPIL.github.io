"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import math

N = 100

# EVOLVE-BLOCK-START
def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    and its gradient with respect to pts (N x 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    # Avoid log(0) by clipping
    dist_safe = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -math.fsum(np.log(d_upper).tolist())
    
    # Gradient: grad[i] = -sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    
    W = np.zeros_like(dist_sq_safe)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq_safe
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _riemannian_conjugate_gradient(pts, max_iter=500, tol=1e-8):
    """
    Perform Riemannian Conjugate Gradient on the sphere with improved line search.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy, current_grad = _log_energy_and_grad(pts)
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    # Initial direction
    d = -grad_tangent
    
    for k in range(max_iter):
        # Adaptive line search (Backtracking with Wolfe-like conditions)
        alpha = 0.1
        improved = False
        best_alpha = 0.0
        best_energy = current_energy
        best_pts_new = pts.copy()
        best_grad_new = current_grad.copy()
        
        for backtrack in range(40):
            # Step in tangent space
            pts_new = pts + alpha * d
            # Retraction (project to sphere)
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, new_grad = _log_energy_and_grad(pts_new)
            
            if new_energy < current_energy:
                best_alpha = alpha
                best_energy = new_energy
                best_pts_new = pts_new
                best_grad_new = new_grad
                improved = True
            else:
                alpha *= 0.7  # More aggressive backtracking
            
            if alpha < 1e-10:
                break
        
        if not improved:
            break
            
        pts = best_pts_new
        current_energy = best_energy
        current_grad = best_grad_new
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
            
        # Compute new tangent gradient
        grad_dot_new = np.einsum('ij,ij->i', current_grad, pts)
        grad_tangent_new = current_grad - grad_dot_new[:, np.newaxis] * pts
        
        # Fletcher-Reeves beta with damping
        num = np.sum(grad_tangent_new ** 2)
        den = np.sum(grad_tangent ** 2)
        if den < 1e-15:
            beta = 0.0
        else:
            beta = num / den
            beta = min(beta, 1.0)  # Damp beta to avoid divergence
            
        # Transport d to new tangent space
        d_dot_new = np.einsum('ij,ij->i', d, pts)
        d_tangent = d - d_dot_new[:, np.newaxis] * pts
        
        # New direction
        d = -grad_tangent_new + beta * d_tangent
        
        # Update grad_tangent for next iteration's beta calculation
        grad_tangent = grad_tangent_new
        
    return pts, current_energy

def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _golden_ratio_spiral(n=N, start_offset=0.0):
    """Generate points using golden ratio spacing with different starting offset."""
    i = np.arange(n) + start_offset
    z = 1.0 - 2.0 * (i + 0.5) / n
    golden = 2.39996322972865332  # 2*pi / phi^2
    phi_angle = golden * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi_angle), r * np.sin(phi_angle), z])

def _icosahedral_based_init(n=N):
    """Generate initial points based on icosahedral symmetry with perturbations."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=np.float64)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add perturbed Fibonacci points
    fib = _fibonacci_spiral(n)
    noise = np.random.normal(0, 0.05, size=(n, 3))
    pts = fib + noise
    return _project_to_sphere(pts)

def _spherical_caps_init(n=N):
    """
    Generate initial points by dividing the sphere into caps and placing points
    randomly within them, then refining slightly. This helps break symmetries.
    """
    # Create a grid of latitudes and longitudes
    num_lat = 10
    num_lon = 10
    pts = []
    for i in range(num_lat):
        lat = -np.pi/2 + np.pi * (i + 0.5) / num_lat
        for j in range(num_lon):
            lon = 2 * np.pi * (j + 0.5) / num_lon
            # Add random jitter
            lat_j = lat + np.random.normal(0, 0.2)
            lon_j = lon + np.random.normal(0, 0.2)
            x = np.cos(lat_j) * np.cos(lon_j)
            y = np.cos(lat_j) * np.sin(lon_j)
            z = np.sin(lat_j)
            pts.append([x, y, z])
    
    pts = np.array(pts)
    # If we have more or fewer than n, adjust
    if len(pts) > n:
        pts = pts[:n]
    elif len(pts) < n:
        # Add random points
        extra = np.random.normal(size=(n - len(pts), 3))
        extra = _project_to_sphere(extra)
        pts = np.vstack([pts, extra])
        
    return _project_to_sphere(pts)

def _simulated_annealing(pts, max_iter=5000, t_start=0.1, t_end=0.001):
    """
    Perform Simulated Annealing on the sphere to escape local minima.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy, _ = _log_energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    t = t_start
    cool_rate = (t_end / t_start) ** (1.0 / max_iter)
    
    for k in range(max_iter):
        # Generate a random move: pick a random point and perturb it
        idx = np.random.randint(N)
        noise = np.random.normal(0, 0.1, size=(3,))
        new_pts = pts.copy()
        new_pts[idx] = _project_to_sphere(new_pts[idx] + noise)
        
        new_energy, _ = _log_energy_and_grad(new_pts)
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e / t)
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / t):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
                
        t *= cool_rate
        
    return best_pts, best_energy

def _refine_with_lbfgs(pts, max_iter=5000):
    """Refine points using L-BFGS-B in 3D with sphere constraint via penalty."""
    def penalty_energy(x):
        pts = x.reshape(-1, 3)
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        dist = np.sqrt(np.maximum(dist_sq, 1e-30))
        dist_safe = np.maximum(dist, 1e-15)
        
        iu = np.triu_indices(N, k=1)
        energy = -np.sum(np.log(dist_safe[iu]))
        
        # Penalty for being off the sphere
        norms = np.linalg.norm(pts, axis=1)
        penalty = 1e6 * np.sum((norms - 1.0)**2)
        
        return energy + penalty
    
    def penalty_grad(x):
        pts = x.reshape(-1, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        dist_sq_safe = np.maximum(dist_sq, 1e-30)
        
        # Gradient of energy
        W = np.zeros_like(dist_sq_safe)
        np.fill_diagonal(W, 0.0)
        W = 1.0 / dist_sq_safe
        np.fill_diagonal(W, 0.0)
        
        row_sum = W.sum(axis=1)
        grad_energy = -pts * row_sum[:, np.newaxis] + W @ pts
        
        # Gradient of penalty
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        grad_penalty = 2e6 * (norms - 1.0) * pts / norms
        
        return (grad_energy + grad_penalty).flatten()
    
    x0 = pts.flatten()
    res = minimize(penalty_energy, x0, jac=penalty_grad, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    best_pts = _project_to_sphere(res.x.reshape(-1, 3))
    return best_pts

def _perturb_and_refine(pts, noise_scale=0.01, max_iter=2000):
    """Perturb points and refine using RCG."""
    noise = np.random.normal(0, noise_scale, size=(N, 3))
    perturbed = _project_to_sphere(pts + noise)
    res_pts, res_energy = _riemannian_conjugate_gradient(perturbed, max_iter=max_iter, tol=1e-10)
    return res_pts, res_energy

def construct():
    # Multiple initial configurations
    inits = []
    
    # 1. Fibonacci spiral (standard)
    inits.append(_fibonacci_spiral(N, offset=0.5))
    
    # 2. Fibonacci spiral with different offset
    inits.append(_fibonacci_spiral(N, offset=0.0))
    
    # 3. Golden ratio spiral
    inits.append(_golden_ratio_spiral(N, start_offset=0.0))
    
    # 4. Golden ratio spiral with offset
    inits.append(_golden_ratio_spiral(N, start_offset=0.5))
    
    # 5. Perturbed Fibonacci
    noise = np.random.normal(0, 0.1, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise))
    
    # 6. Icosahedral-based
    inits.append(_icosahedral_based_init(N))
    
    # 7. Random points
    inits.append(_project_to_sphere(np.random.normal(size=(N, 3))))
    
    # 8. Rotated Fibonacci
    theta = np.random.uniform(0, 2*np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    inits.append(_project_to_sphere(_fibonacci_spiral(N) @ rot.T))
    
    # 9. Spherical Caps Init
    inits.append(_spherical_caps_init(N))
    
    # 10. Another random perturbation
    noise2 = np.random.normal(0, 0.2, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise2))
    
    best_energy = np.inf
    best_pts = inits[0].copy()
    
    # Run Riemannian CG from each initial configuration
    for idx, x0 in enumerate(inits):
        try:
            # First pass: RCG to get close to a local minimum
            res_pts, res_energy = _riemannian_conjugate_gradient(x0, max_iter=1000, tol=1e-8)
            
            # Second pass: Simulated Annealing to escape local minima
            sa_pts, sa_energy = _simulated_annealing(res_pts, max_iter=2000, t_start=0.05, t_end=0.001)
            
            # Third pass: RCG to polish the SA result
            final_pts, final_energy = _riemannian_conjugate_gradient(sa_pts, max_iter=2000, tol=1e-10)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    # Additional refinement passes from best solution
    for pass_num in range(3):
        try:
            # Use SA again on the best solution found so far
            sa_pts, sa_energy = _simulated_annealing(best_pts, max_iter=2000, t_start=0.02, t_end=0.0005)
            
            # Polish with RCG
            final_pts, final_energy = _riemannian_conjugate_gradient(sa_pts, max_iter=2000, tol=1e-10)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts.copy()
        except Exception:
            pass
            
    # L-BFGS-B refinement (optional, can be slow)
    try:
        refined = _refine_with_lbfgs(best_pts, max_iter=1000)
        refined_energy, _ = _log_energy_and_grad(refined)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_pts = refined
    except Exception:
        pass
    
    # Final fine-tuning
    try:
        res_pts, res_energy = _riemannian_conjugate_gradient(best_pts, max_iter=1000, tol=1e-12)
        if res_energy < best_energy:
            best_energy = res_energy
            best_pts = res_pts
    except Exception:
        pass
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # If points are too close, fall back to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
        
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
        
    return best_pts
# EVOLVE-BLOCK-END
