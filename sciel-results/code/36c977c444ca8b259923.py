"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math

def _energy_and_grad(points):
    """
    Compute logarithmic energy and gradient.
    Energy: sum_{i<j} log(1/|xi-xj|)
    Gradient: sum_{j!=i} (xi-xj) / |xi-xj|^2
    """
    n = points.shape[0]
    # Compute all pairwise differences
    # points: (n, 3)
    # diff: (n, n, 3) where diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    
    # Compute distances
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Distances
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = -0.5 * sum_{i!=j} log(dist)
    # log(1/d) = -log(d)
    log_dist = np.log(dist)
    np.fill_diagonal(log_dist, 0.0)
    
    energy = -0.5 * np.sum(log_dist)
    
    # Gradient
    # grad_i = sum_{j!=i} (xi - xj) / dist_ij^2
    # term[i,j] = (xi-xj) / dist_ij^2
    # Note: dist_sq is 0 on diagonal, we set it to 1, but we must zero out the diagonal contribution
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff[i,j] = points[i] - points[j]
    # We need to sum over j for each i.
    # grad = sum_{j} (points[i] - points[j]) / dist_ij^2
    # This is equivalent to:
    # grad = points * sum_j(inv_dist_sq[i,j]) - sum_j(points[j] * inv_dist_sq[i,j])
    
    # sum_inv_dist_sq: (n,)
    sum_inv = np.sum(inv_dist_sq, axis=1)
    
    # weighted_points: (n, 3)
    # weighted_points[i] = sum_j points[j] * inv_dist_sq[i,j]
    weighted_points = np.sum(inv_dist_sq[:, :, np.newaxis] * points[np.newaxis, :, :], axis=1)
    
    grad = points * sum_inv[:, np.newaxis] - weighted_points
    
    return energy, grad

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _project_grad_to_tangent(points, grad):
    """Project gradient to tangent space of the sphere."""
    # grad_tangent = grad - (grad . p) p
    dots = np.sum(grad * points, axis=1, keepdims=True)
    return grad - dots * points

def _initial_fibonacci():
    """Generate initial Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_random(seed=0):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _simulated_annealing(points, T_start=0.1, T_end=1e-4, n_iter=5000, seed=42):
    """
    Simulated Annealing for global optimization.
    """
    rng = np.random.default_rng(seed)
    pts = points.copy()
    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    for i in range(n_iter):
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (i / n_iter)
        
        # Generate random perturbations
        # Perturb each point in tangent space
        perturbations = rng.normal(size=(N, 3))
        # Project to tangent space
        dots = np.sum(perturbations * pts, axis=1, keepdims=True)
        perturbations = perturbations - dots * pts
        
        # Scale perturbations
        step_size = 0.05
        new_pts = pts + step_size * perturbations
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, _ = _energy_and_grad(new_pts)
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        # Occasionally re-center or refine? No, just SA.
        
    return best_pts, best_energy

def _conjugate_gradient(points, max_iter=1000, tol=1e-10):
    """
    Conjugate Gradient method on the sphere manifold.
    """
    pts = points.copy()
    energy, grad = _energy_and_grad(pts)
    grad_tan = _project_grad_to_tangent(pts, grad)
    
    # Initial direction is negative gradient
    d = -grad_tan
    # Normalize direction for stability? Not strictly necessary for CG, but helps.
    
    for k in range(max_iter):
        # Check convergence
        norm_grad = np.linalg.norm(grad_tan)
        if norm_grad < tol:
            break
        
        # Line search: find alpha such that energy(pts + alpha * d) is minimized
        # Use backtracking line search
        alpha = 1.0
        # Ensure d is in tangent space
        d_tan = _project_grad_to_tangent(pts, d)
        
        # If d is zero, restart
        if np.linalg.norm(d_tan) < 1e-12:
            d = -grad_tan
            d_tan = d
            continue
            
        # Backtracking
        for _ in range(20):
            new_pts = pts + alpha * d_tan
            new_pts = _project_to_sphere(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Armijo condition
            if new_energy <= energy + 1e-4 * alpha * np.sum(grad_tan * d_tan):
                break
            alpha *= 0.5
        else:
            # If line search fails, shrink step
            new_pts = pts + 0.01 * d_tan
            new_pts = _project_to_sphere(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)
        
        # Update points
        pts = new_pts
        energy = new_energy
        grad = new_grad
        grad_tan = _project_grad_to_tangent(pts, grad)
        
        # Conjugate gradient update
        # beta = (grad_new_tan . grad_new_tan) / (grad_old_tan . grad_old_tan)
        # Use Polak-Ribiere or Fletcher-Reeves
        # Fletcher-Reeves: beta = ||g_new||^2 / ||g_old||^2
        # Polak-Ribiere: beta = (g_new . (g_new - g_old)) / ||g_old||^2
        # Let's use Polak-Ribiere with non-negative constraint
        
        g_old_tan = grad_tan
        # We need to store previous grad_tan
        # Actually, we updated grad_tan in place? No, we reassigned it.
        # We need the previous one.
        
        # Let's restructure to keep track of g_k and g_{k+1}
        # This simple loop above is slightly messy for CG update. Let's do a proper CG loop.
        
    # The above loop is a simplified/stochastic CG. Let's implement a proper one below.
    
    return pts, energy

def _conjugate_gradient_proper(points, max_iter=2000, tol=1e-10):
    """
    Proper Conjugate Gradient method on the sphere.
    """
    pts = points.copy()
    energy, grad = _energy_and_grad(pts)
    g = _project_grad_to_tangent(pts, grad)
    
    d = -g
    d = _project_grad_to_tangent(pts, d)
    
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
            
        # Line search
        alpha = 1.0
        for _ in range(30):
            new_pts = pts + alpha * d
            new_pts = _project_to_sphere(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)
            new_g = _project_grad_to_tangent(new_pts, new_grad)
            
            # Armijo
            if new_energy <= energy + 1e-4 * alpha * np.sum(g * d):
                break
            alpha *= 0.5
            
        if alpha < 1e-12:
            break
            
        # Update
        pts = new_pts
        energy = new_energy
        grad = new_grad
        g_new = new_g
        
        # CG update (Polak-Ribiere)
        # beta = max(0, (g_new - g) . g_new) / (g . g)
        num = np.sum((g_new - g) * g_new)
        den = np.sum(g * g)
        
        if den > 1e-15:
            beta = max(0.0, num / den)
        else:
            beta = 0.0
            
        d = -g_new + beta * d
        d = _project_grad_to_tangent(pts, d)
        
        # Check if d is orthogonal to g? No, CG property.
        # If d is too small, restart
        if np.linalg.norm(d) < 1e-12:
            d = -g_new
            d = _project_grad_to_tangent(pts, d)
            
        g = g_new
        
    return pts, energy

def construct():
    best_pts = None
    best_energy = float('inf')
    
    # Multi-start strategy
    # 1. Fibonacci start
    init_fib = _initial_fibonacci()
    
    # 2. Random starts
    inits = [init_fib]
    for s in range(1, 5):
        inits.append(_initial_random(seed=s))
    
    for init in inits:
        # Simulated Annealing
        try:
            sa_pts, sa_energy = _simulated_annealing(init, T_start=0.05, T_end=1e-5, n_iter=2000, seed=123)
            
            # Conjugate Gradient Refinement
            cg_pts, cg_energy = _conjugate_gradient_proper(sa_pts, max_iter=500)
            
            if cg_energy < best_energy:
                best_energy = cg_energy
                best_pts = cg_pts.copy()
                
        except Exception as e:
            pass
    
    # Final check and projection
    if best_pts is not None:
        best_pts = _project_to_sphere(best_pts)
        
        # Ensure no points are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < 1e-9:
                    # Perturb
                    rng = np.random.default_rng(i*100+j)
                    best_pts[j] += 1e-6 * rng.randn(3)
                    best_pts = _project_to_sphere(best_pts)
                    
    return best_pts
# EVOLVE-BLOCK-END
