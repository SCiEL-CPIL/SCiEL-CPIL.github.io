"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient.
    Energy = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2  (since d/dx_i [-log||x_i-x_j||] = (x_i-x_j)/||x_i-x_j||^2)
    Wait, let's re-derive carefully.
    E = - sum_{i<j} log(d_ij) where d_ij = ||x_i - x_j||
    dE/dx_i = - sum_{j != i} d(log d_ij)/dx_i
    d(log d_ij)/dx_i = (1/d_ij) * d(d_ij)/dx_i = (1/d_ij) * (x_i - x_j)/d_ij = (x_i - x_j)/d_ij^2
    So dE/dx_i = - sum_{j != i} (x_i - x_j)/d_ij^2
    This is the gradient of the energy. We want to minimize E, so we move in direction -grad.
    """
    # pts is (N, 3)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero on diagonal
    
    # For energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # We only need upper triangle
    iu = np.triu_indices(N, k=1)
    d = np.sqrt(dist_sq[iu])
    energy = -np.sum(np.log(d))
    
    # Gradient computation
    # grad_i = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # This can be written as: for each i, sum over j != i of (x_i - x_j)/dist_sq_ij
    # Note: (x_i - x_j)/dist_sq_ij = diff[i,j] / dist_sq[i,j]
    # We need to sum over j for each i.
    # grad = - sum_j diff[i,j] / dist_sq[i,j]
    # But dist_sq has 1.0 on diagonal, so we need to zero out the diagonal contribution.
    # Actually, the term for j=i is (x_i - x_i)/1 = 0, so it doesn't matter.
    # So grad = - sum_j (diff[i,j] / dist_sq[i,j])
    
    # diff is (N, N, 3), dist_sq is (N, N)
    # We want to sum over axis 1 (the j index)
    # weighted = diff / dist_sq[:,:,np.newaxis]
    weighted = diff / dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted, axis=1)  # (N, 3)
    
    return energy, grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _golden_spiral():
    """Generate golden spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _gradient_descent(pts, max_iter=5000, tol=1e-10, deadline=None):
    """Adaptive step size gradient descent on sphere."""
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    step_size = 0.01
    min_step = 1e-8
    max_step = 0.1
    patience = 0
    max_patience = 50
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
            
        # Project gradient to tangent space
        # Tangent projection: g_tan = g - (g . x) * x
        dot = np.sum(pts * grad, axis=1, keepdims=True)
        grad_tan = grad - dot * pts
        
        # Adaptive step size
        new_pts = _project_to_sphere(pts - step_size * grad_tan)
        new_energy = _energy_and_grad(new_pts)[0]
        
        if new_energy < energy:
            pts = new_pts
            step_size = min(step_size * 1.5, max_step)
            patience = 0
        else:
            step_size = max(step_size * 0.5, min_step)
            if step_size == min_step:
                break
                
        # Check convergence
        if np.linalg.norm(grad_tan) < tol:
            break
            
    return best_pts, best_energy

def _simulated_annealing(pts, max_iter=2000, deadline=None):
    """Simulated annealing for logarithmic energy minimization."""
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    T = 0.1
    cooling = 0.995
    min_T = 1e-6
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        energy, _ = _energy_and_grad(pts)
        
        # Random perturbation
        idx = np.random.randint(N)
        angle = np.random.uniform(0, 2*np.pi)
        # Random direction tangent to sphere at pts[idx]
        random_vec = np.random.randn(3)
        dot = np.sum(pts[idx] * random_vec)
        tangent = random_vec - dot * pts[idx]
        norm_t = np.linalg.norm(tangent)
        if norm_t < 1e-15:
            continue
        tangent /= norm_t
        
        # Move along tangent
        step = T * 0.1
        new_pts = pts.copy()
        new_pts[idx] = pts[idx] + step * tangent
        new_pts[idx] /= np.linalg.norm(new_pts[idx])
        
        new_energy = _energy_and_grad(new_pts)[0]
        
        # Metropolis criterion
        delta_E = new_energy - energy
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            pts = new_pts
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = pts.copy()
        
        T *= cooling
        if T < min_T:
            break
            
    return best_pts, best_energy

def construct():
    """Construct 100 points on unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate diverse initial configurations
    np.random.seed(42)
    
    # Start with golden spiral
    initial_points = [_golden_spiral()]
    
    # Add random perturbations
    for _ in range(5):
        pts = _golden_spiral()
        noise = np.random.randn(N, 3) * 0.1
        pts = _project_to_sphere(pts + noise)
        initial_points.append(pts)
    
    # Add random configurations
    for _ in range(3):
        pts = np.random.randn(N, 3)
        pts = _project_to_sphere(pts)
        initial_points.append(pts)
    
    best_pts = initial_points[0]
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Run gradient descent on each initial configuration
    for i, init_pts in enumerate(initial_points):
        if time.time() > deadline:
            break
        
        result_pts, result_energy = _gradient_descent(init_pts, max_iter=3000, deadline=deadline)
        
        if result_energy < best_energy:
            best_energy = result_energy
            best_pts = result_pts
    
    # Try simulated annealing on the best result if time permits
    if time.time() < deadline - 20.0:
        sa_pts, sa_energy = _simulated_annealing(best_pts, max_iter=5000, deadline=deadline)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_pts = sa_pts
    
    # Final polish with high-precision gradient descent
    if time.time() < deadline - 5.0:
        final_pts, final_energy = _gradient_descent(best_pts, max_iter=10000, tol=1e-12, deadline=deadline)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(best_pts[i] - best_pts[j])
            if dist < 1e-9:
                # Separate the points
                mid = (best_pts[i] + best_pts[j]) / 2
                mid /= np.linalg.norm(mid)
                perp = np.cross(mid, best_pts[i] - best_pts[j])
                perp /= np.linalg.norm(perp)
                best_pts[i] = _project_to_sphere(best_pts[i] + 1e-8 * perp)
                best_pts[j] = _project_to_sphere(best_pts[j] - 1e-8 * perp)
    
    return best_pts
# EVOLVE-BLOCK-END
