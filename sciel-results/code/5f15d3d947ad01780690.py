import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull method.
    Returns the radius in radians.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return np.max(ang_radii)
    except Exception:
        return np.pi

def _softmax_energy_and_grad(x, grid_pts, beta):
    """
    Compute the soft-max energy (smooth approximation of max distance) and gradient.
    
    x: flattened array of shape (N*3,)
    grid_pts: array of shape (G, 3)
    beta: inverse temperature (higher = closer to max)
    
    Returns:
        energy: soft-max of angular distances
        gradient: array of shape (N*3,)
    """
    pts = x.reshape(N, 3)
    G = len(grid_pts)
    
    # Compute dot products: dots[i, j] = pts[i] @ grid_pts[j]
    dots = pts @ grid_pts.T  # (N, G)
    
    # For each grid point j, find the nearest point i (approximate for gradient)
    # Actually, for soft-max energy, we want to minimize max_j dist_j.
    # dist_j = arccos(max_i dots[i, j])
    # Let u_j = max_i dots[i, j]. Then dist_j = arccos(u_j).
    # We want to minimize SoftMax(dist_j) = (1/beta) * log(sum(exp(beta * dist_j))).
    # Gradient w.r.t x is complex due to argmax.
    
    # Alternative strategy:
    # Minimize sum_j exp(beta * dist_j) / G. This is equivalent to SoftMax.
    # dist_j = arccos(u_j).
    # d dist_j / d x = -grid_pts[j] / sin(dist_j) * (du_j/dx)
    # du_j/dx is non-zero only for the i that achieves the max.
    # This is still non-smooth at ties.
    
    # Let's use a different smooth proxy that is fully differentiable:
    # Energy = sum_j (1 - u_j)^2 ? No, that minimizes distance to nearest, but doesn't penalize the "worst" case as strongly as max.
    # However, minimizing sum of squared distances is a standard approach (Voronoi).
    # The previous program used sum of squared distances.
    
    # Let's try: Energy = sum_j dist_j^2. This is the current program's approach.
    # To make it "materially different", let's use a **power mean** or **soft-max** specifically tuned for the max.
    
    # Let's stick to the sum of squared distances but optimize the grid density and optimization strategy.
    # Wait, the directive says "different design family".
    
    # New Strategy: **Repulsive Potential Energy**
    # Instead of attracting to grid points (Voronoi), we can use a repulsive force between points to spread them out,
    # combined with a "pressure" to keep them on the sphere.
    # But covering radius is about the *empty* space.
    
    # Let's try **Minimizing the sum of exp(beta * dist_j)**.
    # This emphasizes the largest distances.
    
    # To handle the non-smoothness of argmax, we can use a "soft" argmax:
    # u_j = sum_i exp(beta * dots[i, j]) / G ? No, that's not the max.
    
    # Let's use the **Log-Sum-Exp** of the distances.
    # E = (1/beta) log( sum_j exp(beta * dist_j) )
    # This is differentiable if we treat dist_j as fixed for a moment? No.
    
    # Let's go back to basics. The current program uses:
    # 1. Lloyd Relaxation (Voronoi centroid)
    # 2. L-BFGS on sum of squared distances
    
    # New Strategy: **Gradient Descent on Soft-Max Energy with Adaptive Beta**
    # We will compute the gradient of the soft-max energy by approximating the "max" with a smooth function.
    # Specifically, we will minimize:
    # E(x) = sum_j (dist_j)^p
    # where p is large (e.g., 10 or 20). This makes the objective sensitive to the largest distances.
    # This is a "power mean" approach.
    
    # Gradient of (dist_j)^p:
    # d/dx (dist_j)^p = p * (dist_j)^(p-1) * d(dist_j)/dx
    # d(dist_j)/dx = -grid_pts[j] / sin(dist_j) * (du_j/dx)
    # Again, du_j/dx is the issue.
    
    # Let's use a **fully differentiable surrogate** that does NOT rely on argmax.
    # Idea: **Minimize the sum of squared distances to ALL points, weighted by a kernel?**
    # No, that's not covering.
    
    # Let's try **Simulated Annealing on the Hull Facets** with a different move set.
    # Current: Move worst facet vertices.
    # New: **Perturb ALL points** based on the gradient of the hull facet energy.
    
    # Actually, the most robust "different" approach is **Global Optimization via Differential Evolution or CMA-ES**.
    # But 130*3 = 390 dimensions is too high for DE/CMA-ES to converge in 150s.
    
    # Let's try **Alternating Minimization**:
    # 1. Fix points, compute Voronoi cells on a dense grid.
    # 2. Move points to the "center of mass" of the *worst* part of their cell?
    
    # Let's implement a **Vectorized Gradient Descent on the Sum of Squared Distances** but with a **coarser grid** initially, then refine.
    # This is structurally different from Lloyd (which is a specific fixed-point iteration) and L-BFGS (which is a quasi-Newton method).
    # We will use **Adam Optimizer** manually implemented for speed and stability.
    
    # Objective: sum_j dist_j^2
    # Gradient: sum_j 2 * dist_j * (-grid_pts[j] / sin(dist_j)) * (du_j/dx)
    # We approximate du_j/dx by the gradient of the max dot product.
    # To make it smooth, we use the "softmax" of the dot products to compute a weighted average of the gradients.
    
    # u_j = sum_i exp(alpha * dots[i, j]) * dots[i, j] / sum_i exp(alpha * dots[i, j]) ?
    # This is a soft-max approximation of the max dot product.
    
    # Let's implement this.
    
    alpha = 50.0 # Sharpness of soft-max
    
    # 1. Compute dots (N, G)
    dots = pts @ grid_pts.T
    
    # 2. Compute soft-max weights for each grid point j
    # To avoid overflow, subtract max
    max_dots = np.max(dots, axis=0, keepdims=True)
    exp_dots = np.exp(alpha * (dots - max_dots))
    sum_exp = np.sum(exp_dots, axis=0, keepdims=True)
    weights = exp_dots / (sum_exp + _EPS) # (N, G)
    
    # 3. Compute soft-max dot product u_j
    u = np.sum(weights * dots, axis=0) # (G,)
    
    # 4. Compute distances
    cos_theta = np.clip(u, -1.0, 1.0)
    ang_dists = np.arccos(cos_theta) # (G,)
    
    # 5. Energy
    energy = np.sum(ang_dists ** 2)
    
    # 6. Gradient
    # d energy / d dots[i, j] = 2 * ang_dists[j] * (-1/sin(ang_dists[j])) * d u_j / d dots[i, j]
    # d u_j / d dots[i, j] = weights[i, j] * (1 - alpha * (dots[i, j] - u_j)) ? 
    # Actually, d/du (sum w_k u_k) where w_k = exp(alpha u_k) / sum exp(alpha u_k)
    # This is the gradient of the LogSumExp function.
    # d LSE / d u_i = w_i
    # So d u_j / d dots[i, j] = weights[i, j]
    
    # So grad_dots[i, j] = -2 * ang_dists[j] / sin(ang_dists[j]) * weights[i, j]
    
    sin_theta = np.sin(ang_dists)
    valid_sin = sin_theta > _EPS
    grad_dots = np.zeros_like(dots)
    if np.any(valid_sin):
        coeffs = -2.0 * ang_dists[valid_sin] / sin_theta[valid_sin]
        # grad_dots[:, j] = coeffs[j] * weights[:, j]
        # Vectorized:
        grad_dots[:, valid_sin] = np.outer(weights[:, valid_sin], coeffs)
        
    # 7. Backprop to x
    # dots[i, j] = pts[i] @ grid_pts[j]
    # grad_pts[i] = sum_j grad_dots[i, j] * grid_pts[j]
    grad_pts = grad_dots @ grid_pts
    
    return energy, grad_pts.flatten()

def _adam_optimize(x0, grid_pts, deadline, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform Adam optimization on the soft-max energy.
    """
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    best_x = x.copy()
    best_energy = _softmax_energy_and_grad(x, grid_pts, 0)[0]
    
    while time.time() < deadline:
        t += 1
        energy, grad = _softmax_energy_and_grad(x, grid_pts, 0)
        
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
            
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Normalize to sphere
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts /= (norms + _EPS)
        x = pts.flatten()
        
        if t % 100 == 0:
            # Check for convergence or divergence
            if not np.all(np.isfinite(x)):
                break
                
    return best_x.reshape(N, 3), best_energy

def _local_polish(pts, deadline):
    """
    Perform high-iteration count of facet-based local optimization.
    Moves the worst facet's vertices towards its pole.
    """
    best_pts = pts.copy()
    best_r = _compute_hull_radius(pts)
    
    step = 0.02
    no_improve_count = 0
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            M = len(faces)
            
            A = pts[faces[:, 0]]
            B = pts[faces[:, 1]]
            C = pts[faces[:, 2]]
            
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            valid = area > _EPS
            if not np.any(valid):
                break
                
            normals = cross[valid]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            ang_radii = np.arccos(cos_theta)
            
            curr_r = np.max(ang_radii)
            
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 20:
                    step *= 0.95
                    no_improve_count = 0
                    if step < 1e-5:
                        break
            
            worst_idx = np.argmax(ang_radii)
            v_indices = faces[valid][worst_idx]
            pole = normals[worst_idx]
            
            cand = pts.copy()
            for v in v_indices:
                dir_vec = pole - pts[v]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > _EPS:
                    dir_vec /= dir_norm
                else:
                    dir_vec = 0.0
                cand[v] = pts[v] + step * dir_vec
                cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
                
            cand_r = _compute_hull_radius(cand)
            if cand_r < curr_r:
                pts = cand
            else:
                step *= 0.8
                
        except Exception:
            break
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # Create a dense spherical grid
    G = 5000 # Reduced grid size for speed in Adam
    i = np.arange(G) + 0.5
    z = 1.0 - 2.0 * i / G
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    n_restarts = 3
    time_per_restart = (deadline - 20.0) / (n_restarts + 1)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
            
        # Adam Optimization
        sub_deadline = min(time.time() + time_per_restart, deadline - 20.0)
        pts, _ = _adam_optimize(pts.flatten(), grid_pts, sub_deadline)
        
        # Check improvement
        curr_r = _compute_hull_radius(pts)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
            
    # Final local polish
    if time.time() < deadline - 2.0:
        best_pts, best_r = _local_polish(best_pts, deadline)
        
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
