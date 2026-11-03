import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    # We formulate the packing problem as a spectral optimization.
    # Let M be a symmetric N x N matrix.
    # Let V be the matrix whose columns are the top N eigenvectors of M.
    # The sphere centers are the rows of V, scaled and shifted to fit in [-0.5, 0.5]^3.
    # However, eigenvectors are in R^N. We need to map them to R^3.
    # The directive says: "representing the sphere centers as the top N eigenvectors".
    # This implies the coordinates of the spheres are the components of the eigenvectors?
    # If we take the first 3 components of each eigenvector, we get points in R^3.
    # Let's refine this:
    # 1. Generate a random symmetric matrix M.
    # 2. Compute eigenvectors v_1, ..., v_N.
    # 3. The center of sphere i is given by (v_i[0], v_i[1], v_i[2])?
    #    This would only use 3 dimensions of the N-dimensional eigenvector space.
    #    This seems to discard information.
    #    
    # Alternative Interpretation:
    # Maybe the "spectral" aspect refers to using a random projection?
    # Or perhaps we optimize M such that the Gram matrix of the projected points has good properties?
    #
    # Let's stick to the most direct interpretation of "centers as top N eigenvectors":
    # We define the centers $c_i \in \mathbb{R}^3$ as the first 3 components of the i-th eigenvector of a symmetric matrix $M \in \mathbb{R}^{N \times N}$.
    # $c_i = [v_i^{(1)}, v_i^{(2)}, v_i^{(3)}]$ where $v_i$ is the i-th eigenvector.
    # We then scale these centers to fit in the cube.
    # We optimize the entries of $M$ to maximize the minimum distance between these projected points.
    
    # To make this efficient, we parameterize M by its upper triangle.
    # We use L-BFGS-B to minimize a smooth function of the "badness" of the packing.
    
    def get_centers_from_M(M_flat):
        """
        Reconstruct symmetric M from flat vector, compute eigenvectors,
        extract first 3 components of each eigenvector, scale to cube.
        """
        M = np.zeros((N, N))
        # Fill upper triangle
        idx = np.triu_indices(N)
        M[idx] = M_flat
        M[(idx[1], idx[0])] = M_flat
        
        # Compute eigenvectors
        # We only need the top N eigenvectors, which is all of them.
        # np.linalg.eigh returns eigenvalues in ascending order.
        # We want the "top" ones? Usually top means largest eigenvalue.
        # The set of eigenvectors is orthonormal regardless of order.
        # Let's take the eigenvectors corresponding to the largest eigenvalues?
        # Actually, the set of all N eigenvectors forms a basis.
        # If we take the first 3 components of ALL eigenvectors, we get N points in R^3.
        
        eigvals, eigvecs = np.linalg.eigh(M)
        
        # Extract first 3 components for each eigenvector
        # eigvecs is N x N. Column j is eigenvector j.
        # So rows are components.
        # We want a N x 3 matrix where row i is the first 3 components of eigenvector i.
        # eigvecs[:, :3] gives a N x 3 matrix where row i is (v_i[0], v_i[1], v_i[2]).
        
        centers = eigvecs[:, :3]
        
        # Scale to fit in [-0.5, 0.5]
        # Find the max absolute value to scale
        max_abs = np.max(np.abs(centers))
        if max_abs < 1e-9:
            # Degenerate case, add small noise
            centers = np.random.normal(0, 1, centers.shape)
            max_abs = np.max(np.abs(centers))
            
        centers = centers / max_abs * 0.45 # Scale to fit with margin
        
        return centers

    def objective(M_flat):
        """
        Objective function to minimize.
        We want to maximize the minimum distance.
        We minimize a smooth approximation: sum of 1/d^p for close pairs.
        """
        centers = get_centers_from_M(M_flat)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e-9)
        d = np.sqrt(d2)
        
        # We want to penalize small distances.
        # Use a power law: sum 1/d^4
        # Only consider off-diagonal
        mask = ~np.eye(N, dtype=bool)
        d_off = d[mask]
        
        # To avoid overflow/instability if d is very small, clip d
        d_safe = np.maximum(d_off, 1e-6)
        penalty = np.sum(1.0 / (d_safe ** 4))
        
        # Also penalize if centers are too close to walls?
        # The scaling ensures they are within 0.45, so wall distance is at least 0.05.
        # But we want to maximize the radius, which is min(d/2, 0.5 - max_coord).
        # Since we scaled to 0.45, the wall distance is fixed relative to the scale?
        # No, the relative positions change.
        # The radius is limited by min(d_min/2, 0.5 - max(|c_ik|)).
        # Our scaling forces max(|c_ik|) <= 0.45.
        # So wall distance is >= 0.05.
        # Inter-sphere distance d_min determines r if d_min/2 < 0.05.
        # If d_min/2 > 0.05, then r is limited by wall.
        # To maximize r, we should ideally make d_min/2 and wall_dist equal.
        # But for now, maximizing d_min is a good proxy.
        
        return penalty

    def gradient(M_flat):
        """
        Numerical gradient for L-BFGS-B.
        """
        eps = 1e-6
        grad = np.zeros_like(M_flat)
        f0 = objective(M_flat)
        
        for i in range(len(M_flat)):
            M_plus = M_flat.copy()
            M_plus[i] += eps
            f_plus = objective(M_plus)
            
            M_minus = M_flat.copy()
            M_minus[i] -= eps
            f_minus = objective(M_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
            
        return grad

    # Initial M: Random symmetric matrix
    np.random.seed(42)
    M_init = np.random.randn(N, N)
    M_init = (M_init + M_init.T) / 2
    M_flat_init = M_init[np.triu_indices(N)]
    
    # Optimize
    start_time = time.time()
    time_limit = 100.0
    
    # Use L-BFGS-B
    # maxiter is limited by time
    res = minimize(
        objective, 
        M_flat_init, 
        method='L-BFGS-B', 
        jac=gradient,
        options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if time.time() - start_time > time_limit:
        # If we ran out of time, use the best result so far?
        # minimize doesn't easily return intermediate results.
        # We'll just use res.x if it exists.
        pass
        
    M_opt = np.zeros((N, N))
    idx = np.triu_indices(N)
    M_opt[idx] = res.x
    M_opt[(idx[1], idx[0])] = res.x
    
    centers = get_centers_from_M(res.x)
    
    # Final polish: Direct coordinate optimization on the resulting centers
    # This corrects any distortions from the spectral projection.
    
    def energy_and_grad_direct(x):
        centers = x
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e-9)
        d = np.sqrt(d2)
        d3 = d ** 3
        np.fill_diagonal(d3, 1e-9)
        
        force_spheres = np.sum(diff / d3[:, :, np.newaxis], axis=1)
        
        # Wall forces
        K_wall = 100.0
        force_walls = np.zeros_like(centers)
        for k in range(3):
            c = centers[:, k]
            dist_pos = np.maximum(0.5 - c, 1e-6)
            dist_neg = np.maximum(c + 0.5, 1e-6)
            f_pos = -K_wall / (dist_pos ** 2)
            f_neg = K_wall / (dist_neg ** 2)
            force_walls[:, k] = f_pos + f_neg
            
        total_force = force_spheres + force_walls
        
        # Calculate radius
        d_min = np.min(d[np.triu_indices(N, k=1)])
        w_min = np.min(np.abs(0.5 - np.abs(centers)))
        r_current = min(d_min / 2.0, w_min)
        
        return total_force, r_current

    def optimize_direct(config, iterations=1000):
        centers = config.copy()
        best_r = 0.0
        best_centers = centers.copy()
        
        for i in range(iterations):
            lr = 0.005 * (0.1 ** (i / iterations))
            
            force, r_current = energy_and_grad_direct(centers)
            
            if r_current > best_r:
                best_r = r_current
                best_centers = centers.copy()
                
            norm_force = np.linalg.norm(force)
            if norm_force > 1e-6:
                step = lr * force
                step_norm = np.linalg.norm(step, axis=1)
                max_step = 0.005
                mask = step_norm > max_step
                if np.any(mask):
                    step[mask] = step[mask] * (max_step / step_norm[mask])[:, np.newaxis]
                
                centers += step
                centers = np.clip(centers, -0.499, 0.499)
                
        force, r_current = energy_and_grad_direct(centers)
        if r_current > best_r:
            best_r = r_current
            best_centers = centers.copy()
            
        return best_centers, best_r

    # Run direct optimization on the spectral result
    if time.time() - start_time < time_limit:
        centers, r = optimize_direct(centers, iterations=1000)
        
    return centers

