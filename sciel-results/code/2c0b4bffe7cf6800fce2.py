"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math

N = 50

# EVOLVE-BLOCK-START

def construct():
    """
    Place 50 equal circles inside the unit circle to maximize their common radius.
    Uses a two-stage optimization:
    1. A coarse global search using Particle Swarm Optimization (PSO) with a lattice initialization.
    2. A fine-grained local refinement using Gradient Descent on a smooth objective function.
    """
    
    # --- Objective Function ---
    # We want to maximize r = min(min_dist/2, 1 - max_norm).
    # This is equivalent to minimizing a "violation" metric or maximizing a smooth proxy.
    # A common smooth proxy for maximizing the minimum distance is:
    # f(X) = sum_{i<j} exp(-k * d_ij) + sum_i exp(-k * (1 - |c_i|))
    # However, the "score" is the raw minimum radius. 
    # To optimize the score directly, we can use a potential that heavily penalizes the minimum distance.
    # Let's define Energy E = - (min_dist/2) + lambda * max(0, max_norm - 1 + r_target)? No, r is variable.
    
    # Better: Minimize E = sum_{i<j} max(0, 2*R - d_ij)^2 + sum_i max(0, |c_i| + R - 1)^2
    # where R is a fixed target radius we try to achieve. If we can achieve R, we increase it.
    # But finding the optimal R is the hard part.
    
    # Alternative: Minimize E = sum_{i<j} 1/d_ij^2 + sum_i 1/(1-|c_i|)^2.
    # This pushes all pairs apart and keeps all points away from the boundary.
    # The minimizer of this function tends to have a large minimum distance.
    # Let's use this for the optimization, and then calculate the true radius at the end.
    
    def calculate_energy(pts):
        """
        Energy function to be minimized.
        Lower energy corresponds to a better packing (larger minimum distance).
        """
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-10)
        
        # Repulsive potential: 1/d^2
        E_pair = np.sum(1.0 / dist_sq)
        
        # Boundary potential: 1/(1-r)^2
        norms = np.linalg.norm(pts, axis=1)
        dist_to_boundary = 1.0 - norms
        dist_to_boundary = np.maximum(dist_to_boundary, 1e-10)
        E_boundary = np.sum(1.0 / (dist_to_boundary**2))
        
        return E_pair + E_boundary

    def calculate_radius(pts):
        """
        Calculate the actual score: the largest common radius.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist = np.sqrt(np.min(dist_sq))
        
        norms = np.linalg.norm(pts, axis=1)
        min_boundary = 1.0 - np.max(norms)
        
        return min(min_dist / 2.0, min_boundary)

    # --- Initialization ---
    # Use a hexagonal lattice initialization to provide a good starting point.
    # The optimal packing for 50 circles is known to be close to a hexagonal arrangement.
    
    def lattice_init():
        pts = []
        # Hexagonal lattice spacing
        # For radius r, spacing is 2r.
        # We want to fit 50 points.
        # Approximate area of unit circle = pi.
        # Area per circle ~ pi/50 ~ 0.0628.
        # Spacing s ~ sqrt(0.0628 / (sqrt(3)/2)) ~ 0.27.
        # Let's try a few spacings and pick the best one that fits inside.
        
        best_pts = None
        best_E = float('inf')
        
        for spacing in [0.25, 0.26, 0.27, 0.28, 0.29, 0.30]:
            # Generate hexagonal points
            candidates = []
            # Center at origin
            candidates.append([0.0, 0.0])
            
            # Rings
            for ring in range(1, 10):
                for k in range(6 * ring):
                    # Angle
                    angle = (k / (6.0 * ring)) * 2 * np.pi
                    # Distance
                    dist = ring * spacing
                    x = dist * np.cos(angle)
                    y = dist * np.sin(angle)
                    # Only keep points inside the unit circle (with some margin)
                    if np.sqrt(x*x + y*y) < 0.95:
                        candidates.append([x, y])
            
            if len(candidates) < N:
                continue
                
            # We have more than N candidates. We need to select N of them.
            # A simple heuristic: select the ones closest to the center? 
            # Or just take the first N? 
            # Actually, a better initialization is to take a subset that is well-distributed.
            # Let's just take the first N from the list (which are ordered by ring, so central first).
            # But we might miss some good outer points.
            # Let's try to select N points that maximize the minimum distance among themselves.
            # This is a combinatorial problem. For now, let's just take the first N.
            # To make it more robust, we can add some randomness to the selection.
            
            # Shuffle candidates and take N? No, that might be bad.
            # Let's take the N points with the largest "local density" or something?
            # Simplest: Take the first N.
            pts_candidate = np.array(candidates[:N])
            
            # Calculate energy
            E = calculate_energy(pts_candidate)
            if E < best_E:
                best_E = E
                best_pts = pts_candidate.copy()
                
        if best_pts is None:
            # Fallback to random
            pts = np.random.rand(N, 2) * 1.8 - 0.9
            norms = np.linalg.norm(pts, axis=1)
            mask = norms > 0.9
            pts[mask] = pts[mask] / norms[mask, None] * 0.9
            return pts
            
        return best_pts

    # --- Optimization 1: Particle Swarm Optimization (PSO) ---
    # PSO is good for global search.
    
    def pso_optimize(init_pts, max_iter=500, swarm_size=10):
        # Initialize swarm
        X = np.zeros((swarm_size, N, 2))
        V = np.zeros((swarm_size, N, 2))
        Pbest = np.zeros((swarm_size, N, 2))
        Pbest_E = np.zeros(swarm_size)
        
        # First particle is the lattice init
        X[0] = init_pts
        # Others are random perturbations of the lattice init
        for i in range(1, swarm_size):
            X[i] = init_pts + np.random.randn(N, 2) * 0.05
            # Ensure inside
            norms = np.linalg.norm(X[i], axis=1)
            mask = norms > 0.95
            X[i][mask] = X[i][mask] / norms[mask, None] * 0.95
            
        # Evaluate initial
        for i in range(swarm_size):
            Pbest[i] = X[i].copy()
            Pbest_E[i] = calculate_energy(X[i])
            
        Gbest = Pbest[0].copy()
        Gbest_E = Pbest_E[0]
        for i in range(1, swarm_size):
            if Pbest_E[i] < Gbest_E:
                Gbest_E = Pbest_E[i]
                Gbest = Pbest[i].copy()
                
        # PSO Parameters
        w = 0.9
        c1 = 1.5
        c2 = 1.5
        w_decay = 0.005
        
        for t in range(max_iter):
            w = max(0.4, w - w_decay)
            
            for i in range(swarm_size):
                r1 = np.random.rand(N, 1)
                r2 = np.random.rand(N, 1)
                
                V[i] = w * V[i] + c1 * r1 * (Pbest[i] - X[i]) + c2 * r2 * (Gbest - X[i])
                
                # Limit velocity
                V[i] = np.clip(V[i], -0.05, 0.05)
                
                X[i] = X[i] + V[i]
                
                # Project to circle
                norms = np.linalg.norm(X[i], axis=1)
                mask = norms > 0.99
                X[i][mask] = X[i][mask] / norms[mask, None] * 0.99
                
                E = calculate_energy(X[i])
                
                if E < Pbest_E[i]:
                    Pbest_E[i] = E
                    Pbest[i] = X[i].copy()
                    if E < Gbest_E:
                        Gbest_E = E
                        Gbest = X[i].copy()
                        
        return Gbest

    # --- Optimization 2: Gradient Descent Refinement ---
    # PSO might get stuck in local minima or leave the solution slightly suboptimal.
    # Gradient descent on the energy function can polish the result.
    
    def gradient_descent_refine(pts, max_iter=5000, step_size=0.01):
        # Numerical gradient
        eps = 1e-5
        
        current_E = calculate_energy(pts)
        
        for i in range(max_iter):
            # Calculate gradient
            grad = np.zeros_like(pts)
            for idx in range(N):
                # x component
                pts_p = pts.copy()
                pts_p[idx, 0] += eps
                E_p = calculate_energy(pts_p)
                
                pts_m = pts.copy()
                pts_m[idx, 0] -= eps
                E_m = calculate_energy(pts_m)
                
                grad[idx, 0] = (E_p - E_m) / (2 * eps)
                
                # y component
                pts_p = pts.copy()
                pts_p[idx, 1] += eps
                E_p = calculate_energy(pts_p)
                
                pts_m = pts.copy()
                pts_m[idx, 1] -= eps
                E_m = calculate_energy(pts_m)
                
                grad[idx, 1] = (E_p - E_m) / (2 * eps)
                
            # Update
            pts -= step_size * grad
            
            # Project to circle
            norms = np.linalg.norm(pts, axis=1)
            mask = norms > 0.99
            pts[mask] = pts[mask] / norms[mask, None] * 0.99
            
            new_E = calculate_energy(pts)
            
            # Simple adaptive step size
            if new_E < current_E:
                current_E = new_E
                # step_size = min(step_size * 1.05, 0.1)
            else:
                step_size *= 0.9
                # Revert if it got much worse?
                # For now, just keep the step size reduction.
                
            if step_size < 1e-6:
                break
                
        return pts

    # --- Main Execution ---
    
    # 1. Lattice Initialization
    init_pts = lattice_init()
    
    # 2. PSO Global Search
    pso_pts = pso_optimize(init_pts, max_iter=200, swarm_size=5)
    
    # 3. Gradient Descent Refinement
    final_pts = gradient_descent_refine(pso_pts, max_iter=2000, step_size=0.005)
    
    # 4. Final Validity Check and Projection
    # Ensure all points are strictly inside the unit circle
    norms = np.linalg.norm(final_pts, axis=1)
    mask = norms > 0.999
    if np.any(mask):
        final_pts[mask] = final_pts[mask] / norms[mask, None] * 0.999
    
    # Calculate final radius to verify
    # r = calculate_radius(final_pts)
    # print(f"Final Radius: {r}")
    
    return final_pts

# EVOLVE-BLOCK-END
