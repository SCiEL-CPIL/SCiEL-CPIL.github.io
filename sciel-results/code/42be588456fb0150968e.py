"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit (35s buffer)
    
    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def get_covering_info(pts):
        """
        Computes the convex hull, finds the facet with the largest circumradius,
        and returns (radius_in_radians, worst_circumcenter, worst_simplex_indices).
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            # Fallback: return a large radius and a random point
            return np.pi, np.random.randn(3) / np.linalg.norm(np.random.randn(3)), None

        max_radius = 0.0
        worst_center = None
        worst_simplex = None
        
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
                
            c = n / norm_n
            # Ensure c is on the correct side (pointing outwards)
            if np.dot(c, v1) < 0:
                c = -c
                
            # Calculate angular distance from c to v1
            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            
            if radius > max_radius:
                max_radius = radius
                worst_center = c
                worst_simplex = simplex.copy()
                
        if worst_center is None:
            return np.pi, np.random.randn(3) / np.linalg.norm(np.random.randn(3)), None
            
        return max_radius, worst_center, worst_simplex

    def local_refine(pts, max_iters=200, step_decay=0.95, initial_step=0.05):
        """
        Local search using gradient-like steps on the worst facet.
        Returns refined points.
        """
        current = pts.copy()
        step = initial_step
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            radius, worst_center, worst_simplex = get_covering_info(current)
            if worst_simplex is None:
                break
            
            improved = False
            for idx in worst_simplex:
                v = current[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir < 1e-10:
                    continue
                dir_vec /= norm_dir
                
                new_v = v + step * dir_vec
                new_v /= np.linalg.norm(new_v)
                
                # Check if this specific move reduces the radius
                # We do a quick check by just applying it and seeing if the next full eval improves.
                # To be efficient, we apply all moves for the worst simplex, then check global radius.
                current[idx] = new_v
                improved = True
            
            if not improved:
                break
                
            # Decay step
            step *= step_decay
            if step < 1e-6:
                break
                
        return current

    # Parallel Tempering / Multi-Start Basin Hopping Strategy
    # We maintain a population of K replicas.
    # Each replica undergoes perturbation and local refinement.
    # We keep the best overall.
    
    K = 10  # Number of replicas
    best_points = None
    best_radius = np.pi
    
    # Initialize replicas
    replicas = []
    for i in range(K):
        if i == 0:
            # First replica starts from Fibonacci
            pts = fibonacci_sphere(N).copy()
        else:
            # Subsequent replicas start from random rotations of Fibonacci or random points
            # Random rotation of Fibonacci
            R = np.random.randn(3, 3)
            R, _ = np.linalg.qr(R)
            pts = fibonacci_sphere(N) @ R.T
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
            # Add some noise to diversify
            noise = np.random.randn(N, 3) * 0.1
            pts += noise
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
        replicas.append(pts)

    # Main Loop: Basin Hopping with Parallel Replicas
    # We iterate until time runs out.
    # In each iteration:
    # 1. For each replica, perform a perturbation (kick) and local refinement.
    # 2. Update the global best.
    # 3. Optionally, swap replicas if one is much worse (to keep diversity, though simple basin hopping on each is often enough).
    
    perturbation_magnitude = 0.1
    min_perturbation = 0.01
    
    while time.time() < deadline - 5:
        for i in range(K):
            if time.time() > deadline - 5:
                break
            
            # 1. Perturb the current replica
            # Random kick to escape local minima
            noise = np.random.randn(N, 3) * perturbation_magnitude
            perturbed = replicas[i] + noise
            perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
            
            # 2. Local Refinement
            # Use a smaller number of iterations for local refine to allow more global moves
            refined = local_refine(perturbed, max_iters=50, initial_step=0.05)
            
            # 3. Evaluate
            radius, _, _ = get_covering_info(refined)
            
            # 4. Update Global Best
            if radius < best_radius:
                best_radius = radius
                best_points = refined.copy()
            
            # 5. Update Replica
            # Accept the new state if it's better than the current replica state, or always in basin hopping?
            # Standard Basin Hopping: accept based on Metropolis criterion or just keep best of (old, new).
            # Let's keep the better of (current replica, refined) to maintain a good starting point for next kick.
            old_radius, _, _ = get_covering_info(replicas[i])
            if radius < old_radius:
                replicas[i] = refined
            else:
                # Keep old replica but maybe mix? 
                # To ensure diversity, we might want to occasionally reset a replica that is stuck.
                pass
            
            # Adaptive perturbation: if no improvement in this replica recently, increase kick
            # Simple heuristic: decay perturbation slowly, but if stuck, boost it.
            # For simplicity, let's just slowly decay the global perturbation magnitude.
        
        # Decay perturbation magnitude over time
        perturbation_magnitude *= 0.98
        if perturbation_magnitude < min_perturbation:
            perturbation_magnitude = min_perturbation
            
        # Occasionally, re-seed the worst replica with a fresh random/Fibonacci start
        if i % 10 == 0: # i is last replica index
            # Find worst replica
            radii = []
            for rep in replicas:
                r, _, _ = get_covering_info(rep)
                radii.append(r)
            worst_idx = np.argmax(radii)
            if radii[worst_idx] > best_radius * 1.1: # If significantly worse than best
                # Reset worst replica
                R = np.random.randn(3, 3)
                R, _ = np.linalg.qr(R)
                new_rep = fibonacci_sphere(N) @ R.T
                new_rep /= np.linalg.norm(new_rep, axis=1, keepdims=True)
                replicas[worst_idx] = new_rep

    # Final Polish on the best point found
    if best_points is not None and time.time() < deadline - 1:
        # One last aggressive local refinement on the best candidate
        final_pts = local_refine(best_points, max_iters=500, initial_step=0.02)
        final_radius, _, _ = get_covering_info(final_pts)
        if final_radius < best_radius:
            best_points = final_pts
            best_radius = final_radius

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
