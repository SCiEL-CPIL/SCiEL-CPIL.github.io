"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient.
    Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient for point i: sum_{j!=i} (x_i - x_j) / d_ij^2
    """
    n = len(pts)
    # Pairwise differences: pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Check for proximity
    if np.any(dist_sq[mask] < 1e-18):
        return float('inf'), None
        
    dists = np.sqrt(dist_sq[mask])
    
    # Energy
    energy = -np.sum(np.log(dists))
    
    # Gradient
    # grad[i] = sum_{j != i} (x_i - x_j) / dist_sq[i,j]
    # Note: The gradient of -log(d) w.r.t x_i is -(1/d) * (x_i - x_j)/d = -(x_i - x_j)/d^2
    # Wait, let's re-derive.
    # E = - sum log(d_ij)
    # dE/dx_i = - sum_{j != i} (1/d_ij) * d(d_ij)/dx_i
    # d(d_ij)/dx_i = (x_i - x_j) / d_ij
    # So dE/dx_i = - sum_{j != i} (x_i - x_j) / d_ij^2
    # This points towards the other points?
    # If x_i is close to x_j, x_i - x_j is small vector.
    # Let's check direction.
    # If we want to minimize E, we move in direction -grad.
    # -grad = sum (x_i - x_j) / d_ij^2.
    # This vector points from x_j to x_i. So it pushes x_i away from x_j. Correct.
    
    # We need to sum over j for each i.
    # dist_sq is n x n. We want to divide diff by dist_sq.
    # diff[i,j] is x_i - x_j.
    # We want to sum over j for each i.
    # But we must exclude j=i (dist_sq=0).
    
    # Set diagonal to 1 to avoid division by zero, then zero out diagonal later.
    dist_sq_safe = dist_sq.copy()
    dist_sq_safe[dist_sq_safe == 0] = 1.0
    
    # Weighted sum
    # term[i,j] = diff[i,j] / dist_sq[i,j]
    terms = diff / dist_sq_safe[:, :, np.newaxis]
    
    # Zero out diagonal
    idx = np.arange(n)
    terms[idx, idx] = 0.0
    
    # Sum over j (axis 1) to get gradient for each i
    grad = np.sum(terms, axis=1)
    
    # The gradient is sum (x_i - x_j) / d^2.
    # To minimize energy, we update x_i <- x_i - step * grad_i.
    # This moves x_i in the direction of -grad, which is sum (x_j - x_i)/d^2.
    # This pushes x_i away from neighbors. Correct.
    
    return energy, grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _local_optimize(pts, time_limit=5.0):
    """
    Perform gradient descent on the sphere for a fixed time limit.
    Uses adaptive step size and projection.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_energy, current_grad = _energy_and_grad(current_pts)
    
    if not np.isfinite(current_energy):
        return current_pts, current_energy
        
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    step_size = 0.05
    min_step = 1e-6
    max_step = 0.5
    patience = 0
    max_patience = 1000
    
    while time.time() - start_time < time_limit:
        # Check gradient norm
        grad_norm = np.linalg.norm(current_grad)
        
        if grad_norm < 1e-7:
            break
            
        # Update
        new_pts = current_pts - step_size * current_grad
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if not np.isfinite(new_energy):
            # Step too large, reduce step size
            step_size *= 0.5
            if step_size < min_step:
                break
            continue
            
        if new_energy < current_energy:
            # Accept
            current_pts = new_pts
            current_energy = new_energy
            current_grad = new_grad
            patience = 0
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
            # Increase step size slightly if improvement is significant
            # Simple heuristic: if improvement > 1e-5, increase step
            if current_energy - new_energy > 1e-6: # Note: new_energy < current_energy, so this is negative? 
                # Actually current_energy was old, new_energy is new.
                # Improvement = current_energy - new_energy > 0
                pass 
            # Let's just keep step size stable or slightly increase
            step_size = min(step_size * 1.05, max_step)
        else:
            # Reject, reduce step size
            step_size *= 0.8
            patience += 1
            
            if patience > max_patience or step_size < min_step:
                break
                
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # Define initial configurations
    configs = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    configs.append(fib_pts)
    
    # 2. Random Uniform on Sphere
    np.random.seed(42)
    rand_pts = np.random.normal(size=(N, 3))
    rand_pts = _project_to_sphere(rand_pts)
    configs.append(rand_pts)
    
    # 3. Icosahedron-based (low symmetry, good spread)
    # Generate icosahedron vertices and subdivide or just use a dense random set
    # Actually, let's just use a few more random starts with different seeds
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        r_pts = np.random.normal(size=(N, 3))
        r_pts = _project_to_sphere(r_pts)
        configs.append(r_pts)
        
    # 4. Two-cluster start (to encourage merging/separation dynamics)
    cluster1 = np.random.normal(size=(N//2, 3))
    cluster2 = np.random.normal(size=(N - N//2, 3))
    # Separate them
    cluster1[:, 0] += 1.0
    cluster2[:, 0] -= 1.0
    two_cluster = np.vstack([cluster1, cluster2])
    two_cluster = _project_to_sphere(two_cluster)
    configs.append(two_cluster)
    
    best_pts_global = fib_pts
    best_energy_global = float('inf')
    
    # Optimize each configuration
    for idx, cfg in enumerate(configs):
        if time.time() > deadline - 10.0:
            break
            
        # Allocate remaining time / remaining configs
        remaining_configs = len(configs) - idx
        time_per_config = (deadline - time.time()) / remaining_configs
        # Cap time per config to 20s to ensure we try all
        time_per_config = min(time_per_config, 20.0)
        
        # If we have very little time left, skip
        if time_per_config < 0.5:
            continue
            
        optimized_pts, energy = _local_optimize(cfg, time_limit=time_per_config)
        
        if energy < best_energy_global:
            best_energy_global = energy
            best_pts_global = optimized_pts
            
        # Print progress for debugging (optional, remove in final if slow)
        # print(f"Config {idx}: Energy {energy:.6f}")
        
    # Final Polish
    # If we have time, do a very fine-tuned optimization on the best result
    if time.time() < deadline - 2.0:
        # Use a smaller step size and longer time for final polish
        # We can just call _local_optimize again with a tight limit
        final_pts, final_energy = _local_optimize(best_pts_global, time_limit=5.0)
        if final_energy < best_energy_global:
            best_pts_global = final_pts
            best_energy_global = final_energy

    # Ensure output is valid
    best_pts_global = _project_to_sphere(best_pts_global)
    
    return best_pts_global
# EVOLVE-BLOCK-END
