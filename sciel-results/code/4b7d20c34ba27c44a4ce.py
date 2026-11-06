"""
Program for packing 50 equal circles in a unit circle.
Strategy: Physics-based simulation (Molecular Dynamics) with Verlet integration.
This avoids the slow convergence and local minima traps of gradient-based methods
on non-smooth objectives by using forces to push circles apart.
"""
import numpy as np
import time
import random

N = 50
DEADLINE = time.time() + 140.0  # Safety margin out of 150s

def compute_radius(points):
    """Compute the packing radius for given centres."""
    points = np.asarray(points, dtype=np.float64)
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(boundary_dists.min(), min_pair_dist / 2.0)

def fast_radius(points):
    """Fast radius computation."""
    points = np.asarray(points, dtype=np.float64)
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    return min(boundary_dists.min(), min_pair_dist / 2.0)

def simulate_packing(initial_points, n_steps=2000, dt=0.001, damping=0.95):
    """
    Perform molecular dynamics simulation to maximize the minimum distance.
    We simulate repulsive forces between circles and a boundary force.
    The "temperature" or step size is annealed down.
    """
    points = initial_points.copy()
    velocities = np.zeros_like(points)
    
    # Precompute pairs? No, N=50 is small, O(N^2) is fine.
    # We want to maximize the minimum distance. 
    # A common trick is to simulate a system where particles repel strongly if close.
    # Or, we can use a potential energy function that penalizes overlaps and rewards spreading.
    # Let's use a simple repulsive force: F_ij = k * (1/d^2) if d < 2*r_current_estimate?
    # Actually, a better approach for packing is to use a potential that is flat for d > 2r and rises sharply for d < 2r.
    # But we don't know r. 
    # Alternative: Maximize the sum of log(d_ij) or similar? No, that's not robust.
    
    # Let's use a force-based approach where we try to push all pairs to a target distance D.
    # We can iterate:
    # 1. Estimate current min distance D_min.
    # 2. Apply forces to increase distances of pairs closer than some target.
    # 3. Apply forces to keep points inside the boundary.
    
    # Better: Use a simulated annealing approach on the positions directly?
    # The prompt suggests MD. Let's do MD with a repulsive potential.
    # Potential: V = sum_{i<j} 1/d_ij^2 + sum_i 1/(1-|r_i|)^2 ?
    # Gradient descent on this potential will push points apart.
    
    # Let's implement a simple gradient descent step with adaptive step size,
    # but framed as "forces".
    
    current_r = fast_radius(points)
    if current_r <= 0:
        return points
    
    # We want to increase the minimum distance.
    # Let's use a soft objective: minimize sum of (1/d_ij)^p for p=2 or 4, 
    # plus boundary penalty.
    # This is effectively what the current code tries to do with Nelder-Mead on rings.
    # But here we do it on free coordinates.
    
    # Let's try a simple repulsive force model.
    # Force on i from j: F = k * (target_dist - d_ij) * unit_vec if d_ij < target_dist
    # What is target_dist? We can set it to 2 * current_r * 1.1 to push them apart.
    
    target_r = current_r * 1.05 # Try to grow radius by 5%
    target_dist = 2 * target_r
    
    for step in range(n_steps):
        if time.time() > DEADLINE - 5.0:
            break
            
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                d = np.linalg.norm(diff)
                if d < 1e-6:
                    # Random kick if coincident
                    forces[i] += np.random.uniform(-0.01, 0.01, 2)
                    forces[j] -= np.random.uniform(-0.01, 0.01, 2)
                    continue
                if d < target_dist:
                    # Force proportional to overlap
                    overlap = target_dist - d
                    force_mag = 10.0 * overlap # Spring constant
                    unit_vec = diff / d
                    forces[i] += force_mag * unit_vec
                    forces[j] -= force_mag * unit_vec
        
        # Boundary repulsion
        for i in range(N):
            r_i = np.linalg.norm(points[i])
            if r_i > 0:
                # We want to keep |r_i| + current_r <= 1
                # So |r_i| <= 1 - current_r
                max_r = 1.0 - current_r
                if r_i > max_r * 0.95: # Start pushing before hitting wall
                    # Force inward
                    overlap = r_i - max_r * 0.95
                    force_mag = 10.0 * overlap
                    unit_vec = -points[i] / r_i
                    forces[i] += force_mag * unit_vec
        
        # Update velocities and positions
        velocities = velocities * damping + forces * dt
        points += velocities * dt
        
        # Hard constraint: keep inside circle
        norms = np.linalg.norm(points, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                points[i] *= 0.999 / norms[i]
                velocities[i] = 0.0 # Stop if hit wall
        
        # Anneal: reduce target growth rate over time?
        # Or just let it settle.
        
        # Every 100 steps, check if we improved
        if step % 100 == 0:
            new_r = fast_radius(points)
            if new_r > current_r:
                current_r = new_r
                target_r = current_r * 1.02 # Slow down growth
                target_dist = 2 * target_r
            else:
                # If no improvement, maybe reduce force magnitude to settle?
                pass
                
    return points

def local_polish(points, n_steps=500, step_size=0.01):
    """
    Simple coordinate descent polish.
    """
    x = points.copy()
    current_r = fast_radius(x)
    
    for _ in range(n_steps):
        if time.time() > DEADLINE - 2.0:
            break
        improved = False
        indices = np.random.permutation(N)
        for i in indices:
            best_x_i = x[i].copy()
            best_r_i = current_r
            
            # Try small moves in 8 directions
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x_i = x[i] + np.array([dx, dy])
                    if np.linalg.norm(new_x_i) >= 1.0:
                        continue
                    x[i] = new_x_i
                    new_r = fast_radius(x)
                    if new_r > best_r_i:
                        best_r_i = new_r
                        best_x_i = new_x_i.copy()
                    else:
                        x[i] = best_x_i # revert
            if best_r_i > current_r:
                x[i] = best_x_i
                current_r = best_r_i
                improved = True
        if not improved:
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
    return x, current_r

def construct():
    global DEADLINE
    DEADLINE = time.time() + 140.0
    
    np.random.seed(42)
    random.seed(42)
    
    best_points = None
    best_radius = 0.0
    
    # Generate multiple initial configurations
    # 1. Random uniform
    # 2. Hexagonal lattice clipped
    # 3. Concentric rings (as a good start)
    
    initial_configs = []
    
    # Random
    for _ in range(5):
        pts = np.random.uniform(-0.8, 0.8, size=(N, 2))
        # Ensure inside
        norms = np.linalg.norm(pts, axis=1)
        pts[norms > 1] *= 0.99 / norms[norms > 1]
        initial_configs.append(pts)
        
    # Hexagonal
    spacing = 0.25
    hex_pts = []
    for j in range(-5, 6):
        for k in range(-5, 6):
            px = k * spacing + (0.5 * spacing if j % 2 else 0)
            py = j * spacing * np.sqrt(3) / 2
            if np.sqrt(px * px + py * py) < 0.9:
                hex_pts.append([px, py])
    if len(hex_pts) >= N:
        # Take the N closest to center? Or random N?
        # Let's take a random subset to break symmetry
        indices = np.random.choice(len(hex_pts), N, replace=False)
        initial_configs.append(np.array(hex_pts)[indices])
        
    # Rings
    # 1 center, 6, 12, 18, 13
    counts = [1, 6, 12, 18, 13]
    radii = np.linspace(0.1, 0.8, len(counts))
    ring_pts = []
    for c, r in zip(counts, radii):
        if c == 1 and r < 0.2:
             ring_pts.append([0.0, 0.0])
        else:
             angles = np.linspace(0, 2*np.pi, c, endpoint=False)
             for a in angles:
                 ring_pts.append([r * np.cos(a), r * np.sin(a)])
    if len(ring_pts) == N:
         initial_configs.append(np.array(ring_pts))
         
    # Shuffle
    random.shuffle(initial_configs)
    
    # Time budget per config
    time_per_config = (DEADLINE - time.time() - 10.0) / len(initial_configs)
    
    for init_pts in initial_configs:
        if time.time() > DEADLINE - 10.0:
            break
            
        # Run MD simulation
        # Adjust n_steps based on time
        n_steps = int(min(2000, max(200, time_per_config * 200)))
        
        sim_pts = simulate_packing(init_pts, n_steps=n_steps)
        
        # Polish
        if time.time() < DEADLINE - 5.0:
            polished_pts, r = local_polish(sim_pts)
            if r > best_radius:
                best_radius = r
                best_points = polished_pts.copy()
        else:
            r = fast_radius(sim_pts)
            if r > best_radius:
                best_radius = r
                best_points = sim_pts.copy()
                
    # Fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = np.random.uniform(-0.8, 0.8, size=(N, 2))
        norms = np.linalg.norm(best_points, axis=1)
        for i in range(N):
            if norms[i] >= 1.0:
                best_points[i] *= 0.999 / norms[i]
                
    # Final validation and clipping
    best_points = np.asarray(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] >= 1.0:
            best_points[i] *= 0.999 / norms[i]
            
    return best_points

if __name__ == "__main__":
    points = construct()
    print("Final radius:", compute_radius(points))
