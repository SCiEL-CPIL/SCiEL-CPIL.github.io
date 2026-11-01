import numpy as np
from scipy.spatial import Voronoi, HalfspaceIntersection
from scipy.optimize import minimize
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Helper functions
    
    def _compute_radius(centers):
        """
        Computes the maximum radius for a given set of centers.
        r = min(min_pair_dist/2, min_wall_dist)
        """
        # Pairwise distances
        # Using broadcasting for memory efficiency if N is small, but for N=50 it's fine.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        # Wall distances
        # Distance to closest wall: 0.5 - |coord|
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist)

    def _lloyd_step(centers):
        """
        Performs one iteration of Lloyd's algorithm.
        Moves each center to the centroid of its Voronoi cell clipped by the cube.
        """
        new_centers = centers.copy()
        
        # Cube half-spaces: a^T x <= b
        # x >= -0.5  => -x <= 0.5
        # x <=  0.5  =>  x <= 0.5
        cube_hs = np.array([
            [-1, 0, 0, 0.5],
            [ 1, 0, 0, 0.5],
            [ 0, -1, 0, 0.5],
            [ 0, 1, 0, 0.5],
            [ 0, 0, -1, 0.5],
            [ 0, 0, 1, 0.5]
        ])
        
        for i in range(N):
            # Construct half-spaces for the Voronoi cell of point i
            # Constraint: ||x - c_i||^2 <= ||x - c_j||^2
            # ||x||^2 - 2x.c_i + ||c_i||^2 <= ||x||^2 - 2x.c_j + ||c_j||^2
            # 2x.(c_j - c_i) <= ||c_j||^2 - ||c_i||^2
            hs_list = [cube_hs]
            
            for j in range(N):
                if i == j:
                    continue
                a = 2.0 * (centers[j] - centers[i])
                b = np.dot(centers[j], centers[j]) - np.dot(centers[i], centers[i])
                hs_list.append(np.array([[a[0], a[1], a[2], b]]))
            
            hs = np.vstack(hs_list)
            
            try:
                # The origin for HalfspaceIntersection must be inside the intersection.
                # centers[i] is inside its own Voronoi cell by definition.
                inter = HalfspaceIntersection(hs, centers[i])
                verts = inter.intersections
                
                if len(verts) == 0:
                    new_centers[i] = centers[i]
                else:
                    # Approximate centroid by average of vertices.
                    # For better accuracy, one could compute the exact centroid of the polytope,
                    # but vertex average is a standard heuristic for Lloyd's in this context 
                    # and is much faster. The subsequent polish will refine it.
                    new_centers[i] = np.mean(verts, axis=0)
            except Exception:
                new_centers[i] = centers[i]
        
        # Ensure we stay inside the cube
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        return new_centers

    def _soft_min_obj(centers_flat, beta):
        """
        Objective function for optimization: -soft_min(distances).
        Minimizing this maximizes the soft-min of distances.
        """
        centers = centers_flat.reshape(N, 3)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # We want the min of (dists/2) and (wall_dists)
        # Let's flatten all relevant distances into a single vector for soft-min
        # Note: Taking min of dists/2 is equivalent to taking min of dists then dividing by 2.
        # Let's work with distances directly to avoid scaling issues in exp, 
        # but we need to be careful with units.
        # Actually, let's just collect the values we care about:
        # 1. Pair distances (divided by 2 to be radii)
        # 2. Wall distances (radii)
        
        vals = []
        # Pairs
        for i in range(N):
            for j in range(i+1, N):
                vals.append(dists[i, j] / 2.0)
        # Walls
        wall_dists = 0.5 - np.abs(centers)
        vals.extend(np.min(wall_dists, axis=1))
        
        vals = np.array(vals)
        
        # Soft min: -1/beta * log(sum(exp(-beta * vals)))
        # To avoid overflow, subtract max
        v_max = np.max(vals)
        exp_vals = np.exp(-beta * (vals - v_max))
        log_sum_exp = np.log(np.sum(exp_vals)) - beta * v_max
        soft_min_val = -log_sum_exp / beta
        
        return -soft_min_val

    def _soft_min_grad(centers_flat, beta):
        """
        Gradient of the soft-min objective.
        """
        centers = centers_flat.reshape(N, 3)
        grad = np.zeros_like(centers)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.argmin(np.abs(centers), axis=1)
        wall_radii = np.min(wall_dists, axis=1)
        
        # Collect values for weights
        vals = np.zeros(N*(N-1)//2 + N)
        idx = 0
        
        # Store gradients for each value
        # This is memory intensive for N=50 (1225 pairs + 50 walls = 1275 values).
        # We can compute weights on the fly or store them.
        # Let's store values first to compute weights.
        
        pair_vals = np.zeros(N*(N-1)//2)
        wall_vals = wall_radii
        
        for i in range(N):
            for j in range(i+1, N):
                pair_vals[idx] = dists[i, j] / 2.0
                idx += 1
        
        all_vals = np.concatenate([pair_vals, wall_vals])
        
        # Compute weights
        v_max = np.max(all_vals)
        exp_vals = np.exp(-beta * (all_vals - v_max))
        weights = exp_vals / np.sum(exp_vals)
        
        # Accumulate gradients
        # Objective is -soft_min. 
        # d(-soft_min)/dx = - d(soft_min)/dx
        # d(soft_min)/dx = sum(w_k * dx_k/dx)
        # So grad_obj = - sum(w_k * grad_val_k)
        
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d > 1e-8:
                    w = weights[idx]
                    # Gradient of dist/2 w.r.t centers[i]
                    # d(||c_i - c_j||/2)/dc_i = (c_i - c_j) / (2 * d) * 2 ? No.
                    # f = 0.5 * ||c_i - c_j||
                    # df/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
                    vec = (centers[i] - centers[j]) / d
                    grad[i] -= w * 0.5 * vec
                    grad[j] += w * 0.5 * vec
                idx += 1
        
        for i in range(N):
            w = weights[idx]
            k = min_wall_idx[i]
            # Gradient of wall_dist = 0.5 - |c_ik|
            # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
            grad[i, k] -= w * (-np.sign(centers[i, k]))
            idx += 1
            
        return grad.flatten()

    def _metropolis_mc(centers, radius, n_steps, temp):
        """
        Metropolis Monte Carlo search.
        """
        current_centers = centers.copy()
        current_radius = radius
        best_centers = centers.copy()
        best_radius = radius
        
        for _ in range(n_steps):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random sphere
            i = np.random.randint(N)
            old_pos = current_centers[i].copy()
            
            # Random perturbation
            step_size = temp * 0.5 # Scale step with temp? Or fixed?
            # Let's use a fixed small step scaled by current radius
            perturbation = np.random.uniform(-step_size, step_size, 3)
            new_pos = old_pos + perturbation
            
            # Check bounds
            if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                continue
                
            current_centers[i] = new_pos
            new_radius = _compute_radius(current_centers)
            
            # Metropolis criterion
            if new_radius > current_radius:
                current_radius = new_radius
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = current_centers.copy()
            else:
                # Accept with probability exp((new_radius - current_radius) / temp)
                # Note: radii are ~0.13, so delta is small.
                # If temp is 0.001, delta/temp is significant.
                delta = new_radius - current_radius
                prob = np.exp(delta / temp)
                if np.random.rand() < prob:
                    current_radius = new_radius
                else:
                    current_centers[i] = old_pos
        
        return best_centers, best_radius

    def _local_search(centers, radius, n_iterations=100):
        """
        Simple local search: try moving each point slightly in random directions.
        """
        current_centers = centers.copy()
        current_radius = radius
        best_centers = centers.copy()
        best_radius = radius
        
        for _ in range(n_iterations):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for i in range(N):
                # Try a few small moves
                for _ in range(5):
                    old_pos = current_centers[i].copy()
                    # Small random step
                    step = 0.005 * current_radius
                    new_pos = old_pos + np.random.uniform(-step, step, 3)
                    
                    if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                        continue
                    
                    current_centers[i] = new_pos
                    new_radius = _compute_radius(current_centers)
                    
                    if new_radius > current_radius:
                        current_radius = new_radius
                        improved = True
                        if new_radius > best_radius:
                            best_radius = new_radius
                            best_centers = current_centers.copy()
                    else:
                        current_centers[i] = old_pos
                        
            if not improved:
                # Reduce step size or break?
                # For now, just continue.
                pass
                
        return best_centers, best_radius

    # 1. Initialization
    # Use a jittered grid to break symmetry and provide a good start.
    # A 4x4x4 grid has 64 points. We pick 50.
    # Alternatively, use FCC lattice points.
    # Let's try a 3x3x3 grid (27) + 23 random points in the gaps? 
    # Or just a 4x4x4 grid subset.
    
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Shuffle and pick 50
    np.random.seed(42)
    indices = np.random.choice(len(grid_points), N, replace=False)
    centers = grid_points[indices].copy()
    
    # Add small jitter
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    centers = np.clip(centers, -0.5, 0.5)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # 2. Lloyd's Algorithm
    # Run for a while to equalize space
    lloyd_iters = 0
    while time.time() < deadline - 30.0:
        old_centers = centers.copy()
        centers = _lloyd_step(centers)
        lloyd_iters += 1
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
        # Check convergence
        if np.linalg.norm(centers - old_centers) < 1e-6:
            break
            
    # 3. Smooth Polish (L-BFGS-B)
    # Use soft-min to refine
    if time.time() < deadline - 10.0:
        x0 = best_centers.flatten()
        
        # Use a moderate beta for the first pass
        beta = 50.0
        try:
            res = minimize(
                _soft_min_obj, 
                x0, 
                args=(beta,),
                method='L-BFGS-B', 
                jac=lambda x: _soft_min_grad(x, beta),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < _soft_min_obj(x0, beta):
                new_centers = res.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        except Exception:
            pass
        
        # Second pass with higher beta for precision
        beta = 200.0
        x0 = best_centers.flatten()
        try:
            res = minimize(
                _soft_min_obj, 
                x0, 
                args=(beta,),
                method='L-BFGS-B', 
                jac=lambda x: _soft_min_grad(x, beta),
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < _soft_min_obj(x0, beta):
                new_centers = res.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        except Exception:
            pass

    # 4. Metropolis Monte Carlo
    # Use remaining time for MC
    if time.time() < deadline - 5.0:
        # Adaptive temperature
        temp = 0.001 * best_radius
        while time.time() < deadline - 2.0:
            # Run a batch of MC steps
            mc_steps = 500
            best_centers, best_radius = _metropolis_mc(
                best_centers, best_radius, mc_steps, temp
            )
            # Cool down
            temp *= 0.9
            
            # Occasionally do a local search polish
            if time.time() < deadline - 5.0 and best_radius > 0:
                best_centers, best_radius = _local_search(best_centers, best_radius, 20)

    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# Keep the function name and signature exactly as requested
def construct():
    return _construct_impl()

# To avoid name collision or confusion, let's just rename the internal one 
# and have the public one call it. Or just use the name directly.
# The prompt says "Keep the program's entry-point function name(s) ... EXACTLY".
# The current program has `def construct():`.
# So I will just define `construct` as the main function.

# I will re-structure to have a single `construct` function.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    N = 50
    
    def _compute_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist)

    def _lloyd_step(centers):
        new_centers = centers.copy()
        cube_hs = np.array([
            [-1, 0, 0, 0.5],
            [ 1, 0, 0, 0.5],
            [ 0, -1, 0, 0.5],
            [ 0, 1, 0, 0.5],
            [ 0, 0, -1, 0.5],
            [ 0, 0, 1, 0.5]
        ])
        
        for i in range(N):
            hs_list = [cube_hs]
            for j in range(N):
                if i == j:
                    continue
                a = 2.0 * (centers[j] - centers[i])
                b = np.dot(centers[j], centers[j]) - np.dot(centers[i], centers[i])
                hs_list.append(np.array([[a[0], a[1], a[2], b]]))
            
            hs = np.vstack(hs_list)
            
            try:
                inter = HalfspaceIntersection(hs, centers[i])
                verts = inter.intersections
                if len(verts) == 0:
                    new_centers[i] = centers[i]
                else:
                    new_centers[i] = np.mean(verts, axis=0)
            except Exception:
                new_centers[i] = centers[i]
        
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        return new_centers

    def _soft_min_obj(centers_flat, beta):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        vals = []
        for i in range(N):
            for j in range(i+1, N):
                vals.append(dists[i, j] / 2.0)
        wall_dists = 0.5 - np.abs(centers)
        vals.extend(np.min(wall_dists, axis=1))
        vals = np.array(vals)
        
        v_max = np.max(vals)
        exp_vals = np.exp(-beta * (vals - v_max))
        log_sum_exp = np.log(np.sum(exp_vals)) - beta * v_max
        soft_min_val = -log_sum_exp / beta
        return -soft_min_val

    def _soft_min_grad(centers_flat, beta):
        centers = centers_flat.reshape(N, 3)
        grad = np.zeros_like(centers)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.argmin(np.abs(centers), axis=1)
        wall_radii = np.min(wall_dists, axis=1)
        
        pair_vals = np.zeros(N*(N-1)//2)
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                pair_vals[idx] = dists[i, j] / 2.0
                idx += 1
        
        all_vals = np.concatenate([pair_vals, wall_radii])
        
        v_max = np.max(all_vals)
        exp_vals = np.exp(-beta * (all_vals - v_max))
        weights = exp_vals / np.sum(exp_vals)
        
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d > 1e-8:
                    w = weights[idx]
                    vec = (centers[i] - centers[j]) / d
                    grad[i] -= w * 0.5 * vec
                    grad[j] += w * 0.5 * vec
                idx += 1
        
        for i in range(N):
            w = weights[idx]
            k = min_wall_idx[i]
            grad[i, k] -= w * (-np.sign(centers[i, k]))
            idx += 1
            
        return grad.flatten()

    def _metropolis_mc(centers, radius, n_steps, temp):
        current_centers = centers.copy()
        current_radius = radius
        best_centers = centers.copy()
        best_radius = radius
        
        for _ in range(n_steps):
            if time.time() > deadline - 5.0:
                break
                
            i = np.random.randint(N)
            old_pos = current_centers[i].copy()
            
            step_size = 0.01 * current_radius
            perturbation = np.random.uniform(-step_size, step_size, 3)
            new_pos = old_pos + perturbation
            
            if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                continue
                
            current_centers[i] = new_pos
            new_radius = _compute_radius(current_centers)
            
            if new_radius > current_radius:
                current_radius = new_radius
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = current_centers.copy()
            else:
                delta = new_radius - current_radius
                prob = np.exp(delta / temp)
                if np.random.rand() < prob:
                    current_radius = new_radius
                else:
                    current_centers[i] = old_pos
        
        return best_centers, best_radius

    def _local_search(centers, radius, n_iterations=50):
        current_centers = centers.copy()
        current_radius = radius
        best_centers = centers.copy()
        best_radius = radius
        
        for _ in range(n_iterations):
            if time.time() > deadline - 5.0:
                break
            
            for i in range(N):
                for _ in range(3):
                    old_pos = current_centers[i].copy()
                    step = 0.005 * current_radius
                    new_pos = old_pos + np.random.uniform(-step, step, 3)
                    
                    if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                        continue
                    
                    current_centers[i] = new_pos
                    new_radius = _compute_radius(current_centers)
                    
                    if new_radius > current_radius:
                        current_radius = new_radius
                        if new_radius > best_radius:
                            best_radius = new_radius
                            best_centers = current_centers.copy()
                    else:
                        current_centers[i] = old_pos
                        
        return best_centers, best_radius

    # 1. Initialization
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.seed(42)
    indices = np.random.choice(len(grid_points), N, replace=False)
    centers = grid_points[indices].copy()
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    centers = np.clip(centers, -0.5, 0.5)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # 2. Lloyd's Algorithm
    while time.time() < deadline - 30.0:
        old_centers = centers.copy()
        centers = _lloyd_step(centers)
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
        if np.linalg.norm(centers - old_centers) < 1e-6:
            break
            
    # 3. Smooth Polish
    if time.time() < deadline - 10.0:
        x0 = best_centers.flatten()
        beta = 50.0
        try:
            res = minimize(
                _soft_min_obj, x0, args=(beta,),
                method='L-BFGS-B', jac=lambda x: _soft_min_grad(x, beta),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < _soft_min_obj(x0, beta):
                new_centers = res.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        except Exception:
            pass
        
        beta = 200.0
        x0 = best_centers.flatten()
        try:
            res = minimize(
                _soft_min_obj, x0, args=(beta,),
                method='L-BFGS-B', jac=lambda x: _soft_min_grad(x, beta),
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < _soft_min_obj(x0, beta):
                new_centers = res.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        except Exception:
            pass

    # 4. Metropolis MC
    if time.time() < deadline - 5.0:
        temp = 0.001 * best_radius
        while time.time() < deadline - 2.0:
            best_centers, best_radius = _metropolis_mc(best_centers, best_radius, 500, temp)
            temp *= 0.9
            if time.time() < deadline - 5.0:
                best_centers, best_radius = _local_search(best_centers, best_radius, 20)

    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

