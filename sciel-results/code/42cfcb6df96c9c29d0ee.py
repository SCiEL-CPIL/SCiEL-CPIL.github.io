import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Generate a dense hexagonal lattice to get a good initial set of N points
    # The hexagonal packing density is ~0.9069. 
    # For N=300, we need a region of area ~ 300 * pi * r^2.
    # Let's just generate a large grid and pick the best N.
    
    def generate_hex_lattice(N_points, radius_limit=1.0):
        # Estimate spacing. Area of unit circle is pi. 
        # Area per circle in hex packing is approx 2 * pi / sqrt(3) * r^2.
        # Total area ~ N * 2 * pi / sqrt(3) * r^2.
        # If they fit in unit circle, N * 2 / sqrt(3) * r^2 <= 1 => r <= sqrt(3 / (2*N)).
        # For N=300, r ~ sqrt(3/600) ~ 0.07.
        # Let's use a spacing slightly smaller than the theoretical max to ensure we have enough points.
        spacing = 2.0 * np.sqrt(3.0 / (2.0 * N)) * 0.9 
        # Actually, let's just use a fixed reasonable spacing and filter.
        # r_est = 0.055. spacing = 2*r_est = 0.11.
        spacing = 0.11
        
        points = []
        # Generate points in a bounding box large enough to cover the unit circle
        # Max x, y is 1.0.
        # Number of rows/cols
        range_val = int(1.5 / spacing) + 2
        
        for j in range(-range_val, range_val + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            if abs(y) > 1.2:
                continue
            offset = 0.0
            if j % 2 != 0:
                offset = spacing / 2.0
            for i in range(-range_val, range_val + 1):
                x = i * spacing + offset
                # Check if inside a slightly larger circle to allow for optimization
                if np.hypot(x, y) <= 1.0 - spacing/2.0 + 0.01: # Buffer
                    points.append([x, y])
        
        points = np.array(points)
        
        # Sort by distance to origin (closest first) to get a compact core
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)
        selected = points[indices]
        
        # We might have more than N points. We want the N closest to origin.
        if len(selected) >= N:
            return selected[:N]
        else:
            # If not enough, generate more with tighter spacing
            return generate_hex_lattice(N_points, radius_limit=1.0, spacing=spacing*0.9)

    # Recursive helper with spacing param
    def gen_lattice(spacing):
        points = []
        range_val = int(1.5 / spacing) + 2
        for j in range(-range_val, range_val + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            if abs(y) > 1.2:
                continue
            offset = 0.0
            if j % 2 != 0:
                offset = spacing / 2.0
            for i in range(-range_val, range_val + 1):
                x = i * spacing + offset
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
             return None
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)
        return points[indices[:N]]

    # Try a few spacings to find a valid initial config
    x = None
    for sp in [0.12, 0.11, 0.10, 0.09]:
        x = gen_lattice(sp)
        if x is not None:
            break
            
    if x is None:
        # Fallback to simple grid if hex fails (unlikely)
        x = np.random.randn(N, 2) * 0.5

    # Ensure valid start: scale down if any points are too close to boundary
    # Calculate initial radius
    def calc_radius(pts):
        min_dist = np.inf
        # Pairwise distances
        # For N=300, N^2 is 90k, which is fast.
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_dist / 2.0, min_boundary)

    r_current = calc_radius(x)
    
    # Optimization Loop: Gradient Ascent on smoothed min-distance
    # Objective: Maximize -L(X, beta) where L is log-sum-exp of negative distances
    # We want to maximize the minimum of:
    # A = { ||x_i - x_j|| / 2 }
    # B = { 1 - ||x_i|| }
    
    # To make it differentiable, we use the log-sum-exp of the *negatives* of the squared distances?
    # No, log-sum-exp of exp(-beta * val) approximates min(val).
    # So we minimize F(X) = (1/beta) * ln( sum_{i<j} exp(-beta * ||x_i-x_j||/2) + sum_i exp(-beta * (1-||x_i||)) )
    # Gradient of F w.r.t x_k:
    # dF/dx_k = -1/beta * 1/Sum * sum_terms * d(term)/dx_k
    # This is a weighted average of the gradients of the individual terms, weighted by their "closeness" to the minimum.
    
    # Let's implement the gradient step directly.
    
    def compute_gradient(pts, beta):
        grads = np.zeros_like(pts)
        
        # Precompute pairwise distances and norms
        # dists[i, j] = ||x_i - x_j||
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf) # Ignore self
        
        # Boundary distances
        norms = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - norms
        
        # We want to minimize the "soft min" of all these distances.
        # Let u be the list of all relevant distances.
        # u = [dists[i,j]/2 for i<j] + [boundary_dists[i] for i]
        
        # To avoid huge memory for N=300 (45k pairs), we can compute in chunks or just vectorize.
        # 300x300 matrix is fine.
        
        # Weights for pairwise
        # w_ij = exp(-beta * dists[i,j]/2)
        # We only need upper triangle.
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        pair_dists = dists[mask]
        pair_weights = np.exp(-beta * pair_dists / 2.0)
        
        # Weights for boundary
        bnd_dists = boundary_dists
        bnd_weights = np.exp(-beta * bnd_dists)
        
        total_weight = np.sum(pair_weights) + np.sum(bnd_weights)
        
        if total_weight == 0:
             return grads
             
        # Gradient of log-sum-exp
        # d/dx_k [ ln( sum w_m ) ] = (1/sum w_m) * sum (w_m * dw_m/dx_k)
        
        # dw_pair/dx_k:
        # w_ij = exp(-beta/2 * ||x_i - x_j||)
        # dw/dx_i = -beta/2 * w_ij * (x_i - x_j) / ||x_i - x_j||
        # dw/dx_j = -beta/2 * w_ij * (x_j - x_i) / ||x_i - x_j||
        
        # We need to accumulate gradients.
        # Instead of looping over all pairs (slow in python), we can use matrix operations.
        # Let W be the matrix of pair weights (zeros on diag).
        W = np.zeros((N, N))
        W[mask] = pair_weights
        
        # Term for x_k from pair (i, j):
        # If k=i: -beta/2 * W_ij * (x_i - x_j) / dists_ij
        # If k=j: -beta/2 * W_ij * (x_j - x_i) / dists_ij
        
        # Let D be the distance matrix.
        # Vector v_ij = (x_i - x_j) / D_ij
        # Grad_x_k = -beta/2 * sum_j W_kj * v_kj  - beta/2 * sum_i W_ik * v_ik ?
        # Note: v_jk = -v_kj.
        # Sum over all j != k: W_kj * (x_k - x_j)/D_kj
        # This is sum_j W_kj * v_kj.
        
        # Let's compute S_k = sum_{j != k} W_kj * (x_k - x_j) / D_kj
        # (x_k - x_j) is row k of diff matrix.
        # D_kj is row k of dists.
        
        # Avoid division by zero on diagonal.
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        
        # Unit vectors from k to j
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        # Weighted sum of unit vectors
        # S_k = sum_j W_kj * unit_vecs[k, j]
        # Note: W is symmetric.
        S = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
        
        pair_grads = -beta / 2.0 * S
        
        # Boundary gradient
        # w_i = exp(-beta * (1 - ||x_i||))
        # dw/dx_i = -beta * w_i * d(1-||x_i||)/dx_i = -beta * w_i * (-x_i/||x_i||) = beta * w_i * x_i/||x_i||
        
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        bnd_unit = pts / safe_norms[:, np.newaxis]
        bnd_grads = beta * bnd_weights[:, np.newaxis] * bnd_unit
        
        # Total gradient of the log-sum-exp part
        # F = (1/beta) * ln(Sum)
        # dF/dx = (1/beta) * (1/TotalWeight) * (Sum of weighted grads)
        
        total_grad = pair_grads + bnd_grads
        grad_F = (1.0 / (beta * total_weight)) * total_grad
        
        # We want to MAXIMIZE the min distance, which is MINIMIZING F.
        # So we take a step in the direction of -grad_F.
        return -grad_F

    # Optimization Loop
    step_size = 0.05
    beta = 10.0
    
    best_x = x.copy()
    best_r = r_current
    
    while time.time() < deadline:
        # Compute gradient
        grad = compute_gradient(x, beta)
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(grad)):
            beta *= 0.5
            continue
            
        # Step
        x_new = x + step_size * grad
        
        # Ensure points stay within a safe bound (e.g. radius 1.0) to prevent divergence
        # If a point goes outside, it's invalid. We should project or revert.
        # Simple check: if any norm > 1.0, revert this step.
        if np.any(np.linalg.norm(x_new, axis=1) > 1.0):
             # Try smaller step
             step_size *= 0.5
             x_new = x + step_size * grad
             if np.any(np.linalg.norm(x_new, axis=1) > 1.0):
                 continue
                 
        # Check if improvement
        r_new = calc_radius(x_new)
        
        if r_new > best_r:
            best_r = r_new
            best_x = x_new
            x = x_new
            # Increase beta to sharpen
            beta *= 1.1
            # Reset step size slightly if we are making progress
            step_size = min(step_size * 1.05, 0.1)
        else:
            # No improvement, decrease step size
            step_size *= 0.9
            # Increase beta to focus on the current bottleneck
            beta *= 1.05
            
        # If step size is too small, we might be stuck
        if step_size < 1e-6:
             break

    # Final Polish: Local search on the best solution found
    # Use a few random perturbations to escape local minima if time permits
    if time.time() < deadline - 5.0:
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            # Perturb
            x_pert = best_x + np.random.randn(N, 2) * 0.01
            # Project back to inside circle if needed? No, just check validity.
            if np.any(np.linalg.norm(x_pert, axis=1) > 1.0):
                continue
            r_pert = calc_radius(x_pert)
            if r_pert > best_r:
                best_r = r_pert
                best_x = x_pert
                
    return best_x

# Keep the original function name and signature
def construct():
    return _construct_impl()

def _construct_impl():
    # The logic above is in the local scope of the previous draft. 
    # Let's wrap it properly.
    import time
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 300
    
    def gen_lattice(spacing):
        points = []
        range_val = int(1.5 / spacing) + 2
        for j in range(-range_val, range_val + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            if abs(y) > 1.2:
                continue
            offset = 0.0
            if j % 2 != 0:
                offset = spacing / 2.0
            for i in range(-range_val, range_val + 1):
                x = i * spacing + offset
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
             return None
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)
        return points[indices[:N]]

    x = None
    for sp in [0.12, 0.11, 0.10, 0.09]:
        x = gen_lattice(sp)
        if x is not None:
            break
            
    if x is None:
        x = np.random.randn(N, 2) * 0.5

    def calc_radius(pts):
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_dist / 2.0, min_boundary)

    r_current = calc_radius(x)
    
    def compute_gradient(pts, beta):
        grads = np.zeros_like(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        norms = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - norms
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists = dists[mask]
        pair_weights = np.exp(-beta * pair_dists / 2.0)
        bnd_dists = boundary_dists
        bnd_weights = np.exp(-beta * bnd_dists)
        
        total_weight = np.sum(pair_weights) + np.sum(bnd_weights)
        if total_weight == 0:
             return grads
             
        W = np.zeros((N, N))
        W[mask] = pair_weights
        
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        S = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
        pair_grads = -beta / 2.0 * S
        
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        bnd_unit = pts / safe_norms[:, np.newaxis]
        bnd_grads = beta * bnd_weights[:, np.newaxis] * bnd_unit
        
        total_grad = pair_grads + bnd_grads
        grad_F = (1.0 / (beta * total_weight)) * total_grad
        return -grad_F

    step_size = 0.05
    beta = 10.0
    
    best_x = x.copy()
    best_r = r_current
    
    while time.time() < deadline:
        grad = compute_gradient(x, beta)
        
        if not np.all(np.isfinite(grad)):
            beta *= 0.5
            continue
            
        x_new = x + step_size * grad
        
        if np.any(np.linalg.norm(x_new, axis=1) > 1.0):
             step_size *= 0.5
             x_new = x + step_size * grad
             if np.any(np.linalg.norm(x_new, axis=1) > 1.0):
                 continue
                 
        r_new = calc_radius(x_new)
        
        if r_new > best_r:
            best_r = r_new
            best_x = x_new
            x = x_new
            beta *= 1.1
            step_size = min(step_size * 1.05, 0.1)
        else:
            step_size *= 0.9
            beta *= 1.05
            
        if step_size < 1e-6:
             break

    if time.time() < deadline - 5.0:
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            x_pert = best_x + np.random.randn(N, 2) * 0.01
            if np.any(np.linalg.norm(x_pert, axis=1) > 1.0):
                continue
            r_pert = calc_radius(x_pert)
            if r_pert > best_r:
                best_r = r_pert
                best_x = x_pert
                
    return best_x

# The task requires keeping function names. The seed had `construct`.
# I will define `construct` to call the implementation.
