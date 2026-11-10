"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def calculate_score(points):
        """Calculate the maximum common radius for a given set of points."""
        if len(points) < 2:
            return 1.0
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.min(np.sqrt(dist_sq))
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)
        return min(min_pair_dist / 2.0, min_bound_dist)

    def gen_hex_lattice(spacing, jitter=0.0):
        """Generate hexagonal lattice points inside the unit circle with optional jitter."""
        k = int(1.0 / spacing) + 2
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if jitter > 0:
                    x += np.random.uniform(-jitter, jitter)
                    y += np.random.uniform(-jitter, jitter)
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        return np.array(pts) if pts else np.empty((0, 2))

    def gen_random_points(n, margin=0.05):
        """Generate n random points inside the unit circle with margin."""
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, (1.0 - margin)**2, n))
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def gen_ring_points(n, num_rings=None):
        """Generate points on concentric rings."""
        if num_rings is None:
            num_rings = int(np.sqrt(n))
        pts = [(0.0, 0.0)]
        remaining = n - 1
        for ring in range(1, num_rings + 1):
            r = ring / num_rings * 0.9
            count = min(remaining, max(1, int(2 * np.pi * r / 0.1)))
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + np.random.uniform(-0.1, 0.1)
            pts.extend([(r * np.cos(a), r * np.sin(a)) for a in angles])
            remaining -= count
            if remaining <= 0:
                break
        return np.array(pts[:n]) if len(pts) >= n else np.vstack([np.array(pts), gen_random_points(n - len(pts))])[:n]

    def epigraph_objective_and_grad(flat_x, beta=50.0):
        """
        Smooth epigraph formulation: minimize sum of soft-min violations.
        Variables: x (N*2) flattened.
        Objective: sum_i softpenalty(d_ij/2 - t) + sum_i softpenalty(1-|x_i| - t) + large_penalty*t
        where softpenalty(v) = log(1+exp(beta*v)) approximates max(0,v) smoothly.
        We minimize over x only, with t determined implicitly as the max violation.
        Simpler: use log-sum-exp to approximate the min.
        """
        x = flat_x.reshape((N, 2))
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # Boundary distances
        r_i = np.linalg.norm(x, axis=1)
        d_b = 1.0 - r_i
        
        # All constraint values: we want all >= t
        # For pairs: d_ij / 2 >= t
        # For boundary: 1 - |x_i| >= t
        # Soft min: -logsumexp(-beta * values) / beta approximates min(values)
        
        # Pair constraint values
        pair_vals = dist / 2.0
        # Exclude diagonal (already inf)
        # Use log-sum-exp for numerical stability
        beta_p = beta
        # Soft min of pair distances
        min_pair_soft = -np.log(np.sum(np.exp(-beta_p * pair_vals))) / beta_p
        
        # Boundary constraint values
        beta_b = beta
        min_bound_soft = -np.log(np.sum(np.exp(-beta_b * d_b))) / beta_b
        
        # Objective: minimize -(min_pair_soft + min_bound_soft) to maximize the min
        # But we need a smooth objective. Let's use:
        # E = -min_pair_soft - min_bound_soft + lambda * max(0, -min_pair_soft) ... 
        # Simpler: just minimize the negative of the soft-min
        # Actually, let's use a different smooth surrogate:
        # E = sum_{i<j} max(0, 2*t - d_ij)^2 + sum_i max(0, t - (1-|x_i|))^2 + t^2 * (-1)
        # This is still non-smooth. Let's use the log-sum-exp approach directly.
        
        # Maximize min(constraints) => minimize -min(constraints)
        # Soft approximation: -min(constraints) ≈ logsumexp(-beta*constraints)/beta
        
        # Combine all constraints
        # Pair constraints (N*(N-1)/2 values) and boundary constraints (N values)
        # For efficiency, compute separately
        
        # Gradient of soft min:
        # d/dx_i [-logsumexp(-beta*v)/beta] = sum_k exp(-beta*v_k) * beta * (dv_k/dx_i) / (beta * sum_m exp(-beta*v_m))
        # = sum_k softmax_k * dv_k/dx_i
        
        # For pairs: v_k = d_ij/2, dv/dx_i = (x_i - x_j)/d_ij
        # For boundary: v_k = 1-|x_i|, dv/dx_i = -x_i/|x_i|
        
        # Compute softmax weights
        exp_neg_pair = np.exp(-beta_p * pair_vals)
        sum_exp_pair = np.sum(exp_neg_pair)
        softmax_pair = exp_neg_pair / sum_exp_pair
        
        exp_neg_bound = np.exp(-beta_b * d_b)
        sum_exp_bound = np.sum(exp_neg_bound)
        softmax_bound = exp_neg_bound / sum_exp_bound
        
        # Objective value
        obj = -min_pair_soft - min_bound_soft
        
        # Gradient
        # grad = -d(min_pair_soft)/dx - d(min_bound_soft)/dx
        # d(min_pair_soft)/dx_i = sum_{j!=i} softmax_pair[i,j] * (x_i - x_j)/d_ij
        # Note: softmax_pair is symmetric, and we sum over j for each i
        
        # Compute gradient for pairs
        # (x_i - x_j)/d_ij = diff / dist
        with np.errstate(divide='ignore', invalid='ignore'):
            dir_pair = diff / dist[:, :, np.newaxis]
        np.fill_diagonal(dir_pair[:, :, 0], 0.0)
        np.fill_diagonal(dir_pair[:, :, 1], 0.0)
        
        # grad_pair[i] = sum_j softmax_pair[i,j] * dir_pair[i,j]
        grad_pair = np.sum(softmax_pair[:, :, np.newaxis] * dir_pair, axis=1)
        
        # Boundary gradient
        with np.errstate(divide='ignore', invalid='ignore'):
            dir_bound = -x / r_i[:, np.newaxis]
        dir_bound[np.abs(r_i) < 1e-9] = 0.0
        grad_bound = softmax_bound[:, np.newaxis] * dir_bound
        
        total_grad = -(grad_pair + grad_bound)
        
        return obj, total_grad

    def local_polish(points, max_iters=200, step_size=0.0005):
        """Coordinate-descent local polish on the bottleneck constraint."""
        current = points.copy()
        current_score = calculate_score(current)
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            # Find bottleneck
            diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_pair_dist = np.min(np.sqrt(dist_sq))
            
            r_i = np.linalg.norm(current, axis=1)
            min_bound_dist = np.min(1.0 - r_i)
            
            pair_r = min_pair_dist / 2.0
            bound_r = min_bound_dist
            
            if pair_r < bound_r:
                # Pair is limiting
                idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
                i, j = idx
                if i == j:
                    continue
                direction = current[i] - current[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-9:
                    direction /= norm
                    current[i] += direction * step_size
                    current[j] -= direction * step_size
            else:
                # Boundary is limiting
                idx = np.argmin(1.0 - r_i)
                point = current[idx]
                norm = np.linalg.norm(point)
                if norm > 1e-9:
                    direction = -point / norm
                    current[idx] += direction * step_size
            
            new_score = calculate_score(current)
            if new_score > current_score:
                current_score = new_score
            else:
                # Revert
                if pair_r < bound_r:
                    idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
                    i, j = idx
                    if i != j:
                        direction = current[i] - current[j]
                        norm = np.linalg.norm(direction)
                        if norm > 1e-9:
                            direction /= norm
                            current[i] -= direction * step_size
                            current[j] += direction * step_size
                else:
                    idx = np.argmin(1.0 - r_i)
                    point = current[idx]
                    norm = np.linalg.norm(point)
                    if norm > 1e-9:
                        direction = -point / norm
                        current[idx] -= direction * step_size
        
        return current, current_score

    # 1. Generate multiple initializations
    spacings = [
        np.sqrt(2.0 * np.pi / (np.sqrt(3.0) * N)) * 0.95,
        np.sqrt(2.0 * np.pi / (np.sqrt(3.0) * N)) * 1.0,
        np.sqrt(2.0 * np.pi / (np.sqrt(3.0) * N)) * 1.05,
    ]
    
    inits = []
    
    # Hex lattice variants
    for spacing in spacings:
        points = gen_hex_lattice(spacing)
        while len(points) < N and spacing > 0.01:
            spacing *= 0.95
            points = gen_hex_lattice(spacing)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            sorted_idx = np.argsort(dists)
            inits.append(points[sorted_idx[:N]])
        elif len(points) > 0:
            # Pad with random
            pad = gen_random_points(N - len(points), margin=0.1)
            inits.append(np.vstack([points, pad])[:N])
    
    # Random initialization
    inits.append(gen_random_points(N, margin=0.05))
    
    # Ring initialization
    inits.append(gen_ring_points(N))
    
    # Jittered hex
    for spacing in spacings[:1]:
        points = gen_hex_lattice(spacing, jitter=spacing * 0.1)
        while len(points) < N and spacing > 0.01:
            spacing *= 0.95
            points = gen_hex_lattice(spacing, jitter=spacing * 0.1)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            sorted_idx = np.argsort(dists)
            inits.append(points[sorted_idx[:N]])
    
    best_points = None
    best_score = -np.inf
    
    # 2. Optimize each initialization
    for init_idx, init_points in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        # Validate initialization
        if not np.all(np.isfinite(init_points)):
            continue
        r_i = np.linalg.norm(init_points, axis=1)
        if np.any(r_i > 1.0):
            # Push inside
            init_points = init_points / np.maximum(r_i[:, np.newaxis], 0.99) * 0.99
        
        init_score = calculate_score(init_points)
        
        # Try L-BFGS-B with epigraph objective
        try:
            res = minimize(
                epigraph_objective_and_grad,
                init_points.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 1500, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            optimized = res.x.reshape((N, 2))
            
            if np.all(np.isfinite(optimized)):
                r_opt = np.linalg.norm(optimized, axis=1)
                if np.all(r_opt <= 1.0):
                    opt_score = calculate_score(optimized)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = optimized.copy()
        except Exception:
            pass
        
        # Local polish
        if time.time() < deadline - 5.0:
            start_pts = best_points.copy() if best_score > init_score else init_points.copy()
            polished, polish_score = local_polish(start_pts, max_iters=100)
            if polish_score > best_score:
                best_score = polish_score
                best_points = polished.copy()
        
        # If this init was better, do another round of polish
        if time.time() < deadline - 3.0 and best_points is not None:
            polished2, polish_score2 = local_polish(best_points.copy(), max_iters=50)
            if polish_score2 > best_score:
                best_score = polish_score2
                best_points = polished2.copy()
    
    # Final polish
    if best_points is not None and time.time() < deadline - 1.0:
        final_pts, final_score = local_polish(best_points.copy(), max_iters=100)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
    
    return best_points
# EVOLVE-BLOCK-END
