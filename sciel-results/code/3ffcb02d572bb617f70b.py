import numpy as np
import time
import sys

N = 300

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s limit

    def _elapsed():
        return time.time() - start_time

    def _dist_matrix(pts):
        """Compute pairwise distance matrix."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def _score(pts):
        """Compute the objective: min(pair_dist/2, 1-|c|)."""
        if len(pts) != N:
            return -1.0
        dmat = _dist_matrix(pts)
        np.fill_diagonal(dmat, np.inf)
        min_pair = np.min(dmat)
        r_vec = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - r_vec)
        return min(min_pair, min_boundary) / 2.0

    def _hex_lattice_init(spacing):
        """Generate hexagonal lattice points within unit circle (with margin for radius)."""
        points = []
        sqrt3 = np.sqrt(3.0)
        k = int(1.0 / spacing) + 3
        for j in range(-k, k + 1):
            y = j * spacing * sqrt3 / 2.0
            x_off = (j % 2) * spacing / 2.0
            for i in range(-k, k + 1):
                x = (i + x_off) * spacing
                if np.hypot(x, y) <= 1.0:
                    points.append((x, y))
        pts = np.array(points)
        # Sort by distance to center, keep closest N
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')
        pts = pts[idx]
        if len(pts) >= N:
            pts = pts[:N]
        else:
            # Pad with random points if needed (shouldn't happen for N=300)
            while len(pts) < N:
                angle = np.random.uniform(0, 2 * np.pi)
                rad = np.random.uniform(0, 0.95)
                pts = np.vstack([pts, [rad * np.cos(angle), rad * np.sin(angle)]])
        return pts

    def _soft_min(dists, m=50.0):
        """Soft minimum of distances (excluding diagonal)."""
        d = dists.copy()
        np.fill_diagonal(d, np.inf)
        # Add boundary terms
        return d

    def _smooth_objective(pts, temperature=0.01):
        """
        Smooth objective: maximize soft-min of pairwise distances and boundary distances.
        Uses log-sum-exp trick for stability.
        """
        dmat = _dist_matrix(pts)
        np.fill_diagonal(dmat, np.inf)
        
        r_vec = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - r_vec
        
        # Combine all distances: pair distances + boundary distances
        # We want max of min of these.
        # Use soft-min: -log-sum-exp(-d / t) * t
        
        all_dists = np.concatenate([dmat[dmat < np.inf].ravel(), boundary_dists])
        
        # For numerical stability, shift by min
        d_min = np.min(all_dists)
        shifted = all_dists - d_min
        # soft_min = d_min - t * log(sum(exp(-shifted/t)))
        # But we want to maximize this, so we minimize -soft_min
        t = temperature
        log_sum = np.log(np.sum(np.exp(-shifted / t)))
        soft_min_val = d_min - t * log_sum
        return soft_min_val

    def _grad_smooth_objective(pts, temperature=0.01):
        """
        Gradient of the smooth objective with respect to positions.
        Returns (objective_value, gradient_array).
        """
        dmat = _dist_matrix(pts)
        np.fill_diagonal(dmat, np.inf)
        
        r_vec = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - r_vec
        
        # All distance values
        n_pairs = N * (N - 1) // 2
        all_dists = np.empty(n_pairs + N)
        all_dists[:n_pairs] = dmat[np.triu_indices(N, k=1)]
        all_dists[n_pairs:] = boundary_dists
        
        d_min = np.min(all_dists)
        t = temperature
        shifted = all_dists - d_min
        exp_neg = np.exp(-shifted / t)
        sum_exp = np.sum(exp_neg)
        
        # soft_min = d_min - t * log(sum_exp)
        obj_val = d_min - t * np.log(sum_exp)
        
        # Gradient: d(soft_min)/d(pts)
        # soft_min = d_min - t * log(sum_exp)
        # d(soft_min)/d(x_k) = sum_j [w_j * d(dist_j)/d(x_k)]
        # where w_j = exp(-shifted_j/t) / sum_exp
        # But d_min also depends on pts... For simplicity, use the standard soft-min gradient:
        # grad = sum_j [ (exp(-d_j/t)/sum_exp) * grad(dist_j) ]
        # (ignoring the d_min derivative which is a bit subtle but negligible for smooth optimization)
        
        weights = exp_neg / sum_exp  # shape: (n_pairs + N,)
        
        grad = np.zeros_like(pts)
        
        # Pair distance gradients
        # For pair (i,j), dist = |p_i - p_j|
        # d(dist)/d(p_i) = (p_i - p_j)/dist
        # d(dist)/d(p_j) = (p_j - p_i)/dist
        
        pair_idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                w = weights[pair_idx]
                if w < 1e-12:
                    pair_idx += 1
                    continue
                v = pts[i] - pts[j]
                d = np.linalg.norm(v)
                if d < 1e-12:
                    pair_idx += 1
                    continue
                grad_v = v / d
                grad[i] += w * grad_v
                grad[j] -= w * grad_v
                pair_idx += 1
        
        # Boundary distance gradients
        # boundary_dist_k = 1 - |p_k|
        # d(boundary_dist_k)/d(p_k) = -p_k/|p_k|
        for k in range(N):
            w = weights[n_pairs + k]
            if w < 1e-12:
                continue
            rk = r_vec[k]
            if rk < 1e-12:
                continue
            grad[k] += w * (-pts[k] / rk)
        
        return obj_val, grad

    # Initial placement using hexagonal lattice
    # Estimate spacing: area per circle ~ pi * r^2, total area pi
    # N * pi * r^2 <= pi => r <= 1/sqrt(N)
    # Spacing ~ 2r ~ 2/sqrt(N) ~ 0.115 for N=300
    base_spacing = 2.0 / np.sqrt(N) * 0.95  # Slightly tighter
    
    # Try multiple initial spacings
    best_points = None
    best_score = -1.0

    # Phase 1: Try several lattice initializations with different spacings
    spacings = [base_spacing * factor for factor in [0.85, 0.90, 0.95, 1.0, 1.05, 1.1]]
    
    for spacing in spacings:
        if _elapsed() > deadline - 20:
            break
        pts = _hex_lattice_init(spacing)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Phase 2: Smooth gradient-based optimization
    # Use the best initial point
    pts = best_points.copy()
    
    # Adaptive temperature schedule
    t_start = 0.05
    t_end = 0.001
    max_iter = 2000
    
    # Simple gradient ascent with adaptive step size
    step = 0.005
    best_iter_score = _score(pts)
    best_iter_points = pts.copy()
    
    no_improve_count = 0
    patience = 300
    
    for it in range(max_iter):
        if _elapsed() > deadline - 5:
            break
        
        obj_val, grad = _grad_smooth_objective(pts, temperature=t_start * 0.5)
        
        # Gradient ascent
        pts_new = pts + step * grad
        
        # Ensure points stay within unit circle (soft projection)
        r_vec = np.hypot(pts_new[:, 0], pts_new[:, 1])
        for k in range(N):
            if r_vec[k] > 0.999:
                scale = 0.999 / r_vec[k]
                pts_new[k] *= scale
        
        s_new = _score(pts_new)
        
        if s_new > best_iter_score:
            best_iter_score = s_new
            best_iter_points = pts_new.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        pts = pts_new
        
        # Adaptive step size
        if no_improve_count > patience:
            step *= 0.5
            no_improve_count = 0
            if step < 1e-6:
                break
        
        # Periodically re-evaluate with actual score to track progress
        if it % 10 == 0:
            if _score(pts) > best_score:
                best_score = _score(pts)
                best_points = pts.copy()
    
    if best_iter_score > best_score:
        best_score = best_iter_score
        best_points = best_iter_points.copy()

    # Phase 3: Local polishing - find closest pair and push apart
    if _elapsed() < deadline - 10:
        pts = best_points.copy()
        polish_iters = 500
        for it in range(polish_iters):
            if _elapsed() > deadline - 5:
                break
            
            dmat = _dist_matrix(pts)
            np.fill_diagonal(dmat, np.inf)
            r_vec = np.hypot(pts[:, 0], pts[:, 1])
            
            min_pair = np.min(dmat)
            min_boundary = np.min(1.0 - r_vec)
            current_score = min(min_pair, min_boundary) / 2.0
            
            if current_score > best_score:
                best_score = current_score
                best_points = pts.copy()
            
            # Find closest pair
            min_idx = np.unravel_index(np.argmin(dmat), dmat.shape)
            i, j = min_idx[0], min_idx[1]
            
            # Check if boundary is the limiting factor
            if min_boundary < min_pair:
                # Find point closest to boundary
                k = np.argmin(1.0 - r_vec)
                # Push it inward
                if r_vec[k] > 1e-12:
                    push = -pts[k] / r_vec[k] * 0.001
                    pts[k] += push
                    # Project back if needed
                    rk = np.hypot(pts[k, 0], pts[k, 1])
                    if rk > 0.999:
                        pts[k] *= 0.999 / rk
            else:
                # Push closest pair apart
                v = pts[i] - pts[j]
                d = np.linalg.norm(v)
                if d < 1e-12:
                    # Random small push
                    pts[i] += np.random.normal(0, 0.001, 2)
                    pts[j] -= np.random.normal(0, 0.001, 2)
                else:
                    v_unit = v / d
                    push = 0.001
                    pts[i] += v_unit * push
                    pts[j] -= v_unit * push
                    # Project both back if outside
                    for k in [i, j]:
                        rk = np.hypot(pts[k, 0], pts[k, 1])
                        if rk > 0.999:
                            pts[k] *= 0.999 / rk
    
    # Final score check
    final_score = _score(best_points)
    if final_score > best_score:
        best_score = final_score

    # Validate
    if best_points is None or len(best_points) != N:
        # Fallback: regular hex lattice
        spacing = 2.0 / np.sqrt(N) * 0.95
        best_points = _hex_lattice_init(spacing)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(best_points)):
        spacing = 2.0 / np.sqrt(N) * 0.95
        best_points = _hex_lattice_init(spacing)

    return best_points

# EVOLVE-BLOCK-END
