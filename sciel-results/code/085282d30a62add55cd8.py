"""
Circle packing via force-directed relaxation + vectorized gradient ascent.
Strategy:
1. Initialize with a structured lattice (hexagonal or ring-based).
2. Run a force-directed simulation where circles repel each other and are attracted to the center,
   but with a strong repulsion core that enforces non-overlap. This finds a "natural" packing.
3. Polish the result using gradient ascent on the soft-minimum radius.
   The objective is to maximize r such that all constraints (dist >= 2r, dist_from_origin <= 1-r) hold.
   We use a vectorized gradient computation for speed.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill


def _compute_score(pts):
    """Compute the true packing radius (nonsmooth)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Vectorized pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(pts ** 2, axis=-1))
    min_margin = np.min(1.0 - radii)
    return min(min_pair, min_margin)


def _softmin(vals, beta):
    """Log-sum-exp approximation of min(vals)."""
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return 0.0
    v_max = np.max(vals)
    # Avoid overflow
    log_sum = np.log(np.sum(np.exp(beta * (vals - v_max)))) / beta
    return v_max + log_sum


def _force_relax(pts, iterations=500, dt=0.01, repulsion_strength=1.0, center_attraction=0.05):
    """
    Force-directed relaxation.
    Forces:
    1. Pairwise repulsion: F_ij = k * (1/d_ij^2) * unit_vec if d_ij < 2*r_est else 0
       Actually, to maximize radius, we want to push them apart until they touch.
       Let's use a potential U = -1/d. Force = -grad U = -1/d^2.
       But we need to balance with boundary.
       
       Better approach: 
       Objective to maximize: min( min_ij d_ij/2, min_i (1-|c_i|) )
       
       Force-directed:
       Repulsion between i,j: F = (1/d_ij^2) * (r_i - r_j)/d_ij  (pushes apart)
       Attraction to center: F = -k * r_i (pulls in)
       Boundary constraint: If |r_i| > 1 - r_est, push in strongly.
       
       Let's use a simpler "expand until touch" logic.
       Start with small r. Increase r? No, r is determined by positions.
       
       Let's just use a repulsive force that is always active but decays.
       F_repulse_ij = k_rep * (1/d_ij^2) * unit_vec_ij
       F_attract_i = k_attr * r_i  (towards origin)
       
       This will naturally find a configuration where repulsion balances attraction,
       which corresponds to a local maximum of the minimum distance.
    """
    pts = pts.copy()
    n = len(pts)
    
    # Estimate current radius to scale forces
    # We want the forces to be significant relative to the scale of the problem.
    
    for _ in range(iterations):
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        # Vectorized calculation of forces is complex, so we do a simple loop for stability
        # or use a vectorized approach if n is small enough. N=50 is small.
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-6:
                    # Random jitter
                    diff = np.random.normal(0, 0.01, 2)
                    d = np.linalg.norm(diff)
                # Force magnitude: k / d^2
                # We want this to be strong enough to separate circles.
                # Scale with 1/d^3 to get unit vector * 1/d^2
                f_mag = repulsion_strength / (d * d * d)
                f_vec = diff * f_mag
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Attraction to center
        # F = -k * pos
        forces -= center_attraction * pts
        
        # Boundary constraint: push in if outside 0.9
        radii = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if radii[i] > 0.9:
                # Strong push towards center
                forces[i] -= 10.0 * (pts[i] / radii[i])
        
        # Update positions
        pts += dt * forces
        
        # Project back inside unit circle (hard constraint)
        radii = np.linalg.norm(pts, axis=1)
        mask = radii > 0.95
        if np.any(mask):
            pts[mask] *= 0.95 / radii[mask]
            
    return pts


def _gradient_ascent_polish(pts, time_limit, beta=100.0):
    """
    Maximize the soft-min radius using gradient ascent.
    Objective: f(x) = softmin( [d_ij/2 - r_target]_ij, [1 - r_target - |c_i|]_i )
    Actually, we want to maximize r.
    Let g(x) = min( min_ij d_ij/2, min_i (1 - |c_i|) ).
    We want to maximize g(x).
    
    Using softmin: S(x, beta) = -1/beta * log( sum exp(-beta * v_k) ) where v_k are the constraint values.
    Wait, softmin(v) approx min(v).
    We want to maximize min(v_k).
    So we minimize -softmin(v_k).
    
    Let's define the objective for L-BFGS-B (which minimizes).
    obj(x) = - softmin( v_k(x), beta )
    
    v_k for pairs: d_ij / 2
    v_k for boundary: 1 - |c_i|
    
    Gradient of softmin:
    softmin(v) = v_max - (1/beta) log( sum exp(-beta (v_k - v_max)) ) ... no, standard formula:
    softmin(v, beta) = - (1/beta) log( sum exp(-beta v_k) )
    
    Let M = max(v).
    softmin = M - (1/beta) log( sum exp(-beta (v_k - M)) )
    
    grad(softmin) w.r.t x:
    Let w_k = exp(-beta (v_k - M)) / sum_j exp(-beta (v_j - M))
    grad(softmin) = sum_k w_k * grad(v_k)
    
    We want to maximize softmin, so we minimize -softmin.
    grad(obj) = - grad(softmin) = - sum_k w_k * grad(v_k)
    
    v_ij = d_ij / 2. grad(v_ij) w.r.t c_i = (c_i - c_j) / (2 * d_ij)
    v_i = 1 - |c_i|. grad(v_i) w.r.t c_i = -c_i / |c_i|
    """
    
    def obj_and_grad(x):
        pts = x.reshape(-1, 2)
        n = len(pts)
        
        # Compute all constraint values v_k
        # Pairs
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        v_pairs = dists / 2.0
        v_pairs = v_pairs.flatten()
        # Remove diagonal (inf)
        v_pairs = v_pairs[v_pairs < 1e15]
        
        # Boundary
        radii = np.sqrt(np.sum(pts ** 2, axis=-1))
        v_bound = 1.0 - radii
        
        # Combine
        all_v = np.concatenate([v_pairs, v_bound])
        
        if len(all_v) == 0:
            return 0.0, np.zeros_like(x)
            
        M = np.max(all_v)
        exps = np.exp(-beta * (all_v - M))
        sum_exps = np.sum(exps)
        weights = exps / sum_exps
        
        obj_val = -(M - np.log(sum_exps) / beta)
        
        # Gradient
        grad = np.zeros_like(x)
        
        # Pair contributions
        # We need to map weights back to pairs
        # This is tricky with flattened arrays. Let's reconstruct or use a different approach.
        # Since N=50, 1225 pairs + 50 bounds = 1275 values.
        # We can compute the gradient contribution for each pair directly.
        
        # Recalculate weights for pairs and bounds separately to simplify indexing?
        # No, the softmin is over the union. The weights are global.
        
        # Let's iterate over pairs for gradient.
        # It's O(N^2) which is 2500 ops, fine for Python loop if not inside a tight inner loop of many iterations.
        # L-BFGS might call this 100-500 times. 500 * 2500 = 1.25M ops. Fine.
        
        # Pair gradients
        # w_pair[i,j] corresponds to v_pair[i,j]
        # We need to extract the weights for pairs.
        # The first len(v_pairs) weights in `weights` correspond to the flattened upper triangle of dists?
        # No, `v_pairs` was flattened from `dists` which is full matrix with inf diagonal.
        # `v_pairs = v_pairs[v_pairs < 1e15]` removes the diagonal.
        # The order is row-major.
        
        # Let's just compute the gradient term by term.
        # For pair (i, j) with i < j:
        #   v = dists[i,j] / 2
        #   w = weight corresponding to this v
        #   grad_v w.r.t c_i = (c_i - c_j) / dists[i,j]
        #   grad_v w.r.t c_j = (c_j - c_i) / dists[i,j]
        
        # We need to map the index in `weights` to (i, j).
        # This is error-prone. Let's use a different strategy:
        # Compute the gradient of the softmin directly using the full matrices where possible.
        
        # Actually, let's just use the fact that:
        # grad(softmin) = sum w_k grad(v_k)
        
        # Let's compute the pair part and bound part separately, but we need the global weights.
        
        # Alternative: Use a slightly different objective that is easier to differentiate.
        # Maximize sum of exp(-beta * v_k) ? No, that's not min.
        
        # Let's stick to the loop for clarity and correctness, N=50 is small.
        
        # Map weights to pairs
        # The flattened v_pairs comes from dists (N x N) with diagonal removed.
        # Index k in v_pairs corresponds to some (i, j).
        
        # Let's pre-compute the indices.
        # This is expensive to do every iteration.
        
        # Better: Compute the gradient contribution analytically.
        # Let P be the matrix of pair values (N x N), with 0 on diagonal.
        # Let B be the vector of boundary values (N).
        # V = [P.flatten() (diag removed), B]
        
        # Let's just compute the gradient of the "effective" potential.
        
        # For simplicity and robustness, let's use a simpler objective for the polish:
        # Minimize sum of squared violations? No, that's not max-min.
        
        # Let's use the loop.
        # To get the weight for pair (i, j):
        # We need to know which index in the flattened array it is.
        
        # Let's reconstruct the weight matrix for pairs.
        # w_pairs_matrix = np.zeros((N, N))
        # We need to map the 1D weights back to 2D.
        
        # This is getting complicated. Let's use a different optimization method for the polish.
        # SLSQP with explicit constraints?
        # Constraints: d_ij >= 2r, |c_i| <= 1-r.
        # Variables: c_i, r.
        # Maximize r.
        
        # This is a standard QP/NLP.
        # But L-BFGS-B is fast.
        
        # Let's try a different formulation for the gradient:
        # Use the "top-k" approximation?
        
        # Actually, for N=50, a simple Python loop for the gradient is fast enough.
        # 1275 items.
        
        # Let's create a list of (weight, grad_vec) for each constraint.
        
        # We need to align `weights` with the constraints.
        # `all_v` was constructed as `np.concatenate([v_pairs, v_bound])`.
        # `v_pairs` was `dists.flatten()` with diagonal removed.
        
        # Let's identify the indices.
        # It's easier to just compute the gradient by iterating over i, j and checking if it was included.
        # But we don't have the mapping.
        
        # Let's change the construction of `all_v` to keep the mapping.
        
        # REWRITE:
        # Instead of flattening and filtering, let's keep the structure.
        
        # This function is called inside `minimize`, so it needs to be fast.
        
        # Let's use a vectorized gradient computation for the pair part.
        # grad_softmin_pairs = sum_{i,j} w_{ij} * (c_i - c_j) / d_ij
        # We can compute the weight matrix W_pairs (N x N) from the 1D weights if we know the mapping.
        
        # Mapping:
        # dists is N x N.
        # v_pairs = dists.flatten()
        # mask = dists.flatten() < 1e15
        # v_pairs_clean = v_pairs[mask]
        # weights_clean = weights[mask]  # IF weights were computed on the full array with inf handled?
        # No, weights were computed on `all_v` which is clean.
        
        # So `weights` corresponds to `v_pairs_clean` and `v_bound`.
        
        # Let's just use the loop. It's simple and less likely to have bugs.
        
        # Get weights for pairs
        n_pairs = len(v_pairs)
        w_pairs = weights[:n_pairs]
        w_bound = weights[n_pairs:]
        
        # Reconstruct weight matrix for pairs
        # We need to map the flattened indices back to (i, j).
        # This is a known problem.
        # Let's just iterate i, j.
        
        # Pre-compute the list of (i, j) indices for the flattened array?
        # No, let's just compute the gradient contribution for each pair (i, j) directly.
        # But we need the specific weight w_{ij}.
        
        # Okay, I will use a different approach for the polish:
        # Use `scipy.optimize.minimize` with method='L-BFGS-B' but define the objective
        # as the negative of the soft-min, and compute the gradient using a finite difference
        # if the analytic gradient is too complex? No, finite difference is slow.
        
        # Let's go back to the analytic gradient with a helper to map indices.
        
        # Helper: get weight for pair (i, j)
        # We can create a sparse matrix or just a dense matrix of weights.
        
        # Let's create a dense weight matrix W (N x N) initialized to 0.
        # And we need to fill it with the values from `w_pairs`.
        # The order of `v_pairs` is row-major of the full matrix with diagonal removed.
        
        # Let's generate the indices.
        # idx = 0
        # for i in range(N):
        #   for j in range(N):
        #     if i == j: continue
        #     W[i, j] = w_pairs[idx]
        #     idx += 1
        
        # This is O(N^2) = 2500 ops. Acceptable.
        
        W = np.zeros((N, N))
        idx = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                W[i, j] = w_pairs[idx]
                idx += 1
        
        # Now compute gradient of pair part
        # grad_i = sum_j W[i, j] * (c_i - c_j) / d_ij
        # Note: d_ij is in dists.
        
        # Vectorized:
        # G[i, j] = W[i, j] * (c_i - c_j) / d_ij
        # grad_pairs[i] = sum_j G[i, j]
        
        # Handle division by zero (diagonal is inf, W is 0 there, so 0*inf is nan? No, we skip i=j)
        
        # Let's compute G
        # diffs is N x N x 2
        # dists is N x N
        # W is N x N
        
        # G = W[:,:,np.newaxis] * diffs / dists[:,:,np.newaxis]
        # But dists diagonal is inf. W diagonal is 0. 0/inf = 0.
        # However, if dists is 0 (coincident points), we have 0/0.
        # We assume points are distinct.
        
        # Let's use np.where to avoid issues.
        # Safe dists
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        
        G = W[:, :, np.newaxis] * diffs / safe_dists[:, :, np.newaxis]
        
        # Set diagonal to 0 explicitly to avoid any numerical noise
        np.fill_diagonal(G, 0.0)
        
        grad_pairs = np.sum(G, axis=1)
        
        # Boundary part
        # w_bound is N
        # grad_bound_i = w_bound[i] * (-c_i / |c_i|)
        
        safe_radii = np.where(radii > 1e-9, radii, 1.0)
        grad_bound = w_bound[:, np.newaxis] * (-pts / safe_radii[:, np.newaxis])
        
        # Total gradient of softmin
        grad_softmin = grad_pairs + grad_bound
        
        # Objective is -softmin
        grad_obj = -grad_softmin
        
        return obj_val, grad_obj.flatten()
    
    t0 = time.time()
    x0 = pts.flatten().astype(np.float64)
    
    # Run L-BFGS-B
    try:
        res = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12,
                                'maxls': 50})
        x_opt = res.x
        pts_opt = x_opt.reshape(-1, 2)
    except Exception:
        pts_opt = pts
    
    return pts_opt


def _hex_init(spacing):
    """Hexagonal lattice initialization."""
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) >= N:
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[np.argsort(radius, kind='stable')[:N]]
    return points


def _ring_init(n_rings=8):
    """Concentric rings initialization."""
    pts = [np.array([0.0, 0.0])]
    for ring in range(1, n_rings):
        r = ring * 0.12
        n_in_ring = min(6 * ring, N - len(pts))
        if n_in_ring <= 0:
            break
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
        for a in angles:
            pts.append(np.array([r * np.cos(a), r * np.sin(a)]))
    pts = np.array(pts[:N])
    return pts


def construct():
    t_start = time.time()
    
    best_pts = None
    best_score = -1.0
    
    # Multiple initializations
    inits = [
        _hex_init(0.25),
        _hex_init(0.22),
        _hex_init(0.28),
        _ring_init(8),
        _ring_init(7),
    ]
    
    # Add random inits
    rng = np.random.RandomState(42)
    for _ in range(3):
        pts = rng.uniform(-1, 1, size=(N, 2))
        r = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[r <= 0.95]
        if len(pts) < N:
            # Pad if necessary
            while len(pts) < N:
                pts = np.vstack([pts, rng.uniform(-1, 1, size=(1, 2)) * 0.9])
        inits.append(pts[:N])
        
    for init_idx, init_pts in enumerate(inits):
        if time.time() - t_start > DEADLINE - 20:
            break
        if len(init_pts) < N:
            continue
            
        # 1. Force Relaxation
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_relax = min(time_remaining * 0.3, 5.0)
        iters = int(time_for_relax * 100) # ~100 iters per second
        
        pts_relaxed = _force_relax(init_pts, iterations=iters, dt=0.01, repulsion_strength=0.5, center_attraction=0.05)
        
        score_relaxed = _compute_score(pts_relaxed)
        
        # 2. Gradient Polish
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_polish = min(time_remaining * 0.4, 10.0)
        
        if time_for_polish > 1.0:
            pts_polished = _gradient_ascent_polish(pts_relaxed, time_limit=time_for_polish, beta=50.0)
            score_polished = _compute_score(pts_polished)
            
            if score_polished > score_relaxed:
                pts_best = pts_polished
                score_best = score_polished
            else:
                pts_best = pts_relaxed
                score_best = score_relaxed
        else:
            pts_best = pts_relaxed
            score_best = score_relaxed
            
        if score_best > best_score:
            best_score = score_best
            best_pts = pts_best.copy()
            
    if best_pts is None:
        best_pts = _hex_init(0.25)
        best_score = _compute_score(best_pts)
        
    # Final high-precision polish on the best found
    if time.time() - t_start < DEADLINE - 5:
        time_limit = DEADLINE - (time.time() - t_start) - 2
        if time_limit > 1.0:
            pts_final = _gradient_ascent_polish(best_pts, time_limit=time_limit, beta=200.0)
            score_final = _compute_score(pts_final)
            if score_final > best_score:
                best_score = score_final
                best_pts = pts_final.copy()
                
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_compute_score(pts)}")
