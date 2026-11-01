"""
Combined packer: smooth soft-min optimization (L-BFGS-B) with KD-tree local search polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 145.0  # safety margin before the 150s wall-clock limit


def _min_dist(points):
    """Efficiently compute the minimum pairwise distance."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(np.min(dists[:, 1]))


def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    # minimum pairwise half-distance
    min_pair_dist = _min_dist(points)
    r_pair = 0.5 * min_pair_dist
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            # prefer inner points (smaller radius) to maximise margin
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _local_search(points, t_budget, rng):
    """
    Fast local search using targeted moves on constrained points.
    Focuses on points involved in minimum-distance pairs.
    """
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    cur_r = best_r
    
    # Initial step size based on typical spacing
    step = 0.005
    
    # Build initial KD-tree for neighbor queries
    tree = cKDTree(cur)
    
    while time.time() - start < t_budget:
        # Find the minimum distance pair and boundary-violating points
        norms = np.linalg.norm(cur, axis=1)
        r_bound = 1.0 - norms
        min_r_bound_idx = np.argmin(r_bound)
        
        # Find closest pair using KD-tree
        dists, indices = tree.query(cur, k=2)
        min_pair_idx = np.argmin(dists[:, 1])
        
        # Choose which point to move: weighted random among constrained points
        constrained_pts = [min_pair_idx, indices[min_pair_idx, 1], min_r_bound_idx]
        # Add a few random points for exploration
        for _ in range(3):
            constrained_pts.append(rng.integers(N))
        
        idx = constrained_pts[rng.integers(len(constrained_pts))]
        
        # Determine move direction: away from nearest neighbor or toward center
        if idx == min_r_bound_idx and r_bound[idx] < 0.05:
            # Move inward
            norm_val = np.linalg.norm(cur[idx])
            if norm_val > 1e-10:
                direction = -cur[idx] / norm_val
            else:
                direction = np.array([1.0, 0.0])
        else:
            # Move away from nearest neighbor
            nn_idx = indices[idx, 1]
            diff = cur[idx] - cur[nn_idx]
            nn_dist = np.linalg.norm(diff)
            if nn_dist > 1e-10:
                direction = diff / nn_dist
            else:
                ang = rng.random() * 2.0 * np.pi
                direction = np.array([np.cos(ang), np.sin(ang)])
        
        # Try multiple step sizes
        improved = False
        for mag_factor in [1.0, 0.5, 0.25, 2.0]:
            mag = step * mag_factor
            new_pt = cur[idx] + mag * direction
            
            # Check boundary
            if np.hypot(new_pt[0], new_pt[1]) > 1.0:
                continue
            
            old_pt = cur[idx].copy()
            cur[idx] = new_pt
            
            # Quick score check using KD-tree
            new_min_dist = _min_dist(cur)
            new_r = 0.5 * new_min_dist
            new_r_bound = 1.0 - np.hypot(new_pt[0], new_pt[1])
            new_r = min(new_r, new_r_bound)
            
            if new_r > cur_r + 1e-12:
                cur_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best = cur.copy()
                improved = True
                # Update tree
                tree = cKDTree(cur)
                break
            else:
                cur[idx] = old_pt
        
        if not improved:
            step *= 0.95
            if step < 0.0005:
                step = 0.0005
        else:
            step *= 1.05
            if step > 0.05:
                step = 0.05

    return best, best_r


def _soft_min_optimize(points, t_budget, rng):
    """
    Smooth soft-min optimization using L-BFGS-B.
    Objective: minimize -soft_min(constraint_distances)
    Uses efficient pairwise distance computation.
    """
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    
    # Precompute indices for pairs
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    n_pairs = len(i_idx)
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate pairwise distances efficiently using vectorized ops
        # Only compute upper triangle distances
        pi = p[i_idx]
        pj = p[j_idx]
        diff = pi - pj
        pair_dists = np.sqrt(np.sum(diff**2, axis=-1)) / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Soft-min calculation using log-sum-exp trick for numerical stability
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        all_z = np.concatenate([z_pairs, z_bounds])
        max_z = np.max(all_z)
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        # Objective value: -soft_min = (max_z + log(sum_exp)) / beta
        val = (max_z + np.log(sum_exp)) / beta
        
        # Gradient
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        # Pair contributions
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        coeff = w_pairs * 0.25 / safe_pair_dists
        vecs = diff  # pi - pj
        
        np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
        
        # Boundary contributions
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = -p / safe_norms[:, np.newaxis]
        
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        # Return negative gradient (since we want to minimize -soft_min)
        return val, -grad

    # Adaptive beta schedule: start moderate, increase based on progress
    beta = 30.0
    cur = points.copy().flatten()
    last_improvement = 0.0
    no_improve_count = 0
    
    while time.time() - start < t_budget:
        t_left = t_budget - (time.time() - start)
        if t_left < 3.0:
            break
        
        # Run L-BFGS-B
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                cur,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 1000, 'gtol': 1e-8}
            )
            cur = res.x
        except Exception:
            break
        
        # Check score
        pts = cur.reshape(-1, 2)
        r = _score(pts)
        if r > best_r:
            best_r = r
            best = pts.copy()
            last_improvement = r
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Adaptive beta: increase if no improvement for a while
        if no_improve_count >= 2:
            beta *= 1.8
            if beta > 5000.0:
                beta = 5000.0
            no_improve_count = 0
        
        # If we're close to the deadline, break
        if time.time() - start > t_budget - 2.0:
            break
    
    return best, best_r


def _final_polish(points, t_budget):
    """
    High-precision final polish using subgradient method on exact min-distance.
    """
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    
    lr = 0.002
    decay = 0.99
    
    while time.time() - start < t_budget:
        t_left = t_budget - (time.time() - start)
        if t_left < 0.5:
            break
        
        # Compute exact min distance and boundary margin
        norms = np.linalg.norm(cur, axis=1)
        r_bound = 1.0 - norms
        min_bound_idx = np.argmin(r_bound)
        
        # Find min pairwise distance
        tree = cKDTree(cur)
        dists, indices = tree.query(cur, k=2)
        min_pair_idx = np.argmin(dists[:, 1])
        min_pair_dist = dists[min_pair_idx, 1]
        r_pair = 0.5 * min_pair_dist
        
        # Determine which constraint is active
        if r_bound[min_bound_idx] < r_pair:
            # Boundary is active
            active_idx = min_bound_idx
            # Gradient: push inward
            norm_val = norms[active_idx]
            if norm_val > 1e-10:
                grad = np.zeros((N, 2))
                grad[active_idx] = -cur[active_idx] / norm_val
            else:
                continue
        else:
            # Pair constraint is active
            i = min_pair_idx
            j = indices[min_pair_idx, 1]
            diff = cur[i] - cur[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            grad = np.zeros((N, 2))
            grad[i] = diff / dist
            grad[j] = -diff / dist
        
        # Apply gradient step
        cur += lr * grad
        
        # Project back into unit circle
        rad = np.linalg.norm(cur, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cur[mask] = cur[mask] / rad[mask, None] * (1.0 - 1e-9)
        
        # Check improvement
        r = _score(cur)
        if r > best_r:
            best_r = r
            best = cur.copy()
        
        # Decay learning rate
        lr *= decay
        if lr < 1e-6:
            break
    
    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score(best)

    # Phase 1: Soft-min optimization (main workhorse)
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 40.0:
        t_alloc = min(t_left - 20.0, 80.0)
        cand, r = _soft_min_optimize(best, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # Phase 2: Local search polish
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 10.0:
        t_alloc = min(t_left - 5.0, 25.0)
        cand, r = _local_search(best, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # Phase 3: Multi-start with soft-min if time remains
    t_left = DEADLINE - (time.time() - t0)
    restart_count = 0
    while t_left > 25.0 and restart_count < 5:
        # Structured perturbation: move a random subset of points
        n_move = rng.integers(10, 50)
        move_idx = rng.choice(N, size=n_move, replace=False)
        jitter_scale = 0.005 + 0.005 * rng.random()
        jitter = jitter_scale * rng.standard_normal((N, 2))
        jitter[~np.isin(np.arange(N), move_idx)] = 0.0
        cand = best + jitter
        
        # Project back into unit circle if needed
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)

        t_left = DEADLINE - (time.time() - t0)
        if t_left < 15.0:
            break
        t_alloc = min(t_left - 8.0, 25.0)
        
        cand, r = _soft_min_optimize(cand, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand
        
        # Quick local search polish
        t_left = DEADLINE - (time.time() - t0)
        if t_left > 8.0:
            t_alloc = min(t_left - 3.0, 12.0)
            cand, r = _local_search(cand, t_alloc, rng)
            if r > best_r:
                best_r = r
                best = cand
        
        restart_count += 1

    # Phase 4: Final high-precision polish
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 3.0:
        t_alloc = min(t_left - 1.0, 10.0)
        cand, r = _final_polish(best, t_alloc)
        if r > best_r:
            best_r = r
            best = cand

    # Final safety
    final_r = _score(best)
    if final_r <= 0.0:
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
