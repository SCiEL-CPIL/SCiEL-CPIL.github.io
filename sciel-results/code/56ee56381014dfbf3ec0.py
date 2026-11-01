# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    """
    Place 50 equal circles in a unit circle to maximize the common radius.
    Uses Nelder-Mead on an augmented vector [centers, radius] with a penalty objective.
    """
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin

    def get_radius_from_aug(x_aug):
        """Extract radius and check validity from augmented vector."""
        pts = x_aug[:2*N].reshape(N, 2)
        R = x_aug[2*N]
        # Check boundaries
        d_origin = np.linalg.norm(pts, axis=1)
        if np.any(d_origin > 1.0):
            return 0.0
        # Check overlaps
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        min_pair_d = np.sqrt(np.min(d2)) / 2.0
        min_boundary = 1.0 - np.max(d_origin)
        actual_r = min(min_pair_d, min_boundary)
        # The R in the vector might be larger than actual_r if constraints were violated during opt
        # But we want the actual valid radius.
        # However, the optimizer minimizes penalty - R. If penalty is 0, it maximizes R.
        # So if penalty is 0, actual_r should be close to R.
        # We return the true valid radius.
        return actual_r

    def objective_aug(x_aug, penalty_weight=100.0):
        """
        Objective: Minimize Penalty - R.
        Penalty is sum of squared violations of non-overlap and boundary constraints.
        """
        pts = x_aug[:2*N].reshape(N, 2)
        R = x_aug[2*N]
        
        if R <= 0:
            return 1e6
            
        # Boundary violations
        d_origin = np.linalg.norm(pts, axis=1)
        boundary_viol = np.maximum(0, R - d_origin)
        penalty_boundary = np.sum(boundary_viol ** 2)
        
        # Pairwise violations
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        
        # We only care about upper triangle for pairs to avoid double counting? 
        # Summing all i!=j is fine, just doubles the penalty.
        # Let's use upper triangle for efficiency and correctness of magnitude.
        i, j = np.triu_indices(N, k=1)
        pair_d = d[i, j]
        pair_viol = np.maximum(0, 2*R - pair_d)
        penalty_pairs = np.sum(pair_viol ** 2)
        
        total_penalty = penalty_boundary + penalty_pairs
        
        return penalty_weight * total_penalty - R

    def init_aug(x_flat, R_init=0.1):
        """Create augmented vector from flat centers."""
        return np.concatenate([x_flat, [R_init]])

    def init_ring_pattern(seed):
        """Concentric ring initialization."""
        rng = np.random.default_rng(seed)
        pts = []
        rings = [(1, 0.0), (6, 0.25), (12, 0.50), (15, 0.75), (16, 0.92)]
        for n_ring, r_ring in rings:
            angles = 2 * np.pi * np.arange(n_ring) / n_ring + rng.uniform(0, 2 * np.pi)
            for a in angles:
                rr = r_ring + rng.normal(0, 0.01)
                pts.append([rr * np.cos(a), rr * np.sin(a)])
        pts = np.array(pts[:N])
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.98
        if np.any(too_far):
            pts[too_far] *= (0.98 / dists[too_far])[:, None]
        return pts.flatten()

    def init_hex_grid(seed):
        """Hexagonal lattice initialization."""
        rng = np.random.default_rng(seed)
        pts = []
        s = 0.22
        dx, dy = s, s * np.sqrt(3) / 2
        for row in range(-4, 5):
            for col in range(-4, 5):
                x = col * dx + (row % 2) * dx / 2
                y = row * dy
                if x * x + y * y < 0.88:
                    pts.append([x, y])
        if len(pts) < N:
            while len(pts) < N:
                r = np.sqrt(rng.uniform(0, 0.88))
                a = rng.uniform(0, 2 * np.pi)
                pts.append([r * np.cos(a), r * np.sin(a)])
        pts = np.array(pts[:N])
        pts += rng.normal(0, 0.008, (N, 2))
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.97
        if np.any(too_far):
            pts[too_far] *= (0.97 / dists[too_far])[:, None]
        return pts.flatten()

    def init_random_polar(seed):
        """Random polar coordinates."""
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2 * np.pi, N)
        r = rng.beta(2.5, 1.5, N) * 0.96
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts += rng.normal(0, 0.01, (N, 2))
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.97
        if np.any(too_far):
            pts[too_far] *= (0.97 / dists[too_far])[:, None]
        return pts.flatten()

    def init_random_uniform(seed):
        """Uniform random in disk."""
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2 * np.pi, N)
        r = np.sqrt(rng.uniform(0, 1, N)) * 0.96
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts += rng.normal(0, 0.008, (N, 2))
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.97
        if np.any(too_far):
            pts[too_far] *= (0.97 / dists[too_far])[:, None]
        return pts.flatten()

    def local_refine_aug(x_aug, n_iter=200):
        """
        Simple local refinement on augmented vector.
        If penalty > 0, push points apart.
        If penalty == 0, try to increase R.
        """
        x = x_aug.copy()
        pts = x[:2*N].reshape(N, 2).copy()
        R = x[2*N]
        
        for it in range(n_iter):
            if time.time() - start_time > time_budget - 5:
                break
                
            # Calculate current radius and violations
            d_origin = np.linalg.norm(pts, axis=1)
            diff = pts[:, None, :] - pts[None, :, :]
            d2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(d2, np.inf)
            d = np.sqrt(d2)
            
            # Check if valid
            min_pair_d = np.min(d) / 2.0
            min_boundary = 1.0 - np.max(d_origin)
            current_r = min(min_pair_d, min_boundary)
            
            if current_r > R:
                # We can increase R
                R = current_r
                x[2*N] = R
                # Try to increase R further? No, just update.
                continue
            
            # If R is limited by overlaps, push points apart
            # Find pairs with d < 2R
            i, j = np.triu_indices(N, k=1)
            pair_d = d[i, j]
            violations = pair_d < 2*R
            
            if np.any(violations):
                # Get indices
                viol_i = i[violations]
                viol_j = j[violations]
                
                # Push apart
                for k in range(len(viol_i)):
                    idx1 = viol_i[k]
                    idx2 = viol_j[k]
                    vec = pts[idx1] - pts[idx2]
                    norm = np.linalg.norm(vec)
                    if norm > 1e-10:
                        push = vec / norm * 0.001
                        pts[idx1] += push
                        pts[idx2] -= push
            
            # Check boundary
            too_far = d_origin > 1.0
            if np.any(too_far):
                pts[too_far] *= (1.0 / d_origin[too_far])[:, None]
            
            x[:2*N] = pts.flatten()
        
        return x

    # Generate diverse initializations
    inits_flat = [
        init_ring_pattern(1),
        init_ring_pattern(2),
        init_ring_pattern(3),
        init_hex_grid(10),
        init_hex_grid(20),
        init_random_polar(30),
        init_random_polar(40),
        init_random_polar(50),
        init_random_uniform(60),
        init_random_uniform(70),
    ]
    
    # Augment them
    inits_aug = [init_aug(x, R_init=0.12) for x in inits_flat]

    best_x_aug = None
    best_r = -1.0

    # Phase 1: Optimization
    for i, x0 in enumerate(inits_aug):
        if time.time() - start_time > time_budget - 20:
            break
        try:
            res = minimize(lambda x: objective_aug(x, 100.0), x0,
                          method='Nelder-Mead',
                          options={'maxiter': 1000, 'maxfev': 2000,
                                   'xatol': 1e-6, 'fatol': 1e-7, 'adaptive': True})
            r = get_radius_from_aug(res.x)
            if r > best_r:
                best_r = r
                best_x_aug = res.x.copy()
        except Exception:
            pass

    if best_x_aug is None:
        # Fallback
        return init_ring_pattern(1).reshape(N, 2)

    # Phase 2: Local Refinement
    if time.time() - start_time < time_budget - 10:
        refined_aug = local_refine_aug(best_x_aug, n_iter=300)
        r_ref = get_radius_from_aug(refined_aug)
        if r_ref > best_r:
            best_r = r_ref
            best_x_aug = refined_aug

    # Phase 3: Fine-tune with Nelder-Mead again on the refined solution
    if time.time() - start_time < time_budget - 5:
        try:
            res2 = minimize(lambda x: objective_aug(x, 200.0), best_x_aug,
                           method='Nelder-Mead',
                           options={'maxiter': 1000, 'maxfev': 2000,
                                    'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True})
            r2 = get_radius_from_aug(res2.x)
            if r2 > best_r:
                best_r = r2
                best_x_aug = res2.x.copy()
        except Exception:
            pass

    # Final Local Refinement
    if time.time() - start_time < time_budget - 2:
        final_aug = local_refine_aug(best_x_aug, n_iter=200)
        r_final = get_radius_from_aug(final_aug)
        if r_final > best_r:
            best_r = r_final
            best_x_aug = final_aug

    # Extract centers
    return best_x_aug[:2*N].reshape(N, 2)
# EVOLVE-BLOCK-END
