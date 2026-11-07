"""Multi-objective optimizer: Bayesian optimization with GP surrogates over weight vectors on the simplex."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        used = 0
        all_F = []  # list of objective vectors
        all_X = []  # list of x vectors

        # --- Helper: dominance ---
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # --- Helper: 2D hypervolume ---
        def hv_2d(pts, ref_y, ref_z):
            if len(pts) == 0:
                return 0.0
            pts = np.array(pts)
            mask = (pts[:, 0] < ref_y) & (pts[:, 1] < ref_z)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n_pts = len(pts)
            hv2 = 0.0
            for i in range(n_pts):
                y_next = pts[i + 1, 0] if i + 1 < n_pts else ref_y
                if i + 1 < n_pts:
                    min_z_remaining = np.min(pts[i + 1:, 1])
                else:
                    min_z_remaining = ref_z
                if min_z_remaining > pts[i, 1]:
                    w_y = y_next - pts[i, 0]
                    w_z = min_z_remaining - pts[i, 1]
                    if w_y > 0 and w_z > 0:
                        hv2 += w_y * w_z
            return hv2

        # --- Helper: 3D hypervolume ---
        def hv_3d(points, ref):
            if len(points) == 0:
                return 0.0
            P = np.array(points)
            mask = np.all(P < ref, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0

            order = np.argsort(P[:, 0])
            P_sorted = P[order]
            n_pts = len(P_sorted)
            hv = 0.0
            for i in range(n_pts):
                f0_min = P_sorted[i, 0]
                f0_max = P_sorted[i + 1, 0] if i + 1 < n_pts else ref[0]
                if f0_max <= f0_min:
                    continue
                mask_slice = (P_sorted[:, 0] >= f0_min) & (P_sorted[:, 0] < f0_max)
                slice_pts = P_sorted[mask_slice, 1:3]
                hv_slice = hv_2d(slice_pts, ref[1], ref[2])
                hv += hv_slice * (f0_max - f0_min)
            return hv

        # --- Helper: marginal HV ---
        def marginal_hv(point, archive_F):
            if not np.all(point < ref_point):
                return 0.0
            if len(archive_F) == 0:
                return hv_3d([point], ref_point)
            for f in archive_F:
                if dominates(f, point):
                    return 0.0
            hv_before = hv_3d(archive_F, ref_point)
            archive_new = list(archive_F) + [point]
            hv_after = hv_3d(archive_new, ref_point)
            return hv_after - hv_before

        # --- Helper: GP kernel (RBF) ---
        def rbf_kernel(X1, X2, length_scale=0.3, variance=1.0):
            X1 = np.atleast_2d(X1)
            X2 = np.atleast_2d(X2)
            sq_dists = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
            sq_dists = np.maximum(sq_dists, 0)
            return variance * np.exp(-0.5 * sq_dists / length_scale**2)

        # --- Helper: GP posterior ---
        def gp_posterior(X_train, y_train, X_pred, length_scale=0.3, variance=1.0, noise=1e-6):
            K = rbf_kernel(X_train, X_train, length_scale, variance) + noise * np.eye(len(X_train))
            K_pred = rbf_kernel(X_train, X_pred, length_scale, variance)
            try:
                K_inv = np.linalg.inv(K)
            except:
                K_inv = np.linalg.pinv(K)
            y_mean = np.mean(y_train)
            y_centered = y_train - y_mean
            mu = y_mean + K_pred.T @ K_inv @ y_centered
            v = K_pred.T @ K_inv @ K_pred
            var = variance - np.sum(v, axis=1)
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)

        # --- Helper: Expected Improvement for minimization ---
        def expected_improvement(mu, sigma, best_y):
            sigma = np.maximum(sigma, 1e-10)
            z = (best_y - mu) / sigma
            ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0)

        # --- Phase 1: Initial diverse sampling ---
        # Use a mix of grid and random points for initial coverage
        n_initial = min(40, budget)
        
        # Grid points
        grid_size = 3
        grid_candidates = []
        for a in range(grid_size):
            for b in range(grid_size):
                for c in range(grid_size):
                    for d in range(grid_size):
                        x = np.array([a, b, c, d], dtype=float) / (grid_size - 1)
                        grid_candidates.append(x)
        
        # Random points
        n_random = n_initial - len(grid_candidates)
        if n_random > 0:
            random_candidates = np.random.rand(n_random, dim)
            for x in random_candidates:
                grid_candidates.append(x)
        
        np.random.shuffle(grid_candidates)
        
        for x in grid_candidates:
            if used >= n_initial:
                break
            if time.time() > deadline:
                break
            
            try:
                f = problem(x)
            except Exception:
                continue
            
            f = np.array(f)
            all_F.append(f)
            all_X.append(x)
            used += 1

        # --- Phase 2: Bayesian optimization with weight vectors ---
        # Generate 30 weight vectors on the 2D simplex (for 3 objectives)
        n_weights = 30
        weight_vectors = []
        for i in range(n_weights):
            w1 = i / (n_weights - 1)
            w2 = (n_weights - 1 - i) / (n_weights - 1) * 0.5
            w3 = 1.0 - w1 - w2
            if w3 >= 0:
                weight_vectors.append(np.array([w1, w2, w3]))
        
        # If we don't have exactly 30, adjust
        while len(weight_vectors) < n_weights:
            w = np.random.dirichlet(np.ones(n_obj))
            weight_vectors.append(w)
        
        # For each weight vector, use GP to predict weighted-sum minimizer
        # and select next point via Expected Improvement
        
        # Precompute weighted sums for existing points
        def weighted_sum(f, w):
            return np.dot(f, w)
        
        # GP training data per weight
        gp_data = {}
        for w in weight_vectors:
            if len(all_F) > 0:
                X_train = np.array(all_X)
                y_train = np.array([weighted_sum(f, w) for f in all_F])
                gp_data[w] = (X_train, y_train)
        
        # Iterative BO loop
        max_bo_iters = budget - used
        for it in range(max_bo_iters):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            best_ei = -np.inf
            best_x = None
            
            # Try each weight vector
            for w in weight_vectors:
                if w not in gp_data:
                    continue
                
                X_train, y_train = gp_data[w]
                if len(X_train) < 3:
                    continue
                
                # Find candidate points: existing points + new random candidates
                n_candidates = 20
                candidate_X = [np.array(x) for x in X_train]
                
                # Add random candidates
                for _ in range(n_candidates):
                    x_rand = np.random.uniform(lo, hi)
                    candidate_X.append(x_rand)
                
                # Also add points near current Pareto front
                if len(all_F) > 0:
                    # Find non-dominated points
                    pareto_idx = []
                    for i in range(len(all_F)):
                        is_dominated = False
                        for j in range(len(all_F)):
                            if i != j and dominates(all_F[j], all_F[i]):
                                is_dominated = True
                                break
                        if not is_dominated:
                            pareto_idx.append(i)
                    
                    # Add perturbations around Pareto points
                    for idx in pareto_idx[:5]:
                        for _ in range(3):
                            x_pert = all_X[idx] + np.random.normal(0, 0.05, dim)
                            x_pert = np.clip(x_pert, lo, hi)
                            candidate_X.append(x_pert)
                
                candidate_X = np.array(candidate_X)
                
                # Compute GP posterior
                try:
                    mu, sigma = gp_posterior(X_train, y_train, candidate_X, length_scale=0.3, variance=1.0)
                except:
                    continue
                
                best_y = np.min(y_train)
                ei = expected_improvement(mu, sigma, best_y)
                
                # Select point with max EI
                idx_max = np.argmax(ei)
                if ei[idx_max] > best_ei:
                    best_ei = ei[idx_max]
                    best_x = candidate_X[idx_max]
            
            if best_x is None:
                # Fallback: random point
                best_x = np.random.uniform(lo, hi)
            
            # Evaluate
            try:
                f = problem(best_x)
            except Exception:
                continue
            
            f = np.array(f)
            
            # Update all_F and all_X
            all_F.append(f)
            all_X.append(best_x)
            used += 1
            
            # Update GP data
            for w in weight_vectors:
                if w in gp_data:
                    X_train, y_train = gp_data[w]
                    y_new = weighted_sum(f, w)
                    gp_data[w] = (np.vstack([X_train, best_x]), np.append(y_train, y_new))
                else:
                    y_new = weighted_sum(f, w)
                    gp_data[w] = (best_x.reshape(1, -1), np.array([y_new]))

        # --- Phase 3: Final local refinement ---
        # Find non-dominated points
        archive = []
        for i in range(len(all_F)):
            dominated = False
            for j in range(len(all_F)):
                if i != j and dominates(all_F[j], all_F[i]):
                    dominated = True
                    break
            if not dominated:
                archive.append(i)

        if len(archive) > 0 and used < budget:
            n_refine = min(3, len(archive))
            refine_indices = archive[:n_refine]
            
            for idx in refine_indices:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                center = all_X[idx]
                # Fine-grained local search
                step_sizes = [0.02, 0.01, 0.005]
                for step in step_sizes:
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            for dz in [-step, 0, step]:
                                for dw in [-step, 0, step]:
                                    if dx == 0 and dy == 0 and dz == 0 and dw == 0:
                                        continue
                                    if used >= budget:
                                        break
                                    if time.time() > deadline:
                                        break
                                    
                                    x_new = center + np.array([dx, dy, dz, dw])
                                    x_new = np.clip(x_new, lo, hi)
                                    
                                    # Check if already evaluated
                                    already_evaluated = False
                                    for existing_x in all_X:
                                        if np.allclose(x_new, existing_x, atol=1e-10):
                                            already_evaluated = True
                                            break
                                    if already_evaluated:
                                        continue
                                    
                                    try:
                                        f = problem(x_new)
                                    except Exception:
                                        continue
                                    
                                    f = np.array(f)
                                    all_F.append(f)
                                    all_X.append(x_new)
                                    used += 1

        return all_X
# EVOLVE-BLOCK-END
