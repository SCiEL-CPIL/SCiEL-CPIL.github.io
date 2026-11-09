import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def calc_score(x):
        """Compute the packing radius for given points."""
        if len(x) != N:
            return 0.0
        if not np.all(np.isfinite(x)):
            return 0.0
        norms = np.linalg.norm(x, axis=1)
        if np.any(norms > 1.0):
            return 0.0
        b_min = np.min(1.0 - norms)
        if b_min < 0:
            return 0.0
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        return min(min_pair, b_min)

    def hexagonal_init(spacing):
        """Generate points on a hexagonal lattice clipped to unit circle, pick N closest to origin."""
        points = []
        k = int(1.0 / spacing) + 3
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        return points[indices]

    def ring_init(n_rings, r_start, r_end):
        """Initialize with concentric rings."""
        pts = [[0.0, 0.0]]
        for ring in range(1, n_rings + 1):
            r = r_start + (r_end - r_start) * (ring - 1) / max(n_rings - 1, 1)
            if ring == 1:
                n_pts = 6
            else:
                n_pts = ring * 6
            n_pts = min(n_pts, N - len(pts))
            if n_pts <= 0:
                break
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
        pts = np.array(pts[:N])
        if len(pts) < N:
            return None
        return pts

    # Generate diverse initializations
    candidates = []

    # Hexagonal grids with various spacings
    for s in np.linspace(0.09, 0.26, 25):
        pts = hexagonal_init(s)
        if pts is not None:
            candidates.append(pts)

    # Ring-based inits
    for n_rings in [5, 6, 7, 8]:
        for r_start in [0.05, 0.1, 0.15]:
            pts = ring_init(n_rings, r_start, 0.9)
            if pts is not None:
                candidates.append(pts)

    # Random inits
    for _ in range(15):
        pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, None] * 0.95
        candidates.append(pts)

    # Evaluate all candidates and pick the best few
    scored_candidates = []
    for c in candidates:
        r = calc_score(c)
        scored_candidates.append((r, c))
    scored_candidates.sort(key=lambda x: -x[0])

    # Take top candidates for optimization
    top_candidates = [c for r, c in scored_candidates[:8]]
    if not top_candidates:
        top_candidates = [np.random.rand(N, 2) * 0.8]

    best_x = top_candidates[0].copy()
    best_score = calc_score(best_x)

    # Optimization using SLSQP with explicit constraints
    def slsqp_obj_and_constraints(x):
        """
        Objective: maximize r (min of pairwise distances/2 and boundary distances)
        We minimize -r using a smooth approximation, with constraints ensuring
        all pairwise distances >= 2r and all boundary distances >= r.
        
        Actually, let's use a different approach: we optimize for a fixed target radius r,
        checking feasibility. Or better, use the soft-min approach with SLSQP.
        """
        pass

    def make_slsqp_func(target_r):
        """
        Create SLSQP objective and constraints for a target radius.
        Objective: minimize sum of violations (should be 0 if feasible)
        Constraints: pairwise distances >= 2*target_r, boundary distances >= target_r
        """
        iu = np.triu_indices(N, k=1)
        n_pairs = len(iu[0])

        def obj(flat_x):
            x = flat_x.reshape(N, 2)
            # Objective: minimize negative of the minimum constraint slack
            # This encourages all constraints to be satisfied
            norms = np.linalg.norm(x, axis=1)
            boundary_slacks = norms - (1.0 - target_r)  # should be <= 0
            diff = x[iu[0]] - x[iu[1]]
            pair_dists = np.linalg.norm(diff, axis=1)
            pair_slacks = pair_dists - 2.0 * target_r  # should be >= 0
            
            # Minimize the sum of squared violations
            violations = np.maximum(0, boundary_slacks) + np.maximum(0, -pair_slacks)
            return np.sum(violations ** 2)

        def grad_obj(flat_x):
            x = flat_x.reshape(N, 2)
            grad = np.zeros_like(x)
            norms = np.linalg.norm(x, axis=1)
            boundary_slacks = norms - (1.0 - target_r)
            diff = x[iu[0]] - x[iu[1]]
            pair_dists = np.linalg.norm(diff, axis=1)
            pair_slacks = pair_dists - 2.0 * target_r

            # Gradient of boundary violations
            bnd_viol = np.maximum(0, boundary_slacks)
            norms_safe = np.where(norms > 1e-8, norms, 1e-8)
            norm_x = x / norms_safe[:, np.newaxis]
            grad += 2.0 * bnd_viol[:, np.newaxis] * norm_x

            # Gradient of pair violations
            pair_viol = np.maximum(0, -pair_slacks)
            pair_dists_safe = np.where(pair_dists > 1e-8, pair_dists, 1e-8)
            vecs = diff / pair_dists_safe[:, np.newaxis]
            
            for k in range(n_pairs):
                i_idx = iu[0][k]
                j_idx = iu[1][k]
                grad[i_idx] -= 2.0 * pair_viol[k] * vecs[k]
                grad[j_idx] += 2.0 * pair_viol[k] * vecs[k]

            return grad.flatten()

        def constraint_funcs(flat_x):
            x = flat_x.reshape(N, 2)
            # Constraint: pairwise distances >= 2*target_r
            diff = x[iu[0]] - x[iu[1]]
            pair_dists = np.linalg.norm(diff, axis=1)
            pair_constraints = pair_dists - 2.0 * target_r
            # Constraint: boundary distances >= target_r
            norms = np.linalg.norm(x, axis=1)
            boundary_constraints = (1.0 - target_r) - norms
            return np.concatenate([pair_constraints, boundary_constraints])

        def jac_constraint(flat_x):
            x = flat_x.reshape(N, 2)
            jac = np.zeros((n_pairs + N, 2 * N))
            diff = x[iu[0]] - x[iu[1]]
            pair_dists = np.linalg.norm(diff, axis=1)
            pair_dists_safe = np.where(pair_dists > 1e-8, pair_dists, 1e-8)
            vecs = diff / pair_dists_safe[:, np.newaxis]

            for k in range(n_pairs):
                i_idx = iu[0][k]
                j_idx = iu[1][k]
                jac[k, 2*i_idx:2*i_idx+2] = vecs[k]
                jac[k, 2*j_idx:2*j_idx+2] = -vecs[k]

            norms = np.linalg.norm(x, axis=1)
            norms_safe = np.where(norms > 1e-8, norms, 1e-8)
            norm_x = x / norms_safe[:, np.newaxis]
            for i in range(N):
                jac[n_pairs + i, 2*i:2*i+2] = -norm_x[i]

            return jac

        return obj, grad_obj, constraint_funcs, jac_constraint

    # Try SLSQP optimization for each top candidate
    for cand_idx, current_x in enumerate(top_candidates):
        if time_left() < 10:
            break

        current_x = current_x.copy()
        local_best_x = current_x.copy()
        local_best_score = calc_score(current_x)

        # Try increasing target radii
        target_r = local_best_score
        for _ in range(30):
            if time_left() < 10:
                break

            if target_r <= 0:
                break

            obj, grad_obj, constraint_funcs, jac_constraint = make_slsqp_func(target_r)

            try:
                res = minimize(
                    obj,
                    current_x.flatten(),
                    jac=grad_obj,
                    method='SLSQP',
                    constraints=[{'type': 'ineq', 'fun': constraint_funcs, 'jac': jac_constraint}],
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
                )

                if res.x is not None:
                    new_x = res.x.reshape(N, 2)
                    new_score = calc_score(new_x)
                    if new_score > local_best_score:
                        local_best_score = new_score
                        local_best_x = new_x
                        current_x = new_x
                        target_r = min(new_score * 1.02, 0.15)
                    else:
                        # If we couldn't improve, try a slightly smaller target
                        target_r *= 0.98
            except Exception:
                target_r *= 0.95

        if local_best_score > best_score:
            best_score = local_best_score
            best_x = local_best_x

    # Final polish with L-BFGS using soft-min
    def make_obj_func(beta_val):
        """Create objective function with given beta for soft-min."""
        iu = np.triu_indices(N, k=1)
        n_pairs = len(iu[0])

        def obj_func(flat_x):
            x = flat_x.reshape(N, 2)
            diff = x[iu[0]] - x[iu[1]]
            d_pairs = np.linalg.norm(diff, axis=1)
            dist_boundary = 1.0 - np.linalg.norm(x, axis=1)

            all_d = np.concatenate([d_pairs, dist_boundary])

            exponents = -beta_val * all_d
            max_exp = np.max(exponents)

            if not np.isfinite(max_exp):
                return 100.0, np.zeros_like(flat_x)

            sum_exp = np.sum(np.exp(exponents - max_exp))
            log_sum = max_exp + np.log(sum_exp)
            obj = log_sum / beta_val

            weights = np.exp(exponents - max_exp) / sum_exp
            grad = np.zeros_like(x)

            w_pairs = weights[:n_pairs]
            w_bnds = weights[n_pairs:]

            i_idx, j_idx = iu
            d_safe = np.where(d_pairs > 1e-8, d_pairs, 1e-8)
            vecs = x[i_idx] - x[j_idx]
            norm_vecs = vecs / d_safe[:, np.newaxis]

            np.add.at(grad, i_idx, -w_pairs[:, np.newaxis] * norm_vecs)
            np.add.at(grad, j_idx, w_pairs[:, np.newaxis] * norm_vecs)

            norms = np.linalg.norm(x, axis=1)
            norms_safe = np.where(norms > 1e-8, norms, 1e-8)
            norm_x = x / norms_safe[:, np.newaxis]

            np.add.at(grad, np.arange(N), w_bnds[:, np.newaxis] * norm_x)

            return obj, grad.flatten()
        return obj_func

    # Polish with L-BFGS at high beta
    if time_left() > 15:
        for beta in [100, 200, 300]:
            if time_left() < 10:
                break
            try:
                obj_func = make_obj_func(beta)
                res = minimize(obj_func, best_x.flatten(), method='L-BFGS-B', jac=True,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.x is not None:
                    final_x = res.x.reshape(N, 2)
                    final_score = calc_score(final_x)
                    if final_score > best_score:
                        best_x = final_x
                        best_score = final_score
            except Exception:
                pass

    # Nelder-Mead polish
    if time_left() > 10:
        def nm_obj(flat_x):
            x = flat_x.reshape(N, 2)
            d = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            min_pair = np.min(d) / 2.0
            min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
            return -min(min_pair, min_bound)

        try:
            res_nm = minimize(nm_obj, best_x.flatten(), method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12})
            if res_nm.x is not None:
                final_x = res_nm.x.reshape(N, 2)
                final_score = calc_score(final_x)
                if final_score > best_score:
                    best_x = final_x
                    best_score = final_score
        except Exception:
            pass

    return best_x
