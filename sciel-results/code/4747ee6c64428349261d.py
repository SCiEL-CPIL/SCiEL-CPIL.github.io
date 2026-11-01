import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Vectorized validity check (no sqrt)
    def is_valid(params, tol=1e-9):
        xy = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist2 = dx*dx + dy*dy
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(dist2[iu] < (min_dist[iu] - tol)**2):
            return False
        return True

    def get_score(params):
        return np.sum(params[2*N:])

    def logsumexp(x, beta=50):
        """Stable log-sum-exp for smooth max approximation"""
        x_max = np.max(x)
        if x_max > 500/beta:
            return x_max
        return x_max + np.log(np.sum(np.exp(beta*(x - x_max)))) / beta

    def smooth_max(x, beta=50):
        """Smooth maximum via log-sum-exp"""
        return logsumexp(x, beta)

    def clearance_objective(params):
        xy = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        # Boundary clearances
        b = np.concatenate([
            xy[:, 0] - r,
            1 - xy[:, 0] - r,
            xy[:, 1] - r,
            1 - xy[:, 1] - r
        ])
        # Pairwise clearances
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        p = dist[iu] - min_dist[iu]
        all_c = np.concatenate([b, p])
        # Smooth min: -smooth_max(-all_c)
        smooth_min = -smooth_max(-all_c, beta=80)
        # Penalty for any negative clearance (smooth)
        neg_penalty = smooth_max(-all_c, beta=40)
        return -np.sum(r) + 50.0 * neg_penalty**2 + 5.0 * neg_penalty

    def penalty_objective(params):
        xy = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        b = np.concatenate([
            xy[:, 0] - r,
            1 - xy[:, 0] - r,
            xy[:, 1] - r,
            1 - xy[:, 1] - r
        ])
        # Smooth violation penalty
        bv = smooth_max(-b, beta=60)**2
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        ov = smooth_max(min_dist[iu] - dist[iu], beta=60)**2
        return -np.sum(r) + 100.0*bv + 100.0*ov

    def epigraph_objective(params):
        """
        Epigraph formulation: params = [x_1..x_N, y_1..y_N, r_1..r_N, t]
        Minimize -sum(r) + lambda * t
        Subject to (smoothed): t >= max_violation
        We use a smooth max of all violations as a surrogate for t.
        """
        lam = 200.0
        xy = params[:2*N].reshape(N, 2)
        r = params[2*N:3*N]
        t = params[3*N]
        
        # Boundary violations (should be <= 0)
        b_viol = np.concatenate([
            r - xy[:, 0],
            r - (1 - xy[:, 0]),
            r - xy[:, 1],
            r - (1 - xy[:, 1])
        ])
        
        # Pairwise overlap violations (should be <= 0)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        p_viol = min_dist[iu] - dist[iu]
        
        all_viol = np.concatenate([b_viol, p_viol])
        max_viol = smooth_max(all_viol, beta=80)
        
        # Objective: maximize sum(r) - penalty for violations
        # Using t as an upper bound on max_viol
        obj = -np.sum(r) + lam * np.maximum(t, max_viol) + 0.1 * t
        return obj

    def coordinate_descent_polish(params, n_rounds=30, tol=1e-10):
        """
        Per-circle coordinate descent: for each circle, find optimal (x,y,r)
        given all others fixed. This escapes shallow local minima.
        """
        params = params.copy()
        for rnd in range(n_rounds):
            improved = False
            for i in range(N):
                # Save current state
                old_x = params[2*i]
                old_y = params[2*i+1]
                old_r = params[2*N+i]
                
                # Find max radius for circle i given others
                # Constraints: 
                #   r <= x, r <= 1-x, r <= y, r <= 1-y
                #   r <= dist(i,j) - r_j for all j != i
                max_r = min(old_x, 1-old_x, old_y, 1-old_y)
                for j in range(N):
                    if j == i:
                        continue
                    dx = old_x - params[2*j]
                    dy = old_y - params[2*j+1]
                    d = np.sqrt(dx*dx + dy*dy)
                    max_r = min(max_r, d - params[2*N+j])
                max_r = max(max_r, 0.001)
                
                # Try to increase radius slightly
                new_r = min(max_r, old_r * 1.02 + 0.001)
                
                # Check if we can place with new_r
                # Need to find (x,y) such that:
                #   new_r <= x <= 1-new_r
                #   new_r <= y <= 1-new_r
                #   dist((x,y), (x_j,y_j)) >= new_r + r_j for all j
                # Start from current position
                cx = np.clip(old_x, new_r, 1-new_r)
                cy = np.clip(old_y, new_r, 1-new_r)
                
                # Local search for valid position
                best_pos = (cx, cy)
                best_dist = 0
                for attempt in range(50):
                    # Check feasibility
                    feasible = True
                    for j in range(N):
                        if j == i:
                            continue
                        dx = cx - params[2*j]
                        dy = cy - params[2*j+1]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < new_r + params[2*N+j] - 1e-9:
                            feasible = False
                            break
                    if feasible:
                        # Score: sum of distances to neighbors (more space = better)
                        s = 0
                        for j in range(N):
                            if j == i:
                                continue
                            dx = cx - params[2*j]
                            dy = cy - params[2*j+1]
                            s += np.sqrt(dx*dx + dy*dy)
                        if s > best_dist:
                            best_dist = s
                            best_pos = (cx, cy)
                        break
                    else:
                        # Nudge away from violating neighbor
                        for j in range(N):
                            if j == i:
                                continue
                            dx = cx - params[2*j]
                            dy = cy - params[2*j+1]
                            d = np.sqrt(dx*dx + dy*dy)
                            if d < new_r + params[2*N+j]:
                                if d > 1e-12:
                                    cx += (dx/d) * 0.01
                                    cy += (dy/d) * 0.01
                                else:
                                    ang = np.random.uniform(0, 2*np.pi)
                                    cx += 0.01*np.cos(ang)
                                    cy += 0.01*np.sin(ang)
                                break
                        cx = np.clip(cx, new_r, 1-new_r)
                        cy = np.clip(cy, new_r, 1-new_r)
                
                if best_dist > 0:
                    params[2*i] = best_pos[0]
                    params[2*i+1] = best_pos[1]
                    params[2*N+i] = new_r
                    if abs(new_r - old_r) > tol or abs(best_pos[0]-old_x) > tol or abs(best_pos[1]-old_y) > tol:
                        improved = True
            if not improved:
                break
        return params

    def push_apart(params, n_iters=100):
        xy = params[:2*N].reshape(N, 2).copy()
        r = params[2*N:].copy()
        for _ in range(n_iters):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    md = r[i] + r[j]
                    if dist < md and dist > 1e-12:
                        ov = md - dist
                        wi = r[i]/(r[i]+r[j]) if r[i]+r[j] > 1e-15 else 0.5
                        wj = 1 - wi
                        px = (dx/dist)*ov
                        py = (dy/dist)*ov
                        xy[i, 0] -= px*wi
                        xy[i, 1] -= py*wi
                        xy[j, 0] += px*wj
                        xy[j, 1] += py*wj
                        moved = True
                    elif dist <= 1e-12:
                        ang = np.random.uniform(0, 2*np.pi)
                        xy[i, 0] -= 0.01*np.cos(ang)
                        xy[i, 1] -= 0.01*np.sin(ang)
                        xy[j, 0] += 0.01*np.cos(ang)
                        xy[j, 1] += 0.01*np.sin(ang)
                        moved = True
            for i in range(N):
                if xy[i, 0] < r[i]: xy[i, 0] = r[i]; moved = True
                if xy[i, 0] > 1-r[i]: xy[i, 0] = 1-r[i]; moved = True
                if xy[i, 1] < r[i]: xy[i, 1] = r[i]; moved = True
                if xy[i, 1] > 1-r[i]: xy[i, 1] = 1-r[i]; moved = True
            if not moved:
                break
        return np.concatenate([xy.flatten(), r])

    def refine(params, obj, max_iter=400, tol=1e-14):
        try:
            res = minimize(obj, params, method='Powell',
                           options={'maxiter': max_iter, 'ftol': tol, 'maxfev': 3000})
            return res.x
        except Exception:
            return params

    def refine_lbfgs(params, max_iter=500, tol=1e-14):
        try:
            res = minimize(penalty_objective, params, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxls': 50, 'maxcor': 20})
            return res.x
        except Exception:
            return params

    def refine_epigraph(params, max_iter=800, tol=1e-14):
        """Refine using epigraph formulation with auxiliary variable t"""
        try:
            # Extend params with t
            t_init = 0.01
            full_params = np.concatenate([params, [t_init]])
            res = minimize(epigraph_objective, full_params, method='Powell',
                           options={'maxiter': max_iter, 'ftol': tol, 'maxfev': 5000})
            return res.x[:3*N]  # Drop t
        except Exception:
            return params

    def high_precision_polish(params, max_iter=2000, tol=1e-16):
        """Final high-precision polish with tighter tolerances"""
        try:
            # First: coordinate descent to escape local minima
            cd_result = coordinate_descent_polish(params, n_rounds=20, tol=1e-10)
            
            # Then: epigraph refinement
            ep_result = refine_epigraph(cd_result, max_iter=1000, tol=1e-15)
            
            # Then: Powell on clearance
            res = minimize(clearance_objective, ep_result, method='Powell',
                           options={'maxiter': max_iter, 'ftol': tol, 'maxfev': 5000})
            
            # Then: L-BFGS-B on penalty
            res2 = minimize(penalty_objective, res.x, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-14, 'maxls': 100, 'maxcor': 30})
            
            # Final coordinate descent
            final = coordinate_descent_polish(res2.x, n_rounds=30, tol=1e-12)
            return final
        except Exception:
            return params

    def initial_configs(rng):
        n = N
        configs_list = []

        # Grid 6x6 (approx)
        side = 6
        cfg = []
        for i in range(n):
            row, col = i//side, i%side
            cfg.append([(col+0.5)/side, (row+0.5)/side, 0.5/side-0.001])
        cfg = np.array(cfg)
        configs_list.append(np.concatenate([cfg[:, :2].flatten(), cfg[:, 2]]))

        # Hexagonal packing (better density)
        xy = np.zeros((n, 2)); r = np.zeros(n)
        row_spacing = 0.16667
        col_spacing = 0.16667
        for i in range(n):
            row, col = i//6, i%6
            x = (col+0.5)*col_spacing + (0.08333 if row % 2 == 1 else 0)
            y = (row+0.5)*row_spacing
            x = np.clip(x, 0.08, 0.92)
            y = np.clip(y, 0.08, 0.92)
            xy[i] = [x, y]; r[i] = 0.075
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # Square grid with larger circles
        xy = np.zeros((n, 2)); r = np.zeros(n)
        for i in range(n):
            row, col = i//6, i%6
            xy[i] = [(col+0.5)/6, (row+0.5)/6]
            r[i] = 1/12 - 0.002
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # Random with medium radii
        xy = rng.uniform(0.1, 0.9, size=(n, 2))
        r = rng.uniform(0.05, 0.08, size=n)
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # Random with smaller radii for more flexibility
        xy = rng.uniform(0.08, 0.92, size=(n, 2))
        r = rng.uniform(0.04, 0.06, size=n)
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # 8x4 grid
        xy = np.zeros((n, 2)); r = np.zeros(n)
        for i in range(n):
            row, col = i//8, i%8
            xy[i] = [(col+0.5)/8, (row+0.5)/4]
            r[i] = 0.5/8 - 0.001
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # 4x8 grid
        xy = np.zeros((n, 2)); r = np.zeros(n)
        for i in range(n):
            row, col = i//4, i%4
            xy[i] = [(col+0.5)/4, (row+0.5)/8]
            r[i] = 0.5/4 - 0.001
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # Perturbed hexagonal
        xy = np.zeros((n, 2)); r = np.zeros(n)
        for i in range(n):
            row, col = i//6, i%6
            x = (col+0.5)/6 + (0.083 if row % 2 == 1 else 0)
            y = (row+0.5)/6
            xy[i] = [x + rng.uniform(-0.01, 0.01), y + rng.uniform(-0.01, 0.01)]
            r[i] = 0.075 + rng.uniform(-0.005, 0.005)
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # Random cluster (some large, some small)
        xy = rng.uniform(0.1, 0.9, size=(n, 2))
        r = np.zeros(n)
        for i in range(n):
            if i < 8:
                r[i] = rng.uniform(0.08, 0.1)
            elif i < 16:
                r[i] = rng.uniform(0.05, 0.07)
            else:
                r[i] = rng.uniform(0.03, 0.05)
        configs_list.append(np.concatenate([xy.flatten(), r]))

        return configs_list

    rng = np.random.RandomState(42)
    best_result = None
    best_score = -np.inf

    def update_best(params):
        nonlocal best_result, best_score
        if is_valid(params):
            s = get_score(params)
            if s > best_score:
                best_score = s
                best_result = params.copy()

    # Phase 1: diverse restarts
    base_configs = initial_configs(rng)
    n_restarts = 45
    for restart in range(n_restarts):
        if elapsed() > deadline - 35:
            break
        # Cycle through base configs with perturbation
        base = base_configs[restart % len(base_configs)]
        scale = 0.008 + 0.015 * (restart // len(base_configs))
        params = base.copy()
        xy_p = params[:2*N].reshape(N, 2).copy()
        r_p = params[2*N:].copy()
        xy_p += rng.uniform(-scale, scale, size=(N, 2))
        r_p += rng.uniform(-scale*0.3, scale*0.3, size=N)
        r_p = np.clip(r_p, 0.005, 0.5)
        xy_p = np.clip(xy_p, r_p[:, None], (1-r_p)[:, None])
        params = np.concatenate([xy_p.flatten(), r_p])
        params = push_apart(params, n_iters=50)
        update_best(params)

        # Powell on clearance objective
        polished = refine(params, clearance_objective, max_iter=350, tol=1e-13)
        update_best(polished)

        # L-BFGS-B on penalty
        polished2 = refine_lbfgs(polished, max_iter=350, tol=1e-13)
        update_best(polished2)

        # Epigraph refinement (new)
        polished3 = refine_epigraph(polished2, max_iter=400, tol=1e-13)
        update_best(polished3)

        # Coordinate descent polish (new)
        if elapsed() < deadline - 40:
            cd_polished = coordinate_descent_polish(polished3, n_rounds=15, tol=1e-10)
            update_best(cd_polished)

    # Phase 2: multi-scale polish best
    if best_result is not None:
        polish_scales = [0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
        for scale in polish_scales:
            if elapsed() > deadline - 10:
                break
            for trial in range(4):
                if elapsed() > deadline - 8:
                    break
                pert = best_result.copy()
                xy_p = pert[:2*N].reshape(N, 2).copy()
                r_p = pert[2*N:].copy()
                xy_p += rng.uniform(-scale, scale, size=(N, 2))
                r_p += rng.uniform(-scale*0.5, scale*0.5, size=N)
                r_p = np.clip(r_p, 0.001, 0.5)
                xy_p = np.clip(xy_p, r_p[:, None], (1-r_p)[:, None])
                pert = np.concatenate([xy_p.flatten(), r_p])
                pert = push_apart(pert, n_iters=60)
                polished = refine(pert, clearance_objective, max_iter=600, tol=1e-15)
                update_best(polished)
                polished2 = refine_lbfgs(polished, max_iter=600, tol=1e-15)
                update_best(polished2)
                # Add epigraph + CD polish in phase 2 as well
                if elapsed() < deadline - 12:
                    polished3 = refine_epigraph(polished2, max_iter=500, tol=1e-15)
                    update_best(polished3)
                    cd_polished = coordinate_descent_polish(polished3, n_rounds=20, tol=1e-11)
                    update_best(cd_polished)

    # Phase 3: high-precision final polish
    if best_result is not None and elapsed() < deadline - 15:
        polished_hp = high_precision_polish(best_result, max_iter=3000, tol=1e-16)
        update_best(polished_hp)
        
        # Extra: try epigraph + CD on the best result one more time
        if best_result is not None and elapsed() < deadline - 8:
            ep_final = refine_epigraph(best_result, max_iter=1500, tol=1e-15)
            update_best(ep_final)
            cd_final = coordinate_descent_polish(ep_final, n_rounds=40, tol=1e-12)
            update_best(cd_final)

    if best_result is not None:
        xy = best_result[:2*N].reshape(N, 2).copy()
        r = best_result[2*N:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final = np.concatenate([xy.flatten(), r])
        if is_valid(final):
            return np.column_stack([xy, r])

    # Fallback grid
    side = 6
    cells = []
    for i in range(N):
        row, col = i//side, i%side
        cells.append([(col+0.5)/side, (row+0.5)/side, 0.5/side - 1e-3])
    return np.array(cells)
