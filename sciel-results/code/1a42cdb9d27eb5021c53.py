"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Discrete combinatorial MWIS on a fine grid + randomized greedy + local search + SLSQP polish.
1. Create a fine grid of candidate centers.
2. For each center, determine the maximum feasible radius (distance to nearest wall/other center).
3. Build a conflict graph where edges connect incompatible circles.
4. Solve MWIS via randomized greedy (weight-sorted, skip conflicts) + 1-swap/2-swap local improvement.
5. Multiple randomized restarts with different grid perturbations.
6. Final SLSQP polish on the best discrete solution.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(circles, tol=1e-8):
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-9):
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _max_radius_at(x, y, other_centers=None):
    """Max radius at (x,y) given walls and optionally other fixed centers."""
    r = min(x, 1.0 - x, y, 1.0 - y)
    if other_centers is not None:
        for ox, oy, orr in other_centers:
            d = np.sqrt((x - ox)**2 + (y - oy)**2)
            r = min(r, d - orr)
    return max(0.0, r)


def _build_grid_candidates(grid_res, radius_step, jitter=0.0):
    """Build list of candidate circles (x, y, r) on a grid.
    grid_res: number of grid points per side
    radius_step: step size for radius enumeration
    jitter: random offset added to grid positions
    Returns list of (x, y, r, idx) where idx is unique id.
    """
    candidates = []
    idx = 0
    xs = np.linspace(0.0, 1.0, grid_res)
    ys = np.linspace(0.0, 1.0, grid_res)
    
    for xi in xs:
        for yi in ys:
            x = xi + (random.uniform(-jitter, jitter) if jitter > 0 else 0.0)
            y = yi + (random.uniform(-jitter, jitter) if jitter > 0 else 0.0)
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            
            r_max = min(x, 1.0 - x, y, 1.0 - y)
            if r_max < 0.01:
                continue
            
            # Enumerate radii from r_max down
            r = r_max
            while r >= 0.01:
                candidates.append((x, y, r, idx))
                idx += 1
                r -= radius_step
            # Also add a few finer radii near r_max
            for frac in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
                r2 = r_max * frac
                if r2 >= 0.01:
                    candidates.append((x, y, r2, idx))
                    idx += 1
    
    return candidates


def _build_conflict_matrix(candidates, n_cand):
    """Build conflict matrix: conflict[i,j] = True if candidates i and j overlap.
    Also check boundary validity (already ensured by construction).
    Returns a set of frozenset pairs for memory efficiency, or a 2D array.
    """
    # Use a set of frozensets for memory
    conflicts = set()
    coords = np.array([(c[0], c[1]) for c in candidates])
    radii = np.array([c[2] for c in candidates])
    
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist_sq = dx * dx + dy * dy
            r_sum = radii[i] + radii[j]
            if dist_sq < r_sum * r_sum - 1e-12:
                conflicts.add((i, j))
    
    return conflicts


def _greedy_mwis(candidates, conflicts, n_select, weight_func=None):
    """Randomized greedy: sort by weight (radius), pick non-conflicting ones.
    Returns list of selected candidate indices.
    """
    n_cand = len(candidates)
    
    if weight_func is None:
        weights = np.array([c[2] for c in candidates])
    else:
        weights = np.array([weight_func(c) for c in candidates])
    
    order = np.argsort(-weights)
    
    selected = []
    selected_set = set()
    
    for idx in order:
        if idx in selected_set:
            continue
        # Check conflicts with already selected
        conflict = False
        for s in selected:
            if (min(idx, s), max(idx, s)) in conflicts:
                conflict = True
                break
        if not conflict:
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= n_select:
                break
    
    return selected


def _local_search_improve(candidates, conflicts, selected, n_select, max_iters=200, deadline=None):
    """1-swap and 2-swap local improvement on the MWIS solution."""
    selected_set = set(selected)
    current_sum = sum(candidates[i][2] for i in selected)
    
    # Precompute conflict sets for faster lookup
    conflict_sets = {}
    for (a, b) in conflicts:
        if a not in conflict_sets:
            conflict_sets[a] = set()
        if b not in conflict_sets:
            conflict_sets[b] = set()
        conflict_sets[a].add(b)
        conflict_sets[b].add(a)
    
    def get_conflicts(idx):
        return conflict_sets.get(idx, set())
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        
        # 1-swap: try replacing one selected with one unselected
        unselected = [i for i in range(len(candidates)) if i not in selected_set]
        random.shuffle(unselected)
        
        for s in list(selected):
            s_conflicts = get_conflicts(s)
            best_gain = 0
            best_replacement = -1
            
            for u in unselected:
                if u in s_conflicts:
                    continue
                # Check u doesn't conflict with other selected
                u_conflicts = get_conflicts(u)
                ok = True
                for s2 in selected:
                    if s2 != s and s2 in u_conflicts:
                        ok = False
                        break
                if not ok:
                    continue
                
                gain = candidates[u][2] - candidates[s][2]
                if gain > best_gain:
                    best_gain = gain
                    best_replacement = u
            
            if best_replacement >= 0 and best_gain > 1e-10:
                selected.remove(s)
                selected_set.remove(s)
                selected.append(best_replacement)
                selected_set.add(best_replacement)
                current_sum += best_gain
                improved = True
                break
        
        if improved:
            continue
        
        # 2-swap: try replacing two selected with two unselected
        if len(selected) >= 2:
            sel_list = list(selected)
            random.shuffle(sel_list)
            found_2swap = False
            
            for a_idx in range(min(20, len(sel_list))):
                for b_idx in range(a_idx + 1, min(20, len(sel_list))):
                    a = sel_list[a_idx]
                    b = sel_list[b_idx]
                    
                    a_conflicts = get_conflicts(a)
                    b_conflicts = get_conflicts(b)
                    
                    # Find unselected that don't conflict with (selected - {a,b})
                    remaining = [s for s in selected if s != a and s != b]
                    remaining_conflicts = set()
                    for s in remaining:
                        remaining_conflicts |= get_conflicts(s)
                    
                    candidates_for_2swap = []
                    for u in unselected:
                        if u in remaining_conflicts:
                            continue
                        candidates_for_2swap.append(u)
                    
                    if len(candidates_for_2swap) < 2:
                        continue
                    
                    # Find best pair
                    best_gain = 0
                    best_pair = None
                    pair_sum_old = candidates[a][2] + candidates[b][2]
                    
                    for i2 in range(min(30, len(candidates_for_2swap))):
                        for j2 in range(i2 + 1, min(30, len(candidates_for_2swap))):
                            u1 = candidates_for_2swap[i2]
                            u2 = candidates_for_2swap[j2]
                            if (min(u1, u2), max(u1, u2)) in conflicts:
                                continue
                            # Check u1, u2 don't conflict with a or b (they shouldn't since a,b removed)
                            gain = candidates[u1][2] + candidates[u2][2] - pair_sum_old
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (u1, u2)
                    
                    if best_pair and best_gain > 1e-10:
                        selected.remove(a)
                        selected.remove(b)
                        selected_set.remove(a)
                        selected_set.remove(b)
                        selected.append(best_pair[0])
                        selected.append(best_pair[1])
                        selected_set.add(best_pair[0])
                        selected_set.add(best_pair[1])
                        current_sum += best_gain
                        improved = True
                        found_2swap = True
                        break
                if found_2swap:
                    break
            
            if found_2swap:
                continue
        
        if not improved:
            break
    
    return selected, current_sum


def _discrete_search(grid_res, radius_step, n_restarts, deadline, jitter_range=(0.0, 0.02)):
    """Run discrete MWIS search with multiple restarts."""
    best_circles = None
    best_sum = -1
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        jitter = random.uniform(*jitter_range)
        candidates = _build_grid_candidates(grid_res, radius_step, jitter=jitter)
        n_cand = len(candidates)
        
        if n_cand < N:
            continue
        
        # Build conflict graph (this is the expensive part)
        t0 = time.time()
        conflicts = _build_conflict_matrix(candidates, n_cand)
        
        if time.time() > deadline - 5.0:
            break
        
        # Randomized greedy MWIS with multiple weight perturbations
        for trial in range(5):
            if time.time() > deadline - 3.0:
                break
            
            # Perturb weights for diversity
            if trial == 0:
                weight_func = None
            else:
                noise = np.random.uniform(0.9, 1.1, size=n_cand)
                def weight_func(c, noise=noise, idx_map={}):
                    return c[2] * noise[c[3] % len(noise)]
            
            selected = _greedy_mwis(candidates, conflicts, N, weight_func=weight_func)
            
            if len(selected) < N:
                continue
            
            # Local search improvement
            selected, sel_sum = _local_search_improve(
                candidates, conflicts, selected, N,
                max_iters=100, deadline=deadline - 3.0
            )
            
            if len(selected) >= N:
                circles = [(candidates[i][0], candidates[i][1], candidates[i][2]) for i in selected[:N]]
                s = _sum_radii(circles)
                if s > best_sum and _is_valid(circles, tol=1e-6):
                    best_circles = circles
                    best_sum = s
    
    return best_circles, best_sum


def _slsqp_polish(circles, deadline):
    """SLSQP polish with hard constraints."""
    params = np.array([val for c in circles for val in c])
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    
    constraints = []
    
    for i in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i] - p[3*i+2],
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2],
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i+1] - p[3*i+2],
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2],
        })
    
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
            })
    
    def objective(p):
        return -sum(p[3*i+2] for i in range(N))
    
    try:
        result = minimize(
            objective, params,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if result.success:
            new_circles = [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(N)]
            if _is_valid(new_circles, tol=1e-6):
                return new_circles
    except Exception:
        pass
    
    return circles


def _init_hexagonal():
    """Hexagonal packing initial layout."""
    circles = []
    r = 0.075
    rows = [5, 5, 5, 5, 6]
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    
    row_idx = 0
    for count in rows:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.03
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if len(circles) < N:
                circles.append((x, y, r))
        row_idx += 1
    
    circles = circles[:N]
    while len(circles) < N:
        circles.append((0.5, 0.5, 0.01))
    return circles


def _init_grid():
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.4 / side
        circles.append((x, y, r))
    return circles


def _softplus(x, beta=50):
    return np.log1p(np.exp(beta * x)) / beta


def _penalty_objective(params, penalty_weight=100.0):
    n = N
    penalty = 0.0
    for i in range(n):
        x, y, r = params[3*i], params[3*i+1], params[3*i+2]
        penalty += _softplus(r - x)
        penalty += _softplus(r - y)
        penalty += _softplus(r - (1.0 - x))
        penalty += _softplus(r - (1.0 - y))
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
            xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            penalty += _softplus(ri + rj - dist)
    return -sum(params[3*i+2] for i in range(n)) + penalty_weight * penalty


def _lbfgs_optimize(circles, deadline, max_iter=100):
    params = np.array([val for c in circles for val in c])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    try:
        result = minimize(
            _penalty_objective, params,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        if result.success:
            new_circles = [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(N)]
            if _is_valid(new_circles, tol=1e-6):
                return new_circles
    except Exception:
        pass
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_circles = None
    best_sum = -1
    
    # Phase 1: Discrete combinatorial search with fine grid
    # Use moderate grid resolution to keep conflict graph manageable
    grid_configs = [
        (15, 0.02, 8),   # 15x15 grid, radius step 0.02, 8 restarts
        (20, 0.02, 6),   # 20x20 grid
        (25, 0.03, 5),   # 25x25 grid, coarser radius
    ]
    
    for grid_res, radius_step, n_restarts in grid_configs:
        if time.time() > deadline - 30.0:
            break
        
        circles, s = _discrete_search(grid_res, radius_step, n_restarts, deadline)
        if circles is not None and s > best_sum:
            best_circles = circles
            best_sum = s
    
    # Phase 2: Also try from structured inits with L-BFGS + SLSQP
    inits = [_init_hexagonal(), _init_grid()]
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        lbfgs_res = _lbfgs_optimize(init, deadline, max_iter=80)
        if _is_valid(lbfgs_res, tol=1e-6) and _sum_radii(lbfgs_res) > best_sum:
            best_circles = lbfgs_res
            best_sum = _sum_radii(lbfgs_res)
    
    # Phase 3: SLSQP polish on best
    if best_circles is not None and time.time() < deadline - 10.0:
        polished = _slsqp_polish(best_circles, deadline)
        if _is_valid(polished, tol=1e-6) and _sum_radii(polished) > best_sum:
            best_circles = polished
            best_sum = _sum_radii(polished)
    
    # Phase 4: Another L-BFGS round from polished
    if best_circles is not None and time.time() < deadline - 5.0:
        lbfgs2 = _lbfgs_optimize(best_circles, deadline, max_iter=60)
        if _is_valid(lbfgs2, tol=1e-6) and _sum_radii(lbfgs2) > best_sum:
            best_circles = lbfgs2
            best_sum = _sum_radii(lbfgs2)
    
    # Fallback
    if best_circles is None:
        best_circles = _init_grid()
    
    best_circles = _post_process(best_circles)
    
    if not _is_valid(best_circles):
        best_circles = _post_process(_init_grid())
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
