"""Circle packing in unit square: 26 circles, maximize sum of radii.
Uses a multi-replica simulated annealing / replica-exchange Monte Carlo
with a final SLSQP local polish under explicit inequality constraints.
"""
N = 26

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 115.0  # seconds; keep margin before 120s evaluator kill


def _is_valid(pos, rad):
    """Check all circles are inside [0,1]^2 and disjoint."""
    if np.any(pos < -1e-9) or np.any(pos > 1 + 1e-9):
        return False
    if np.any(rad < 1e-12):
        return False
    # wall constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    if np.any(rad > pos[:, 0] + 1e-9) or np.any(rad > 1 - pos[:, 0] + 1e-9):
        return False
    if np.any(rad > pos[:, 1] + 1e-9) or np.any(rad > 1 - pos[:, 1] + 1e-9):
        return False
    # pair constraints
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2
            rs = rad[i] + rad[j]
            if d2 < rs * rs - 1e-9:
                return False
    return True


def _sum_radii(rad):
    return float(np.sum(rad))


def _slsqp_polish(pos, rad, timeout=20.0):
    """Polish a feasible packing with SLSQP using explicit constraints."""
    x0 = np.concatenate([pos.ravel(), rad])
    n = len(x0)

    def objective(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        # Small penalty for infeasibility to guide SLSQP
        pen = 0.0
        for i in range(N):
            pen += max(0, r[i] - p[i, 0]) ** 2
            pen += max(0, r[i] - (1 - p[i, 0])) ** 2
            pen += max(0, r[i] - p[i, 1]) ** 2
            pen += max(0, r[i] - (1 - p[i, 1])) ** 2
        for i in range(N):
            for j in range(i + 1, N):
                d2 = (p[i, 0] - p[j, 0]) ** 2 + (p[i, 1] - p[j, 1]) ** 2
                rs = r[i] + r[j]
                if d2 < rs * rs:
                    pen += (rs - np.sqrt(d2)) ** 2
        return -np.sum(r) + 100.0 * pen

    # Constraints: g(x) >= 0
    cons = []
    # Wall constraints
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i] - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2 * i] - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i + 1] - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2 * i + 1] - x[2 * N + i]})
    # Pair constraints: dist^2 - (r_i + r_j)^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[2 * i] - x[2 * j]) ** 2 +
                    (x[2 * i + 1] - x[2 * j + 1]) ** 2 -
                    (x[2 * N + i] + x[2 * N + j]) ** 2
                )
            })
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((1e-6, 1.0))  # x_i
        bounds.append((1e-6, 1.0))  # y_i
    for i in range(N):
        bounds.append((1e-6, 0.5))  # r_i

    t_start = time.time()
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 200, 'ftol': 1e-10})
        if res.success or res.fun < objective(x0):
            new_pos = res.x[:2 * N].reshape(N, 2)
            new_rad = res.x[2 * N:]
            if _is_valid(new_pos, new_rad):
                return new_pos, new_rad
    except Exception:
        pass
    return pos, rad


def _random_feasible_init(rng):
    """Generate a random feasible packing via sequential placement with random radii."""
    pos = np.zeros((N, 2))
    rad = np.zeros(N)
    # Place circles one by one, choosing random positions and shrinking radius to fit
    for i in range(N):
        max_r = 0.5
        for attempt in range(200):
            x = rng.uniform(0, 1)
            y = rng.uniform(0, 1)
            # Max radius allowed by walls
            r_wall = min(x, 1 - x, y, 1 - y)
            r_max = min(max_r, r_wall)
            if r_max < 1e-4:
                continue
            # Check against existing circles
            feasible_r = r_max
            for j in range(i):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                feasible_r = min(feasible_r, max(0, d - rad[j]))
            if feasible_r > 1e-4:
                # Pick a random fraction of feasible radius to allow more space
                r = feasible_r * rng.uniform(0.5, 1.0)
                pos[i] = [x, y]
                rad[i] = r
                break
        else:
            # Fallback: place at a corner-ish position with tiny radius
            pos[i] = [0.5, 0.5]
            rad[i] = 0.01
    return pos, rad


def _grid_init():
    """Generate a grid-based initial packing."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    # Use first N
    arr = np.array(cells[:N])
    return arr[:, :2], arr[:, 2]


def _replica_exchange_mc(pos0, rad0, n_replicas=8, total_steps=50000, rng=None):
    """Parallel tempering Monte Carlo to optimize sum of radii."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Initialize replicas with slightly perturbed versions
    replicas = []
    for rep in range(n_replicas):
        p = pos0.copy()
        r = rad0.copy()
        if rep > 0:
            # Perturb positions and radii
            p = p + rng.normal(0, 0.02, p.shape)
            r = np.clip(r * rng.uniform(0.8, 1.2, N), 1e-4, 0.5)
            # Project back to feasibility (shrink radii to fix overlaps)
            for _ in range(10):
                for i in range(N):
                    r[i] = min(r[i], p[i, 0], 1 - p[i, 0], p[i, 1], 1 - p[i, 1])
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.sqrt((p[i, 0] - p[j, 0]) ** 2 + (p[i, 1] - p[j, 1]) ** 2)
                        # If overlapping, shrink the larger one
                        if d < r[i] + r[j]:
                            excess = r[i] + r[j] - d
                            # Shrink proportionally
                            r[i] = max(1e-4, r[i] - 0.5 * excess)
                            r[j] = max(1e-4, r[j] - 0.5 * excess)
                # Ensure wall constraints
                for i in range(N):
                    r[i] = min(r[i], p[i, 0], 1 - p[i, 0], p[i, 1], 1 - p[i, 1])
        replicas.append((p, r))

    # Temperatures (higher = more exploration)
    temps = np.linspace(0.01, 1.0, n_replicas)
    # Track best
    best_pos = pos0.copy()
    best_rad = rad0.copy()
    best_val = _sum_radii(rad0)

    for step in range(total_steps):
        # Pick a random replica to update
        rep_idx = rng.integers(n_replicas)
        p, r = replicas[rep_idx]
        temp = temps[rep_idx]

        # Random move: perturb one circle's position and/or radius
        ci = rng.integers(N)
        move_type = rng.integers(3)
        if move_type == 0:
            # Move position
            dx = rng.normal(0, 0.05)
            dy = rng.normal(0, 0.05)
            new_p = p.copy()
            new_p[ci, 0] += dx
            new_p[ci, 1] += dy
            new_r = r.copy()
        elif move_type == 1:
            # Grow/shrink radius
            delta = rng.normal(0, 0.02)
            new_p = p.copy()
            new_r = r.copy()
            new_r[ci] += delta
        else:
            # Both
            dx = rng.normal(0, 0.03)
            dy = rng.normal(0, 0.03)
            delta = rng.normal(0, 0.01)
            new_p = p.copy()
            new_p[ci, 0] += dx
            new_p[ci, 1] += dy
            new_r = r.copy()
            new_r[ci] += delta

        # Check feasibility of new state
        if _is_valid(new_p, new_r):
            new_val = _sum_radii(new_r)
            old_val = _sum_radii(r)
            delta_val = new_val - old_val
            # Metropolis criterion (we want to maximize, so accept if improves or with prob)
            if delta_val >= 0:
                accept = True
            else:
                # Accept with probability exp(delta_val / temp)
                if temp > 1e-6:
                    accept = rng.random() < np.exp(delta_val / temp)
                else:
                    accept = False
            if accept:
                replicas[rep_idx] = (new_p, new_r)
                if new_val > best_val:
                    best_val = new_val
                    best_pos = new_p.copy()
                    best_rad = new_r.copy()
        else:
            # Reject infeasible moves
            pass

        # Replica exchange: try to swap two adjacent replicas
        if step % 5 == 0 and n_replicas > 1:
            i = rng.integers(n_replicas - 1)
            j = i + 1
            pi, ri = replicas[i]
            pj, rj = replicas[j]
            vi = _sum_radii(ri)
            vj = _sum_radii(rj)
            ti = temps[i]
            tj = temps[j]
            # Compute acceptance probability for swap
            # P_swap ∝ exp(vi/tj + vj/ti - vi/ti - vj/tj)
            log_ratio = vi / tj + vj / ti - vi / ti - vj / tj
            if log_ratio >= 0 or rng.random() < np.exp(log_ratio):
                replicas[i] = (pj, rj)
                replicas[j] = (pi, ri)

    return best_pos, best_rad


def construct():
    t_start = time.time()
    rng = np.random.default_rng(12345)

    # Start with grid init
    pos0, rad0 = _grid_init()
    best_pos = pos0.copy()
    best_rad = rad0.copy()
    best_val = _sum_radii(rad0)

    # Phase 1: SLSQP polish on grid init
    pos1, rad1 = _slsqp_polish(pos0, rad0, timeout=15.0)
    if _sum_radii(rad1) > best_val:
        best_val = _sum_radii(rad1)
        best_pos = pos1.copy()
        best_rad = rad1.copy()

    # Check time
    if time.time() - t_start > _DEADLINE - 60:
        return np.column_stack([best_pos, best_rad])

    # Phase 2: Generate several random initializations and run replica exchange MC
    n_inits = 6
    for init_i in range(n_inits):
        if time.time() - t_start > _DEADLINE - 40:
            break
        if init_i == 0:
            # Use the best from SLSQP as one start
            p_start, r_start = best_pos.copy(), best_rad.copy()
        else:
            p_start, r_start = _random_feasible_init(rng)

        # Run replica exchange MC with time budget
        remaining = _DEADLINE - 30 - (time.time() - t_start)
        steps = int(remaining * 800)  # ~800 steps per second budget
        steps = max(steps, 10000)
        steps = min(steps, 60000)

        p_mc, r_mc = _replica_exchange_mc(p_start, r_start, n_replicas=6,
                                          total_steps=steps, rng=rng)
        val_mc = _sum_radii(r_mc)
        if val_mc > best_val:
            best_val = val_mc
            best_pos = p_mc.copy()
            best_rad = r_mc.copy()

        # SLSQP polish on MC result
        if time.time() - t_start < _DEADLINE - 20:
            p_pol, r_pol = _slsqp_polish(p_mc, r_mc, timeout=10.0)
            val_pol = _sum_radii(r_pol)
            if val_pol > best_val:
                best_val = val_pol
                best_pos = p_pol.copy()
                best_rad = r_pol.copy()

    # Final SLSQP polish
    if time.time() - t_start < _DEADLINE - 5:
        p_final, r_final = _slsqp_polish(best_pos, best_rad, timeout=10.0)
        if _sum_radii(r_final) > best_val:
            best_pos = p_final
            best_rad = r_final
            best_val = _sum_radii(r_final)

    # Clamp to ensure validity
    for i in range(N):
        best_pos[i, 0] = np.clip(best_pos[i, 0], 0, 1)
        best_pos[i, 1] = np.clip(best_pos[i, 1], 0, 1)
        best_rad[i] = min(best_rad[i], best_pos[i, 0], 1 - best_pos[i, 0],
                          best_pos[i, 1], 1 - best_pos[i, 1])
        best_rad[i] = max(best_rad[i], 1e-6)

    # Fix any remaining overlaps by shrinking
    for _ in range(50):
        fixed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((best_pos[i, 0] - best_pos[j, 0]) ** 2 +
                            (best_pos[i, 1] - best_pos[j, 1]) ** 2)
                rs = best_rad[i] + best_rad[j]
                if d < rs - 1e-9:
                    excess = rs - d
                    best_rad[i] = max(1e-6, best_rad[i] - 0.5 * excess)
                    best_rad[j] = max(1e-6, best_rad[j] - 0.5 * excess)
                    fixed = True
            best_rad[i] = min(best_rad[i], best_pos[i, 0], 1 - best_pos[i, 0],
                              best_pos[i, 1], 1 - best_pos[i, 1])
        if not fixed:
            break

    return np.column_stack([best_pos, best_rad])
# EVOLVE-BLOCK-END
