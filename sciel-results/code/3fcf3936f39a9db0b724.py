"""Circle packing in unit square: pack 26 disjoint circles maximizing sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 26

# EVOLVE-BLOCK-START

def _objective(params, soft=1.0):
    """
    params: shape (3N,) = [x_1..x_N, y_1..y_N, r_1..r_N]
    Objective to MINIMIZE: -sum(r) + penalty for constraint violations.
    Constraints:
      - Circles inside [0,1]^2: x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
      - Disjoint interiors: dist(i,j) >= r_i + r_j for all i != j
    We use a soft penalty: sum of violations * lambda.
    """
    n = len(params) // 3
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]

    # Boundary violations
    bnd = np.maximum(0, -x + r) + np.maximum(0, x + r - 1) + \
          np.maximum(0, -y + r) + np.maximum(0, y + r - 1)
    bnd_penalty = np.sum(bnd)

    # Overlap violations
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            viol = (r[i] + r[j]) - d
            if viol > 0:
                overlap_penalty += viol

    total_penalty = bnd_penalty + overlap_penalty
    return -np.sum(r) + soft * total_penalty


def _objective_smooth(params, soft=1.0, beta=50.0):
    """
    Smoother version using log-sum-exp for soft penalties.
    """
    n = len(params) // 3
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]

    # Boundary violations (soft)
    bnd_viol = np.concatenate([
        -x + r,
        x + r - 1,
        -y + r,
        y + r - 1
    ])
    # Soft sum of positive parts
    bnd_penalty = np.sum(np.logaddexp(0, beta * bnd_viol) - np.log(1 + np.exp(0))) / beta if beta > 0 else np.sum(np.maximum(0, bnd_viol))
    # Simpler: just use max(0, viol) but smooth via softplus
    bnd_penalty = np.sum(np.log1p(np.exp(beta * bnd_viol)) / beta)

    # Overlap violations
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = np.sqrt(max(d_sq, 1e-12))
            viol = (r[i] + r[j]) - d
            overlap_penalty += np.log1p(np.exp(beta * viol)) / beta

    total_penalty = bnd_penalty + overlap_penalty
    return -np.sum(r) + soft * total_penalty


def _is_feasible(params, tol=1e-7):
    """Check if a packing is feasible."""
    n = len(params) // 3
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]

    # Boundary
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol) or \
       np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False

    # Overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] - tol:
                return False
    return True


def _project_feasible(params, tol=1e-10):
    """Project a near-feasible solution to be strictly feasible by shrinking radii slightly if needed."""
    n = len(params) // 3
    x = params[0:n].copy()
    y = params[n:2*n].copy()
    r = params[2*n:3*n].copy()

    # Ensure boundaries
    for i in range(n):
        r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
        r[i] = max(r[i], 1e-10)

    # Ensure no overlaps by shrinking
    changed = True
    max_iter = 100
    it = 0
    while changed and it < max_iter:
        changed = False
        it += 1
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - tol:
                    # Shrink both radii proportionally
                    excess = (r[i] + r[j]) - d + tol
                    # Split the excess
                    if r[i] + r[j] > 0:
                        r[i] -= excess * r[i] / (r[i] + r[j])
                        r[j] -= excess * r[j] / (r[i] + r[j])
                    else:
                        r[i] -= excess / 2
                        r[j] -= excess / 2
                    r[i] = max(r[i], 1e-10)
                    r[j] = max(r[j], 1e-10)
                    changed = True
        # Re-check boundaries after shrinking
        for i in range(n):
            r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
            r[i] = max(r[i], 1e-10)

    return np.concatenate([x, y, r])


def _generate_seeds():
    """Generate diverse initial seeds."""
    seeds = []

    # Seed 1: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side))
    cells = cells[:N]
    params = np.array([c[0] for c in cells] + [c[1] for c in cells] + [c[2] for c in cells])
    seeds.append(params)

    # Seed 2: Hexagonal packing pattern
    # Try to arrange in a hex-like pattern
    rows = []
    cols_per_row = []
    remaining = N
    row = 0
    while remaining > 0:
        if row % 2 == 0:
            cols = min(6, remaining)
        else:
            cols = min(5, remaining)
        cols_per_row.append(cols)
        remaining -= cols
        row += 1

    # Place hexagonally
    spacing = 1.0 / (max(cols_per_row) + 0.5)
    x_list, y_list, r_list = [], [], []
    row_idx = 0
    for cols in cols_per_row:
        y_pos = (row_idx + 0.5) / (len(cols_per_row) + 0.3)
        offset = 0.5 * spacing if row_idx % 2 == 1 else 0
        for col in range(cols):
            x_pos = (col + 0.5) / (max(cols_per_row) + 0.5) + offset * 0.5
            x_pos = min(max(x_pos, 0.05), 0.95)
            y_pos = min(max(y_pos, 0.05), 0.95)
            x_list.append(x_pos)
            y_list.append(y_pos)
            r_list.append(spacing * 0.45)
        row_idx += 1

    params = np.array(x_list + y_list + r_list)
    seeds.append(params)

    # Seed 3: Random
    np.random.seed(42)
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.03, 0.08, N)
    params = np.concatenate([x, y, r])
    seeds.append(params)

    # Seed 4: Another random with different seed
    np.random.seed(123)
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.02, 0.1, N)
    params = np.concatenate([x, y, r])
    seeds.append(params)

    # Seed 5: Clusters
    np.random.seed(7)
    x = np.random.uniform(0.2, 0.8, N)
    y = np.random.uniform(0.2, 0.8, N)
    r = np.full(N, 0.05)
    params = np.concatenate([x, y, r])
    seeds.append(params)

    return seeds


def _anneal_optimize(params_init, deadline, max_iters=2000):
    """
    Run L-BFGS-B with annealing of the penalty parameter.
    Returns best feasible solution found.
    """
    best_params = None
    best_score = -np.inf

    # Annealing schedule for soft penalty
    # Start with high penalty to enforce feasibility, then reduce to allow growth
    n_phases = 8
    soft_values = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    iters_per_phase = max(50, max_iters // n_phases)

    current = params_init.copy()

    for phase in range(n_phases):
        if time.time() > deadline - 5:
            break

        soft = soft_values[phase]
        beta = 30.0 + 20.0 * phase  # Increase smoothness sharpness

        # Run L-BFGS-B
        try:
            res = minimize(
                _objective_smooth,
                current,
                method='L-BFGS-B',
                jac=None,
                options={'maxiter': iters_per_phase, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
            )
            current = res.x
        except Exception:
            pass

        # Project to feasibility
        current = _project_feasible(current)

        # Evaluate
        if _is_feasible(current, tol=1e-6):
            score = np.sum(current[2*N:3*N])
            if score > best_score:
                best_score = score
                best_params = current.copy()

    # Final polish: more iterations with lower penalty
    if time.time() < deadline - 10 and best_params is not None:
        try:
            res = minimize(
                _objective_smooth,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            polished = _project_feasible(res.x)
            if _is_feasible(polished, tol=1e-6):
                score = np.sum(polished[2*N:3*N])
                if score > best_score:
                    best_score = score
                    best_params = polished
        except Exception:
            pass

    return best_params, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    seeds = _generate_seeds()

    best_params = None
    best_score = -np.inf

    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 15:
            break

        # Add small perturbation for diversity
        np.random.seed(1000 + seed_idx)
        perturbed = seed + np.random.normal(0, 0.01, len(seed))
        # Ensure radii positive
        perturbed[2*N:3*N] = np.maximum(perturbed[2*N:3*N], 0.01)
        # Ensure in bounds
        perturbed[0:N] = np.clip(perturbed[0:N], 0.05, 0.95)
        perturbed[N:2*N] = np.clip(perturbed[N:2*N], 0.05, 0.95)

        params, score = _anneal_optimize(perturbed, deadline)

        if params is not None and score > best_score:
            best_score = score
            best_params = params

    # If no feasible solution found, use the seed grid as fallback
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_params = np.array([c[0] for c in cells[:N]] + [c[1] for c in cells[:N]] + [c[2] for c in cells[:N]])

    # Final projection
    best_params = _project_feasible(best_params)

    # Build output array
    x = best_params[0:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]

    result = np.column_stack([x, y, r])
    return result

# EVOLVE-BLOCK-END
