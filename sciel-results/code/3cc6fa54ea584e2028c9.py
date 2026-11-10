# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _build_sofa_grid():
    """Build the sofa grid points (450, 300) x (450, 300) coordinates."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    return XX, YY


def _point_in_base(u, v):
    """Vectorized check: is (u,v) inside the base L-corridor?
    Base = {0<=u<=1, -3<=v<=1} U {-3<=u<=1, 0<=v<=1}
    """
    c1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    c2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return c1 | c2


def _densify_path(path):
    """Densify path by inserting linear interpolations between consecutive rows.
    The evaluator prepends (0,0,0) and appends (-2,0,90).
    Between consecutive rows it adds m = max(1, ceil(|a2-a1|)) equal steps in (x,y,a).
    """
    full = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    out = [full[0]]
    for i in range(len(full) - 1):
        a1 = full[i, 2]
        a2 = full[i + 1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        if m > 1:
            for k in range(1, m):
                t = k / m
                pt = full[i] + t * (full[i + 1] - full[i])
                out.append(pt)
        out.append(full[i + 1])
    return np.array(out)


def _evaluate_area(path, XX, YY):
    """Compute the area (number of grid points * 0.0001) of the intersection of all
    corridor placements along the densified path."""
    placements = _densify_path(path)
    n_pts = XX.shape[0] * XX.shape[1]
    # Flatten grid
    xf = XX.ravel()
    yf = YY.ravel()
    # Start with all True
    inside = np.ones(n_pts, dtype=bool)
    for (px, py, pa) in placements:
        a_rad = np.deg2rad(pa)
        ca = np.cos(a_rad)
        sa = np.sin(a_rad)
        # R(-a) rotates by -a: [cos(-a), -sin(-a); sin(-a), cos(-a)] = [ca, sa; -sa, ca]
        # u = ca*(x-px) + sa*(y-py)
        # v = -sa*(x-px) + ca*(y-py)
        dx = xf - px
        dy = yf - py
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        inside &= _point_in_base(u, v)
        # Early exit if nothing left
        if not inside.any():
            return 0.0
    return float(inside.sum()) * 0.0001


def _initial_path():
    """Generate an initial path close to the known-good baseline."""
    return np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)


def _mutate(path, sigma):
    """Gaussian perturbation of the path."""
    n = path.shape[0]
    pert = np.random.randn(n, 3) * sigma
    # Constrain angles to be roughly increasing and in (0, 90)
    pert[:, 2] *= 0.5  # smaller angle perturbation
    new_path = path + pert
    # Ensure angles are in a reasonable range and roughly monotonic
    # We don't strictly enforce monotonicity but keep angles in (-10, 100)
    new_path[:, 2] = np.clip(new_path[:, 2], -10.0, 100.0)
    # Keep x,y in a reasonable range
    new_path[:, 0] = np.clip(new_path[:, 0], -2.0, 1.0)
    new_path[:, 1] = np.clip(new_path[:, 1], -1.0, 2.0)
    return new_path


def _crossover(p1, p2):
    """Uniform crossover."""
    mask = np.random.rand(p1.shape[0], 3) < 0.5
    return np.where(mask, p1, p2)


def moving_sofa_2d() -> np.ndarray:
    """
    Evolutionary optimization of the moving sofa path.
    Returns: np.ndarray of shape (n, 3), the optimized path.
    """
    t_start = time.time()
    DEADLINE = 620.0  # safety margin before 630s kill

    XX, YY = _build_sofa_grid()

    # Initial population
    pop_size = 60
    base = _initial_path()
    population = [base.copy()]
    for _ in range(pop_size - 1):
        population.append(_mutate(base, 0.15))

    # Evaluate initial population
    scores = np.array([_evaluate_area(p, XX, YY) for p in population])
    best_idx = int(np.argmax(scores))
    best_path = population[best_idx].copy()
    best_score = scores[best_idx]

    # Evolutionary loop
    gen = 0
    elite_frac = 0.1
    elite_count = max(1, int(pop_size * elite_frac))
    sigma = 0.12
    stall_count = 0

    while time.time() - t_start < DEADLINE:
        gen += 1
        # Sort by score
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = scores[order]

        # Update best
        if scores[0] > best_score:
            best_score = scores[0]
            best_path = population[0].copy()
            stall_count = 0
        else:
            stall_count += 1

        # Reduce sigma if stalled
        if stall_count > 10:
            sigma *= 0.85
            if sigma < 0.005:
                break

        # Create next generation
        new_pop = []
        # Elites
        for i in range(elite_count):
            new_pop.append(population[i].copy())

        # Fill rest with offspring
        while len(new_pop) < pop_size:
            # Pick parents via tournament
            i1 = int(np.random.randint(0, min(10, pop_size)))
            i2 = int(np.random.randint(0, min(10, pop_size)))
            if scores[i1] >= scores[i2]:
                p1 = population[i1]
            else:
                p1 = population[i2]
            # Mutation
            child = _mutate(p1, sigma)
            new_pop.append(child)

        population = new_pop
        # Evaluate new population
        # Batch evaluation with deadline check
        new_scores = np.empty(pop_size)
        for i in range(pop_size):
            if time.time() - t_start > DEADLINE - 30:
                # Not enough time, keep old scores for unevaluated
                new_scores[i] = -1.0
            else:
                new_scores[i] = _evaluate_area(population[i], XX, YY)
        scores = new_scores

    # Final polish: local refinement around best
    if time.time() - t_start < DEADLINE - 10:
        polish_sigma = 0.02
        for _ in range(50):
            if time.time() - t_start > DEADLINE - 5:
                break
            cand = _mutate(best_path, polish_sigma)
            s = _evaluate_area(cand, XX, YY)
            if s > best_score:
                best_score = s
                best_path = cand
            polish_sigma *= 0.95

    # Ensure valid output
    if not np.all(np.isfinite(best_path)):
        best_path = base

    return best_path


# EVOLVE-BLOCK-END
