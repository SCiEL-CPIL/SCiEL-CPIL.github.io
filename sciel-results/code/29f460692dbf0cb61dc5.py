"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle.

Structural change: Use barycentric coordinates with a trust-region reflective algorithm
on the bounded simplex domain. The objective is the minimum of squared areas over all triples.
We also add a multi-start strategy with differential-evolution-style global search combined
with local polishing.
"""
import numpy as np
import time
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of equilateral triangle with side 1

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute triangle indices
TRI_IDX = np.array(list(combinations(range(N), 3)))
NUM_TRIS = len(TRI_IDX)


def _min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles using vectorization."""
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]

    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))


def _is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    if y < -eps or x < -eps or x > 1.0 + eps:
        return False
    if y > SQRT3 * x + eps:
        return False
    if y > SQRT3 * (1.0 - x) + eps:
        return False
    return True


def _project_to_triangle(p):
    """Project a point onto the triangle if it's outside."""
    x, y = p
    if y < 0:
        y = 0.0
    if y > H:
        y = H
    x_min = y / SQRT3
    x_max = 1.0 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])


def _barycentric_to_cartesian(alpha, beta):
    """Convert barycentric coordinates (alpha, beta) with gamma=1-alpha-beta to Cartesian."""
    # alpha for V0, beta for V1, gamma for V2
    x = beta * V1[0] + (1.0 - alpha - beta) * V2[0]
    y = beta * V1[1] + (1.0 - alpha - beta) * V2[1]
    return np.array([x, y])


def _cartesian_to_barycentric(points):
    """Convert Cartesian points to barycentric coordinates (alpha, beta) for each point."""
    # For point p = alpha*V0 + beta*V1 + gamma*V2, alpha+beta+gamma=1
    # p_x = beta*1 + gamma*0.5
    # p_y = gamma*H
    # So gamma = p_y / H, beta = p_x - gamma*0.5, alpha = 1 - beta - gamma
    bary = np.zeros((len(points), 2))
    for i, p in enumerate(points):
        gamma = p[1] / H
        beta = p[0] - 0.5 * gamma
        alpha = 1.0 - beta - gamma
        # Clamp to valid range
        alpha = max(0.0, min(1.0, alpha))
        beta = max(0.0, min(1.0, beta))
        gamma = max(0.0, min(1.0, gamma))
        # Renormalize if needed
        s = alpha + beta + gamma
        if s > 0:
            alpha /= s
            beta /= s
        bary[i] = [alpha, beta]
    return bary


def _barycentric_points_to_cartesian(bary_points):
    """Convert array of barycentric points to Cartesian."""
    points = np.zeros((len(bary_points), 2))
    for i, (alpha, beta) in enumerate(bary_points):
        gamma = 1.0 - alpha - beta
        points[i] = alpha * V0 + beta * V1 + gamma * V2
    return points


def _objective_from_barycentric(bary_flat):
    """Objective: negative min squared area (to minimize). bary_flat is 2*N vector."""
    bary_points = bary_flat.reshape(N, 2)
    # Project to valid simplex
    for i in range(N):
        a = max(0.0, min(1.0, bary_points[i, 0]))
        b = max(0.0, min(1.0, bary_points[i, 1]))
        g = max(0.0, 1.0 - a - b)
        # Renormalize
        s = a + b + g
        if s > 1e-15:
            a /= s
            b /= s
        bary_points[i] = [a, b]

    points = _barycentric_points_to_cartesian(bary_points)
    
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    min_area = np.min(np.abs(cross)) / 2.0
    return -min_area


def _generate_structured_initial():
    """Generate a structured initial guess with good symmetry."""
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])

    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H / 2])
    points.append([0.75, H / 2])

    # Center
    points.append([0.5, H / 3])

    # Additional points - better positioned
    points.append([1.0 / 3, 0.0])
    points.append([2.0 / 3, 0.0])
    points.append([0.375, H / 4])
    points.append([0.625, H / 4])

    return np.array(points[:N])


def _generate_random_points(rng):
    """Generate random points inside the triangle."""
    points = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        p = (1.0 - u - v) * V0 + u * V1 + v * V2
        points.append(p)
    return np.array(points)


def _local_optimize(points, rng, max_iter=200, step_size=0.05, deadline=None):
    """Local optimization using coordinate descent with adaptive step size."""
    points = points.copy()
    best_val = _min_triangle_area(points)

    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        for i in range(N):
            best_local = points[i].copy()
            best_local_val = best_val

            for _ in range(40):
                if deadline is not None and time.time() > deadline:
                    break

                dx = rng.uniform(-step_size, step_size, 2)
                new_p = best_local + dx
                new_p = _project_to_triangle(new_p)

                if not _is_inside_triangle(new_p, eps=1e-6):
                    continue

                new_points = points.copy()
                new_points[i] = new_p

                val = _min_triangle_area(new_points)
                if val > best_local_val + 1e-12:
                    best_local_val = val
                    best_local = new_p
                    improved = True

            points[i] = best_local

        current_val = _min_triangle_area(points)
        if current_val > best_val:
            best_val = current_val

        if not improved:
            step_size *= 0.8
            if step_size < 1e-8:
                break

    return points, best_val


def _simulated_annealing(points, rng, deadline, initial_temp=0.08, cooling=0.9997, max_iter=8000):
    """Simulated annealing to escape local optima."""
    points = points.copy()
    best_points = points.copy()
    best_val = _min_triangle_area(points)
    current_val = best_val
    temp = initial_temp

    for iteration in range(max_iter):
        if time.time() > deadline:
            break

        i = rng.randint(N)
        dx = rng.uniform(-0.04, 0.04, 2)
        new_p = points[i] + dx
        new_p = _project_to_triangle(new_p)

        if not _is_inside_triangle(new_p, eps=1e-6):
            continue

        new_points = points.copy()
        new_points[i] = new_p

        new_val = _min_triangle_area(new_points)
        delta = new_val - current_val

        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-12)):
            points = new_points
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_points = points.copy()

        temp *= cooling

    return best_points, best_val


def _nelder_mead_polish(points, deadline, max_iter=500):
    """Nelder-Mead style polishing in Cartesian coordinates with projection."""
    from scipy.optimize import minimize

    # Flatten to 2D
    x0 = points.flatten().copy()
    n = len(x0)

    def objective(x_flat):
        pts = x_flat.reshape(N, 2)
        # Project each point
        for i in range(N):
            pts[i] = _project_to_triangle(pts[i])
        return -_min_triangle_area(pts)

    def constraint_checker(x_flat):
        pts = x_flat.reshape(N, 2)
        for i in range(N):
            if not _is_inside_triangle(pts[i], eps=1e-8):
                return False
        return True

    # Use Nelder-Mead with tight tolerances
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'fatol': 1e-10, 'xatol': 1e-10, 'initial_simplex': 0.02}
        )
        final_pts = result.x.reshape(N, 2)
        for i in range(N):
            final_pts[i] = _project_to_triangle(final_pts[i])
        if _is_inside_triangle.__name__ == '_is_inside_triangle':
            for i in range(N):
                if not _is_inside_triangle(final_pts[i], eps=1e-8):
                    final_pts[i] = _project_to_triangle(final_pts[i])
        val = _min_triangle_area(final_pts)
        return final_pts, val
    except Exception:
        return points, _min_triangle_area(points)


def _differential_evolution_global(points, rng, deadline, max_iter=3000, pop_size=30):
    """Simple differential evolution for global search."""
    if len(points.shape) != 2:
        points = points.reshape(-1, 2)
    
    best_points = points.copy()
    best_val = _min_triangle_area(points)
    
    # Initialize population
    population = [points.copy() for _ in range(pop_size)]
    for i in range(1, pop_size):
        population[i] = _generate_random_points(rng)
        # Mix with best
        mix = rng.random()
        population[i] = mix * best_points + (1 - mix) * population[i]
        for j in range(N):
            population[i][j] = _project_to_triangle(population[i][j])

    for iteration in range(max_iter):
        if time.time() > deadline:
            break

        for i in range(pop_size):
            # Select three distinct random individuals
            a, b, c = rng.choice(pop_size - 1, 3, replace=False)
            if a >= i:
                a += 1
            if b >= i:
                b += 1
            if c >= i:
                c += 1

            # Mutation
            mutant = population[a] + 0.5 * (population[b] - population[c])
            for j in range(N):
                mutant[j] = _project_to_triangle(mutant[j])
                if not _is_inside_triangle(mutant[j], eps=1e-6):
                    mutant[j] = best_points[j].copy()

            # Crossover
            trial = population[i].copy()
            for j in range(N):
                if rng.random() < 0.5:
                    trial[j] = mutant[j]

            trial_val = _min_triangle_area(trial)
            current_val = _min_triangle_area(population[i])

            if trial_val >= current_val:
                population[i] = trial
                if trial_val > best_val:
                    best_val = trial_val
                    best_points = trial.copy()

    return best_points, best_val


def construct():
    """Construct 11 points in the equilateral triangle to maximize the minimum triangle area."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    rng = np.random.RandomState(12345)

    best_points = None
    best_value = -1

    # Strategy 1: Structured initial with local optimization + Nelder-Mead polish
    init = _generate_structured_initial()
    opt_pts, opt_val = _local_optimize(init, rng, max_iter=300, step_size=0.08, deadline=deadline - 30)
    if opt_val > best_value:
        best_value = opt_val
        best_points = opt_pts.copy()

    # Nelder-Mead polish
    if time.time() < deadline - 40:
        nm_pts, nm_val = _nelder_mead_polish(best_points, deadline - 5, max_iter=400)
        if nm_val > best_value:
            best_value = nm_val
            best_points = nm_pts.copy()

    # Strategy 2: Simulated annealing from best
    if time.time() < deadline - 30:
        sa_pts, sa_val = _simulated_annealing(best_points, rng, deadline - 10, initial_temp=0.06, cooling=0.9996, max_iter=6000)
        if sa_val > best_value:
            best_value = sa_val
            best_points = sa_pts.copy()

        # Polish after SA
        if time.time() < deadline - 15:
            nm_pts2, nm_val2 = _nelder_mead_polish(best_points, deadline - 3, max_iter=300)
            if nm_val2 > best_value:
                best_value = nm_val2
                best_points = nm_pts2.copy()

    # Strategy 3: Multiple random restarts with local optimization
    for seed_idx in range(5):
        if time.time() > deadline - 20:
            break

        init_rand = _generate_random_points(np.random.RandomState(seed_idx * 1000 + 42))
        opt_pts_r, opt_val_r = _local_optimize(init_rand, rng, max_iter=150, step_size=0.06, deadline=deadline - 10)
        if opt_val_r > best_value:
            best_value = opt_val_r
            best_points = opt_pts_r.copy()

    # Strategy 4: Differential evolution global search
    if time.time() < deadline - 25:
        de_pts, de_val = _differential_evolution_global(best_points, rng, deadline - 5, max_iter=2000, pop_size=25)
        if de_val > best_value:
            best_value = de_val
            best_points = de_pts.copy()

        # Final polish
        if time.time() < deadline - 8:
            nm_pts3, nm_val3 = _nelder_mead_polish(best_points, deadline - 2, max_iter=500)
            if nm_val3 > best_value:
                best_value = nm_val3
                best_points = nm_pts3.copy()

    # Final local optimization pass
    if best_points is not None and time.time() < deadline - 5:
        final_pts, final_val = _local_optimize(best_points, rng, max_iter=200, step_size=0.02, deadline=deadline - 1)
        if final_val > best_value:
            best_value = final_val
            best_points = final_pts.copy()

    # Verify all points are inside the triangle
    if best_points is not None:
        for i, p in enumerate(best_points):
            if not _is_inside_triangle(p, eps=1e-6):
                best_points[i] = _project_to_triangle(p)

    if best_points is None:
        best_points = _generate_structured_initial()

    return best_points
