"""
Circle packing in a perimeter-4 rectangle.
Combined strategy: Differential evolution for global center/aspect optimization,
followed by force-directed local refinement with radius growth.

The DE stage optimizes center positions and box aspect ratio to maximize the
sum of maximum feasible radii (a smooth-ish objective). The force-directed
stage then refines the solution by allowing radii to grow while resolving
overlaps and wall collisions, exploiting the local geometry better.
"""
import numpy as np
import time
import random

N = 21
DEADLINE = 115.0  # Safety margin before 120s limit


def _max_feasible_radii(centers, w, h):
    """Compute the maximum feasible radius for each circle given centers and box."""
    n = len(centers)
    max_r = np.zeros(n)
    for i in range(n):
        xi, yi = centers[i]
        # Distance to walls
        r = min(xi, w - xi, yi, h - yi)
        if r <= 0:
            max_r[i] = 0.0
            continue
        # Distance to other circles
        for j in range(n):
            if i != j:
                dx = xi - centers[j][0]
                dy = yi - centers[j][1]
                d = np.sqrt(dx * dx + dy * dy)
                r = min(r, d / 2.0)
                if r <= 0:
                    break
        max_r[i] = max(0.0, r)
    return max_r


def _objective(x):
    """
    Objective for DE: returns negative sum of max feasible radii.
    x = [w, x1, y1, x2, y2, ..., xN, yN]
    """
    w = x[0]
    h = 2.0 - w
    if w < 0.15 or h < 0.15:
        return 1e6

    centers = x[1:].reshape(N, 2)
    # Clamp centers to box
    centers[:, 0] = np.clip(centers[:, 0], 0.0, w)
    centers[:, 1] = np.clip(centers[:, 1], 0.0, h)

    max_r = _max_feasible_radii(centers, w, h)
    return -np.sum(max_r)


def _differential_evolution(objective, bounds, pop_size=30, max_iter=300,
                            mutation=(0.5, 1.0), crossover_prob=0.7,
                            time_limit=100.0):
    """Simple differential evolution."""
    n_dims = len(bounds)
    start_time = time.time()

    # Initialize population
    pop = np.zeros((pop_size, n_dims))
    for i in range(pop_size):
        for j in range(n_dims):
            pop[i, j] = random.uniform(bounds[j][0], bounds[j][1])

    # Evaluate initial population
    fitness = np.array([objective(ind) for ind in pop])

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]

    for gen in range(max_iter):
        if time.time() - start_time > time_limit:
            break

        new_pop = []
        new_fitness = []

        for i in range(pop_size):
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            F = random.uniform(*mutation)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            for j in range(n_dims):
                mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])

            trial = pop[i].copy()
            for j in range(n_dims):
                if random.random() < crossover_prob:
                    trial[j] = mutant[j]

            f_trial = objective(trial)
            if f_trial <= fitness[i]:
                new_pop.append(trial)
                new_fitness.append(f_trial)
            else:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])

        pop = np.array(new_pop)
        fitness = np.array(new_fitness)

        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_f:
            best_f = fitness[gen_best_idx]
            best_x = pop[gen_best_idx].copy()

    return best_x, best_f


def _force_refine(w, h, x, y, r, max_iter=200, step_size=0.02, growth_rate=0.005,
                  deadline=None):
    """
    Force-directed refinement: resolve overlaps and grow radii.
    Returns (x, y, r, w, h).
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = len(x)

    if deadline is None:
        deadline = time.time() + 5.0

    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Circle-Circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    if dist > 1e-8:
                        overlap = min_dist - dist
                        force = overlap * 5.0
                        fx[i] += force * (dx / dist)
                        fy[i] += force * (dy / dist)
                        fx[j] -= force * (dx / dist)
                        fy[j] -= force * (dy / dist)
                    else:
                        fx[i] += random.uniform(-0.5, 0.5)
                        fy[i] += random.uniform(-0.5, 0.5)
                        fx[j] -= random.uniform(-0.5, 0.5)
                        fy[j] -= random.uniform(-0.5, 0.5)

        # Wall repulsion
        for i in range(n):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 5.0
            if w - x[i] < r[i]:
                fx[i] -= (r[i] - (w - x[i])) * 5.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 5.0
            if h - y[i] < r[i]:
                fy[i] -= (r[i] - (h - y[i])) * 5.0

        # Update positions
        x += fx * step_size
        y += fy * step_size
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)

        # Try to grow radii
        new_r = r * (1.0 + growth_rate)
        valid = True
        for i in range(n):
            if new_r[i] > x[i] or new_r[i] > w - x[i] or new_r[i] > y[i] or new_r[i] > h - y[i]:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    if dx * dx + dy * dy < (new_r[i] + new_r[j]) ** 2:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            r = new_r

    return x, y, r, w, h


def _validate(w, h, x, y, r, tol=1e-9):
    if w < 0.1 or h < 0.1:
        return False
    if np.any(r < 0):
        return False
    if np.any(x - r < -tol) or np.any(w - x - r < -tol):
        return False
    if np.any(y - r < -tol) or np.any(h - y - r < -tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j] - tol) ** 2:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    random.seed(42)
    np.random.seed(42)

    best_result = None
    best_score = -np.inf

    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score
        if _validate(w, h, x, y, r):
            x_c = np.clip(x, 0, w)
            y_c = np.clip(y, 0, h)
            r_c = np.clip(r, 0, None)
            score = np.sum(r_c)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x_c, y_c, r_c])

    # Bounds for DE: w in [0.3, 1.7], centers in [0, 2]
    bounds = [(0.3, 1.7)]
    for _ in range(N * 2):
        bounds.append((0.01, 1.99))

    # Run DE with multiple aspect ratio seeds
    aspect_seeds = [1.0, 0.8, 1.2, 0.9, 1.1]

    for aspect in aspect_seeds:
        if time.time() > deadline - 20.0:
            break

        # Set up DE with initial population biased towards target aspect
        w_target = 2.0 * aspect / (1.0 + aspect)
        h_target = 2.0 - w_target

        # Custom initialization
        pop_size = 25
        n_dims = 1 + N * 2
        pop = np.zeros((pop_size, n_dims))
        for i in range(pop_size):
            # First few individuals start near target aspect
            if i < 5:
                pop[i, 0] = w_target + random.uniform(-0.1, 0.1)
            else:
                pop[i, 0] = random.uniform(0.3, 1.7)
            for j in range(N):
                pop[i, 1 + 2 * j] = random.uniform(0.05, w_target - 0.05)
                pop[i, 2 + 2 * j] = random.uniform(0.05, h_target - 0.05)

        # Evaluate initial population
        fitness = np.array([objective(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # DE iterations
        de_start = time.time()
        de_time_limit = min(15.0, deadline - time.time() - 25.0)
        if de_time_limit < 3.0:
            continue

        for gen in range(100):
            if time.time() - de_start > de_time_limit:
                break

            new_pop = []
            new_fitness = []
            for i in range(pop_size):
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                F = random.uniform(0.5, 1.0)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                for j in range(n_dims):
                    mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])

                trial = pop[i].copy()
                for j in range(n_dims):
                    if random.random() < 0.7:
                        trial[j] = mutant[j]

                f_trial = objective(trial)
                if f_trial <= fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(f_trial)
                else:
                    new_pop.append(pop[i])
                    new_fitness.append(fitness[i])

            pop = np.array(new_pop)
            fitness = np.array(new_fitness)

            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_f:
                best_f = fitness[gen_best_idx]
                best_x = pop[gen_best_idx].copy()

        # Extract solution
        w = np.clip(best_x[0], 0.3, 1.7)
        h = 2.0 - w
        centers = best_x[1:].reshape(N, 2)
        centers[:, 0] = np.clip(centers[:, 0], 0.0, w)
        centers[:, 1] = np.clip(centers[:, 1], 0.0, h)

        x = centers[:, 0]
        y = centers[:, 1]
        r = _max_feasible_radii(centers, w, h)

        # Force-directed refinement
        if time.time() < deadline - 10.0:
            refine_deadline = min(deadline - 5.0, time.time() + 8.0)
            x, y, r, w, h = _force_refine(w, h, x, y, r, max_iter=150,
                                           step_size=0.015, growth_rate=0.003,
                                           deadline=refine_deadline)

        update_best(w, h, x, y, r)

    # Final polish if time remains
    if best_result is not None and time.time() < deadline - 5.0:
        # Reconstruct from best_result
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        # Estimate w, h from bounding box
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        r_max = np.max(r)
        w = x_max + r_max
        h = y_max + r_max
        # Adjust to perimeter 4
        w = np.clip(w, 0.5, 1.5)
        h = 2.0 - w
        # Re-clamp
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)

        if time.time() < deadline - 3.0:
            x, y, r, w, h = _force_refine(w, h, x, y, r, max_iter=80,
                                           step_size=0.005, growth_rate=0.001,
                                           deadline=deadline - 2.0)
            update_best(w, h, x, y, r)

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        n_cols = 5
        n_rows = 5
        cell_w = w / n_cols
        cell_h = h / n_rows
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * cell_w
                y[idx] = (i + 0.5) * cell_h
                r[idx] = min(cell_w, cell_h) * 0.4
                idx += 1
        best_result = np.column_stack([x, y, r])

    return best_result
