# EVOLVE-BLOCK-START
import numpy as np
import time


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon that maximizes the minimum vertex equidistance score.

    Strategy:
      - Parameterize vertices by angle deviations from the regular 20-gon.
      - Use a low-order Fourier series for the angle perturbations to keep
        the polygon close to regular (hence convex) while allowing enough
        freedom to tune chord-length distributions.
      - Optimize with a bounded global search (differential-evolution-like)
        followed by local refinement (Nelder-Mead / gradient-free).
      - Deadline guard: stop before 630 s and return the best valid polygon.
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 5.0  # seconds before the hard limit

    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    # We'll perturb angles: theta_i = base_angles[i] + delta_i
    # Represent delta as a Fourier series with a few harmonics.
    # delta_i = sum_{m=1}^{M} (a_m * cos(m * phi_i) + b_m * sin(m * phi_i))
    # where phi_i = 2*pi*i/n (the base angle).
    # Using M=3 gives 6 parameters, which is manageable.
    M = 3
    n_params = 2 * M

    def params_to_polygon(params):
        """Convert Fourier coefficients to a (20, 2) polygon."""
        phi = base_angles
        delta = np.zeros(n)
        for m in range(1, M + 1):
            delta += params[2 * (m - 1)] * np.cos(m * phi)
            delta += params[2 * (m - 1) + 1] * np.sin(m * phi)
        thetas = base_angles + delta
        # Sort thetas to maintain cyclic order
        thetas_sorted = np.sort(thetas)
        # Ensure they are in [0, 2*pi)
        thetas_sorted = thetas_sorted % (2.0 * np.pi)
        thetas_sorted = np.sort(thetas_sorted)
        # If the polygon wraps, we need to make sure we have n distinct angles
        # in cyclic order. The sort gives us that.
        x = np.cos(thetas_sorted)
        y = np.sin(thetas_sorted)
        return np.stack([x, y], axis=1)

    def is_convex(polygon):
        """Check if polygon is strictly convex: all cross products same sign, nonzero."""
        n = len(polygon)
        cross_signs = []
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n]
            p2 = polygon[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) < 1e-12:
                return False
            cross_signs.append(np.sign(cross))
        if len(set(cross_signs)) != 1:
            return False
        return True

    def compute_score(polygon):
        """
        Compute the minimum vertex score as defined in the task.
        Returns 0 if invalid.
        """
        # Normalize: subtract centroid, divide by mean distance to centroid
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return 0.0
        normalized = centered / mean_dist

        # Check convexity
        if not is_convex(normalized):
            return 0.0

        # Check no repeated points
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                if np.linalg.norm(normalized[i] - normalized[j]) < 1e-10:
                    return 0.0

        n = len(normalized)
        min_score = 1.0
        for v_idx in range(n):
            v = normalized[v_idx]
            # Distances to all other vertices
            d = np.linalg.norm(normalized - v, axis=1)
            d[v_idx] = np.inf  # exclude self
            d_sorted = np.sort(d)
            # d_sorted has 19 elements (d_0 to d_18)

            # If any distance < 0.05, score 0
            if d_sorted[0] < 0.05:
                return 0.0

            # Find k: first index minimizing d_{k+1} - d_k
            gaps = np.diff(d_sorted)  # length 18
            k = np.argmin(gaps)

            D = (d_sorted[k] + d_sorted[k + 1]) / 2.0
            if D < 0.1:
                return 0.0

            # Select four distances
            if k == 0:
                selected = d_sorted[0:4]
            elif k == 17:
                selected = d_sorted[15:19]
            else:
                selected = d_sorted[k - 1:k + 3]

            # raw = mean of max(d/D, D/d) over selected
            ratios = np.maximum(selected / D, D / selected)
            raw = np.mean(ratios)
            vertex_score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
            min_score = min(min_score, vertex_score)

        return min_score

    def objective(params):
        """Negative score for minimization."""
        if time.time() - start_time > deadline - safety_margin:
            return 1.0  # high penalty to stop
        try:
            poly = params_to_polygon(params)
            return -compute_score(poly)
        except Exception:
            return 1.0

    # Initial best: regular polygon
    best_params = np.zeros(n_params)
    best_poly = params_to_polygon(best_params)
    best_score = compute_score(best_poly)

    # Differential Evolution-like global search
    np.random.seed(42)
    population_size = 40
    bounds = [(-0.3, 0.3)] * n_params  # perturbations in radians

    # Initialize population
    population = np.random.uniform(
        [b[0] for b in bounds],
        [b[1] for b in bounds],
        (population_size, n_params)
    )
    # Include the zero vector
    population[0] = np.zeros(n_params)
    population[1] = best_params.copy()

    # Evaluate initial population
    scores = np.array([objective(p) for p in population])

    # DE parameters
    F = 0.8  # differential weight
    CR = 0.9  # crossover rate
    max_generations = 200

    for gen in range(max_generations):
        if time.time() - start_time > deadline - safety_margin:
            break

        # Update best
        best_idx = np.argmin(scores)
        best_params = population[best_idx].copy()
        best_score = -scores[best_idx]
        if best_score > 0.999:
            break

        # Generate new population
        new_population = population.copy()
        for i in range(population_size):
            if time.time() - start_time > deadline - safety_margin:
                break

            # Select three distinct individuals
            a, b, c = np.random.choice(
                [j for j in range(population_size) if j != i],
                size=3,
                replace=False
            )

            # Differential evolution trial vector
            trial = population[a] + F * (population[b] - population[c])

            # Crossover
            crossover_mask = np.random.random(n_params) < CR
            if not np.any(crossover_mask):
                crossover_mask[np.random.randint(n_params)] = True
            trial = np.where(crossover_mask, trial, population[i])

            # Bound check
            trial = np.clip(trial, [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate
            trial_score = objective(trial)

            # Selection
            if trial_score <= scores[i]:
                new_population[i] = trial
                scores[i] = trial_score

        population = new_population

        # Track global best
        best_idx = np.argmin(scores)
        if -scores[best_idx] > best_score:
            best_params = population[best_idx].copy()
            best_score = -scores[best_idx]
            best_poly = params_to_polygon(best_params)

    # Local refinement using Nelder-Mead on the best parameters
    if time.time() - start_time < deadline - safety_margin:
        from scipy.optimize import minimize

        def local_objective(params):
            if time.time() - start_time > deadline - safety_margin:
                return 1.0
            try:
                poly = params_to_polygon(params)
                return -compute_score(poly)
            except Exception:
                return 1.0

        # Try a few local refinements from different starting points
        local_starts = [best_params]
        # Add a few random perturbations around best
        for _ in range(5):
            pert = best_params + np.random.normal(0, 0.05, n_params)
            pert = np.clip(pert, [b[0] for b in bounds], [b[1] for b in bounds])
            local_starts.append(pert)

        for ls in local_starts:
            if time.time() - start_time > deadline - safety_margin:
                break
            try:
                res = minimize(
                    local_objective,
                    ls,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                             'adaptive': True}
                )
                if res.fun < -best_score:
                    best_params = res.x
                    best_score = -res.fun
                    best_poly = params_to_polygon(best_params)
            except Exception:
                pass

    # Final check
    final_poly = params_to_polygon(best_params)
    final_score = compute_score(final_poly)
    if final_score > best_score:
        best_poly = final_poly
        best_score = final_score

    # Ensure the returned polygon is valid
    if not is_convex(best_poly):
        # Fallback to regular polygon
        best_poly = params_to_polygon(np.zeros(n_params))

    return best_poly


# EVOLVE-BLOCK-END
