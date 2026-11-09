# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon that maximizes the minimum vertex equidistance score.

    Strategy:
      - Parameterize vertices using a radial distance function r(theta) = 1 + sum c_k cos(k*theta)
        evaluated at 20 equally-spaced angles. This directly controls chord lengths.
      - For small coefficients, convexity is guaranteed.
      - Use differential evolution for global search, then Nelder-Mead for local refinement.
      - Deadline guard: stop before 630 s and return the best valid polygon.
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0

    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    # Fourier series for radial perturbations: r(theta) = 1 + sum_{k=1}^{M} c_k cos(k*theta)
    M = 5
    n_params = M

    def params_to_polygon(params):
        """Convert radial Fourier coefficients to a (20, 2) polygon in cyclic order."""
        # params has length M, coefficients c_1..c_M
        radii = np.ones(n)
        for k in range(1, M + 1):
            radii += params[k - 1] * np.cos(k * base_angles)
        
        # Ensure radii are positive
        radii = np.maximum(radii, 0.1)
        
        x = radii * np.cos(base_angles)
        y = radii * np.sin(base_angles)
        return np.stack([x, y], axis=1)

    def is_convex(polygon):
        """Check if polygon is strictly convex."""
        n = len(polygon)
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n]
            p2 = polygon[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) < 1e-12:
                return False
            if i == 0:
                sign0 = np.sign(cross)
            else:
                if np.sign(cross) != sign0:
                    return False
        return True

    def normalize_polygon(polygon):
        """Normalize: subtract centroid, divide by mean distance to centroid."""
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return None
        return centered / mean_dist

    def vertex_scores(polygon):
        """Compute per-vertex scores. Returns array of 20 scores."""
        normalized = normalize_polygon(polygon)
        if normalized is None:
            return np.zeros(n)

        if not is_convex(normalized):
            return np.zeros(n)

        # Check no repeated points
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                if np.linalg.norm(normalized[i] - normalized[j]) < 1e-10:
                    return np.zeros(n)

        scores = np.ones(n)
        for v_idx in range(n):
            v = normalized[v_idx]
            d = np.linalg.norm(normalized - v, axis=1)
            d[v_idx] = np.inf
            d_sorted = np.sort(d)

            if d_sorted[0] < 0.05:
                scores[v_idx] = 0.0
                continue

            gaps = np.diff(d_sorted)
            k = int(np.argmin(gaps))

            D = (d_sorted[k] + d_sorted[k + 1]) / 2.0
            if D < 0.1:
                scores[v_idx] = 0.0
                continue

            if k == 0:
                selected = d_sorted[0:4]
            elif k == 17:
                selected = d_sorted[15:19]
            else:
                selected = d_sorted[k - 1:k + 3]

            ratios = np.maximum(selected / D, D / selected)
            raw = np.mean(ratios)
            scores[v_idx] = min(1.0, 1.0 / raw) if raw > 0 else 0.0

        return scores

    def compute_score(polygon):
        """Compute the minimum vertex score."""
        scores = vertex_scores(polygon)
        return np.min(scores)

    def objective(params):
        """Negative exact score for minimization."""
        if time.time() - start_time > deadline - safety_margin:
            return 1.0
        try:
            poly = params_to_polygon(params)
            return -compute_score(poly)
        except Exception:
            return 1.0

    # Initial best: regular polygon (all params zero)
    best_params = np.zeros(n_params)
    best_poly = params_to_polygon(best_params)
    best_score = compute_score(best_poly)

    # Differential Evolution global search
    np.random.seed(42)
    population_size = 60
    bounds = [(-0.3, 0.3)] * n_params

    population = np.random.uniform(
        [b[0] for b in bounds],
        [b[1] for b in bounds],
        (population_size, n_params)
    )
    population[0] = np.zeros(n_params)
    population[1] = best_params.copy()

    scores_arr = np.array([objective(p) for p in population])

    F = 0.8
    CR = 0.9
    max_generations = 400

    for gen in range(max_generations):
        if time.time() - start_time > deadline - safety_margin:
            break

        best_idx = np.argmin(scores_arr)
        current_best_score = -scores_arr[best_idx]
        if current_best_score > best_score:
            best_score = current_best_score
            best_params = population[best_idx].copy()
            best_poly = params_to_polygon(best_params)
        
        if best_score > 0.999:
            break

        new_population = population.copy()
        for i in range(population_size):
            if time.time() - start_time > deadline - safety_margin:
                break

            candidates = [j for j in range(population_size) if j != i]
            a, b, c = np.random.choice(candidates, size=3, replace=False)

            trial = population[a] + F * (population[b] - population[c])

            crossover_mask = np.random.random(n_params) < CR
            if not np.any(crossover_mask):
                crossover_mask[np.random.randint(n_params)] = True
            trial = np.where(crossover_mask, trial, population[i])

            trial = np.clip(trial, [b[0] for b in bounds], [b[1] for b in bounds])

            trial_score = objective(trial)

            if trial_score <= scores_arr[i]:
                new_population[i] = trial
                scores_arr[i] = trial_score

        population = new_population

    # Local refinement with Nelder-Mead
    if time.time() - start_time < deadline - safety_margin:
        def local_objective(params):
            if time.time() - start_time > deadline - safety_margin:
                return 1.0
            try:
                poly = params_to_polygon(params)
                return -compute_score(poly)
            except Exception:
                return 1.0

        # Multiple local starts from the best and its neighborhood
        local_starts = [best_params]
        for scale in [0.02, 0.05, 0.1, 0.2]:
            for _ in range(5):
                pert = best_params + np.random.normal(0, scale, n_params)
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
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10,
                             'adaptive': True}
                )
                if res.fun < -best_score:
                    best_params = res.x
                    best_score = -res.fun
                    best_poly = params_to_polygon(best_params)
            except Exception:
                pass

    # Final high-precision polish
    if time.time() - start_time < deadline - safety_margin:
        def final_objective(params):
            if time.time() - start_time > deadline - safety_margin:
                return 1.0
            try:
                poly = params_to_polygon(params)
                return -compute_score(poly)
            except Exception:
                return 1.0

        try:
            res = minimize(
                final_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12,
                         'adaptive': True}
            )
            if res.fun < -best_score:
                best_params = res.x
                best_score = -res.fun
                best_poly = params_to_polygon(best_params)
        except Exception:
            pass

    # Final validation
    final_poly = params_to_polygon(best_params)
    final_score = compute_score(final_poly)
    if final_score > best_score:
        best_poly = final_poly
        best_score = final_score

    if not is_convex(best_poly):
        best_poly = params_to_polygon(np.zeros(n_params))

    return best_poly


# EVOLVE-BLOCK-END
