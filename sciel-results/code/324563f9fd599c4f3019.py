# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using a structured population with
    weighted-sum niching, local refinement, and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 300  # 300s limit, keep margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Fixed reference point for hypervolume (not used in optimization, just for context)
        # ref = np.array([1.1, 1.3, 1.1])

        # Use numpy random with seed set by evaluator
        # The evaluator calls np.random.seed(seed), so we use np.random directly

        # Initial population: mix of random points and structured points
        # Use a smaller initial population to save budget for refinement
        init_size = min(30, budget // 5)
        X = np.random.uniform(lo, hi, size=(init_size, dim))
        F = np.array([problem(x) for x in X])
        used = init_size

        # Helper: check if point is dominated by any point in F
        def is_dominated(f, F_set):
            # f is dominated if there exists f' in F_set such that f' <= f and f' != f
            for fp in F_set:
                if np.all(fp <= f) and np.any(fp < f):
                    return True
            return False

        # Helper: find non-dominated set
        def non_dominated_indices(F_set):
            n = len(F_set)
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j:
                        if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                            dominated = True
                            break
                if not dominated:
                    nd.append(i)
            return nd

        # Weighted-sum niching: define a set of weight vectors
        # For 3 objectives, use a simplex of weight vectors
        # Generate weight vectors on the simplex
        n_weights = 20
        weights = []
        # Simplex-lattice style
        for i in range(n_weights + 1):
            for j in range(n_weights + 1 - i):
                k = n_weights - i - j
                if k >= 0:
                    w = np.array([i, j, k], dtype=float) / n_weights
                    weights.append(w)
        weights = np.array(weights)
        n_weights = len(weights)

        # Assign each initial point to its closest weight
        # For each point, find which weight it best satisfies
        def assign_weight(f, weights):
            # Normalize f to [0,1] range based on current min/max
            # Use a simple approach: for each weight, compute weighted sum
            # and find the weight that minimizes the weighted sum
            best_w = 0
            best_val = np.inf
            for idx, w in enumerate(weights):
                val = np.dot(w, f)
                if val < best_val:
                    best_val = val
                    best_w = idx
            return best_w

        # Track which weight each point is assigned to
        # This helps maintain diversity

        # Main loop: alternate between exploration and exploitation
        # Use a population size that fits within budget
        pop_size = min(40, budget // 4)

        # If we have more budget, use a larger population
        if budget >= 100:
            pop_size = min(60, budget // 3)

        # Ensure we have enough budget for initial + main loop
        if used + pop_size > budget:
            pop_size = budget - used

        # Pad or trim initial population to pop_size
        if len(X) < pop_size:
            extra = pop_size - len(X)
            X_new = np.random.uniform(lo, hi, size=(extra, dim))
            F_new = np.array([problem(x) for x in X_new])
            used += extra
            X = np.vstack([X, X_new])
            F = np.vstack([F, F_new])
        elif len(X) > pop_size:
            # Keep the best pop_size points (non-dominated first, then by weighted sum)
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) >= pop_size:
                # Keep first pop_size non-dominated
                keep = nd_idx[:pop_size]
            else:
                keep = list(nd_idx)
                # Add more points sorted by sum of objectives
                remaining = [i for i in range(len(F)) if i not in keep]
                remaining.sort(key=lambda i: np.sum(F[i]))
                keep.extend(remaining[:pop_size - len(keep)])
            X = X[keep]
            F = F[keep]

        # Main optimization loop
        # Use adaptive mutation: start with larger steps, shrink over time
        base_step = 0.15
        min_step = 0.01
        n_iter = budget - used

        # Track best hypervolume proxy: sum of (ref - f) for non-dominated points
        # Actually, let's just track non-dominated points

        # Local refinement function
        def local_refine(x, f, budget_left):
            """Perform local refinement around x using gradient-free search."""
            if budget_left < 5:
                return x, f, 0

            # Try small perturbations in each dimension
            best_x = x.copy()
            best_f = f.copy()
            used_local = 0
            step = base_step * 0.5

            for _ in range(min(4, budget_left // 2)):
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        new_x = best_x.copy()
                        new_x[d] += direction * step * (hi[d] - lo[d])
                        new_x = np.clip(new_x, lo, hi)
                        new_f = problem(new_x)
                        used_local += 1
                        # Check if new point dominates best_f
                        if np.all(new_f <= best_f) and np.any(new_f < best_f):
                            best_x = new_x
                            best_f = new_f
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < min_step:
                        break

            return best_x, best_f, used_local

        # Weighted-sum local optimization
        def weighted_sum_optimize(w, x_init, f_init, budget_left):
            """Optimize weighted sum using Nelder-Mead style search."""
            if budget_left < 5:
                return x_init, f_init, 0

            # Use scipy minimize if available, otherwise manual
            try:
                from scipy.optimize import minimize

                def obj(x):
                    return np.dot(w, problem(x))

                res = minimize(obj, x_init, method='Nelder-Mead',
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': max(1, budget_left // 2),
                                        'xatol': 1e-6, 'fatol': 1e-8})
                f_final = problem(res.x)
                used_local = res.nfev if hasattr(res, 'nfev') else budget_left // 2
                return res.x, f_final, min(used_local, budget_left)
            except Exception:
                # Fallback: simple coordinate descent
                best_x = x_init.copy()
                best_f = f_init.copy()
                best_ws = np.dot(w, best_f)
                used_local = 0
                step = base_step

                for _ in range(min(10, budget_left)):
                    improved = False
                    for d in range(dim):
                        for direction in [-1, 1]:
                            new_x = best_x.copy()
                            new_x[d] += direction * step * (hi[d] - lo[d])
                            new_x = np.clip(new_x, lo, hi)
                            new_f = problem(new_x)
                            used_local += 1
                            new_ws = np.dot(w, new_f)
                            if new_ws < best_ws:
                                best_x = new_x
                                best_f = new_f
                                best_ws = new_ws
                                improved = True
                    if not improved:
                        step *= 0.5
                        if step < min_step:
                            break

                return best_x, best_f, used_local

        # Main loop
        iteration = 0
        stagnation_counter = 0
        best_nd_size = 0

        while used < budget and (time.time() - t_start) < deadline - 5:
            iteration += 1

            # Find non-dominated points
            nd_idx = non_dominated_indices(F)
            nd_size = len(nd_idx)

            # Update stagnation counter
            if nd_size > best_nd_size:
                best_nd_size = nd_size
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Adaptive step size
            progress = used / budget
            step_size = base_step * (1 - 0.7 * progress) + min_step * 0.7 * progress

            # Choose strategy based on stagnation
            if stagnation_counter > 10:
                # Restart some points
                n_restart = max(1, pop_size // 5)
                for _ in range(n_restart):
                    if used >= budget:
                        break
                    idx = np.random.randint(len(X))
                    X[idx] = np.random.uniform(lo, hi)
                    F[idx] = problem(X[idx])
                    used += 1
                continue

            # Pick a parent: prefer non-dominated points
            if nd_size > 0:
                # Weighted random selection from non-dominated points
                # Use crowding distance approximation: distance to nearest neighbor
                nd_F = F[nd_idx]
                if len(nd_idx) > 1:
                    # Compute crowding distance
                    distances = np.zeros(len(nd_idx))
                    for i in range(len(nd_idx)):
                        for j in range(len(nd_idx)):
                            if i != j:
                                distances[i] += np.linalg.norm(nd_F[i] - nd_F[j])
                    # Normalize
                    distances = distances / (np.max(distances) + 1e-10)
                    # Sample with probability proportional to distance
                    probs = distances / np.sum(distances)
                    parent_local = np.random.choice(len(nd_idx), p=probs)
                else:
                    parent_local = 0
                parent_idx = nd_idx[parent_local]
            else:
                parent_idx = np.random.randint(len(X))

            # Generate child via mutation
            child = X[parent_idx].copy()
            # Gaussian mutation
            child += step_size * (hi - lo) * np.random.randn(dim)
            child = np.clip(child, lo, hi)

            # Evaluate child
            child_f = problem(child)
            used += 1

            # Decide whether to accept child
            # Accept if it dominates parent or is non-dominated
            dominates_parent = np.all(child_f <= F[parent_idx]) and np.any(child_f < F[parent_idx])
            is_nd = not is_dominated(child_f, F)

            if dominates_parent:
                X[parent_idx] = child
                F[parent_idx] = child_f
            elif is_nd:
                # Replace the worst point (highest sum of objectives)
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = child_f
            else:
                # Try local refinement on the child
                if used < budget - 5:
                    refined_x, refined_f, local_used = local_refine(child, child_f, budget - used)
                    used += local_used
                    if np.all(refined_f <= child_f) and np.any(refined_f < child_f):
                        child = refined_x
                        child_f = refined_f
                        is_nd = not is_dominated(child_f, F)
                        if is_nd:
                            worst = int(np.argmax(np.sum(F, axis=1)))
                            X[worst] = child
                            F[worst] = child_f

            # Occasionally do weighted-sum optimization on a non-dominated point
            if iteration % 5 == 0 and used < budget - 10 and nd_size > 0:
                # Pick a random non-dominated point
                nd_idx_current = non_dominated_indices(F)
                if len(nd_idx_current) > 0:
                    parent_idx2 = nd_idx_current[np.random.randint(len(nd_idx_current))]
                    # Pick a random weight
                    w = weights[np.random.randint(n_weights)]
                    if used < budget - 10:
                        refined_x, refined_f, local_used = weighted_sum_optimize(
                            w, X[parent_idx2], F[parent_idx2], budget - used)
                        used += local_used
                        if np.all(refined_f <= F[parent_idx2]) and np.any(refined_f < F[parent_idx2]):
                            X[parent_idx2] = refined_x
                            F[parent_idx2] = refined_f
                        elif not is_dominated(refined_f, F):
                            worst = int(np.argmax(np.sum(F, axis=1)))
                            X[worst] = refined_x
                            F[worst] = refined_f

        return X
# EVOLVE-BLOCK-END
