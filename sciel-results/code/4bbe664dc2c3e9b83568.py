"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based multi-objective optimizer (MOEA/D style)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Generate weight vectors for decomposition (3 objectives)
        n_obj = self.n_obj
        dim = self.dim
        # Use a simplex-based weight generation for 3 objectives
        # We'll create a set of weight vectors on the 2-simplex
        n_w = 15  # number of weight vectors / subproblems
        weights = []
        # Generate points on the 2-simplex using a grid
        levels = 4  # resolution
        for i in range(levels + 1):
            for j in range(levels + 1 - i):
                k = levels - i - j
                w = np.array([i, j, k], dtype=float) / levels
                weights.append(w)
        weights = np.array(weights)
        # Subsample if too many
        if len(weights) > n_w:
            idx = rng.choice(len(weights), size=n_w, replace=False)
            weights = weights[idx]
        n_sub = len(weights)

        # Initialize population with random points
        X = lo + span * rng.random((n_sub, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = n_sub

        # Ideal point (best known per objective)
        ideal = np.min(F, axis=0)

        # For each subproblem, keep track of its best solution
        # We'll maintain a population of n_sub solutions, one per weight vector

        # Local search neighborhood: each subproblem connects to a few others
        # For simplicity, use a fixed neighborhood size
        T = 5  # neighborhood size

        # Precompute neighbors based on weight similarity
        neighbors = []
        for i in range(n_sub):
            dists = np.sum((weights - weights[i]) ** 2, axis=1)
            dists[i] = np.inf
            nbrs = np.argsort(dists)[:T]
            neighbors.append(nbrs)

        # Tchebycheff scalarization
        def tchebycheff(f, w, ideal):
            # Normalize by ideal point (or use a reference)
            # Use a simple normalization: divide by (ideal + small) to avoid div by zero
            # Actually for Tchebycheff we use: max_i w_i * (f_i - z_i^*) / (F_i^max - z_i^*)
            # But since we don't have F_max, use a simpler approach
            # Use weighted sum as a fallback for simplicity, or use the standard Tchebycheff
            # with ideal point only
            # Standard: h(x,w) = max_i { w_i * |f_i(x) - z_i^*| }
            # But this doesn't handle scaling well. Let's use a normalized version.
            # For 3 objectives, let's use a simple weighted sum with the ideal point
            # Actually, let's use the standard Tchebycheff approach:
            # h(x, w) = max_i { w_i * (f_i(x) - z_i^*) / (F_i^max - z_i^*) }
            # Since we don't have F_max, we'll use a running max
            pass
            return None

        # Use a simpler approach: for each subproblem, minimize a scalarized objective
        # We'll use weighted sum with the current ideal point for normalization
        # But to keep it simple and effective, let's use a hybrid:
        # - Use weighted sum for selection
        # - Use local search to improve

        # Actually, let's implement a cleaner MOEA/D:
        # 1. For each subproblem i with weight w_i, minimize h_i(x) = max_j { w_ij * (f_j(x) - z_j^*) / (F_j^max - z_j^*) }
        # 2. Update z^* and F^max as we go
        # 3. Local search: for each subproblem, try to improve using neighbors

        # Let's use a simpler but effective approach:
        # - Maintain population
        # - Each iteration: pick a subproblem, do local search around it
        # - Replace the worst solution in the neighborhood

        # Initialize F_max for normalization
        F_max = np.max(F, axis=0)

        def tchebycheff_val(f, w):
            # h(x,w) = max_i { w_i * (f_i - ideal_i) / (F_max_i - ideal_i + eps) }
            denom = F_max - ideal + 1e-10
            norm = (f - ideal) / denom
            return np.max(w * norm)

        def scalarize(F_row, w):
            return np.array([tchebycheff_val(f, w) for f in F_row])

        # Main loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Pick a subproblem to work on
            i = rng.integers(n_sub)
            w_i = weights[i]

            # Get neighborhood
            nbrs = neighbors[i]

            # Local search: mutate the solution at i
            # Adaptive mutation: start larger, shrink over time
            progress = used / self.budget
            mut_rate = 0.3 * (1 - progress) + 0.05  # from 0.3 to 0.05
            child = X[i] + mut_rate * span * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update ideal and F_max
            ideal = np.minimum(ideal, fc)
            F_max = np.maximum(F_max, fc)

            # Determine which solution to replace in the neighborhood
            # Compare child against each neighbor using Tchebycheff
            best_replace = -1
            best_score = np.inf

            for j in nbrs:
                # Score the child and the current solution at j using weight w_j
                # Actually, in MOEA/D, we compare using the weight of the subproblem
                # Let's compare using w_i for consistency
                score_child = tchebycheff_val(fc, w_i)
                score_cur = tchebycheff_val(F[j], w_i)
                if score_child < score_cur:
                    # Child is better for this subproblem's weight
                    if best_replace == -1 or score_child < best_score:
                        best_replace = j
                        best_score = score_child

            # If child is better than the current solution at i, replace it
            score_i_cur = tchebycheff_val(F[i], w_i)
            score_i_child = tchebycheff_val(fc, w_i)
            if score_i_child < score_i_cur:
                if best_replace == -1 or score_i_child < best_score:
                    best_replace = i
                    best_score = score_i_child

            # Replace the worst solution
            if best_replace >= 0:
                X[best_replace] = child
                F[best_replace] = fc
            else:
                # Even if not better, keep the child in a global archive
                # Actually, let's just discard if not useful
                pass

        # Return the final population (evaluator records all returned objective vectors)
        # We should return all unique non-dominated points we've found
        # For simplicity, return the current population
        return X
# EVOLVE-BLOCK-END
