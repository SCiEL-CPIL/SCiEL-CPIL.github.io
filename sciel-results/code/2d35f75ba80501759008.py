"""Multi-objective optimizer for WFG7 using MOEA/D with Tchebycheff decomposition."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """MOEA/D with Tchebycheff decomposition and adaptive local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 270.0
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Number of subproblems (weight vectors)
        N = 30
        # Neighborhood size
        T = 5
        # Local search probability
        P_local = 0.3
        # Max local search steps per subproblem
        max_ls_steps = 3
        # Step size for local search
        ls_step = 0.1

        # Generate weight vectors using simplex lattice
        weights = self._generate_weights(N, n_obj)
        
        # Initialize population
        X = lo + span * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        # Ideal point (minimum of each objective)
        ideal = np.min(F, axis=0)

        # Build neighborhood
        def build_neighborhood(weights, T):
            """Build neighborhood list for each weight vector."""
            n = len(weights)
            # Compute distances between weight vectors
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dists[i, j] = np.linalg.norm(weights[i] - weights[j])
            # For each i, find T nearest neighbors
            neighbors = []
            for i in range(n):
                order = np.argsort(dists[i])
                neighbors.append(order[1:T+1].tolist())
            return neighbors

        neighbors = build_neighborhood(weights, T)

        def tchebycheff_fitness(f, w, ideal):
            """Tchebycheff scalarization."""
            return np.max(w * (f - ideal))

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Local search using small perturbations
        def local_search(x, f, w, ideal, max_steps):
            """Simple local search with small random perturbations."""
            best_x = x.copy()
            best_f = f.copy()
            best_tch = tchebycheff_fitness(best_f, w, ideal)
            for _ in range(max_steps):
                if used >= budget:
                    break
                # Random perturbation
                step = rng.random(dim) * 2 - 1
                step *= ls_step * span
                trial = np.clip(best_x + step, lo, hi)
                trial_f = problem(trial)
                used += 1
                trial_tch = tchebycheff_fitness(trial_f, w, ideal)
                if trial_tch < best_tch:
                    best_x = trial
                    best_f = trial_f
                    best_tch = trial_tch
            return best_x, best_f

        # Main loop
        while used < budget:
            if time.perf_counter() > deadline:
                break

            for i in range(N):
                if used >= budget:
                    break
                if time.perf_counter() > deadline:
                    break

                # Select neighborhood
                neigh = neighbors[i]
                if rng.random() < 0.9:
                    # Select from neighborhood
                    j = neigh[rng.integers(len(neigh))]
                else:
                    # Select randomly
                    j = rng.integers(N)

                if j == i:
                    j = (i + 1) % N

                # Create trial solution
                if rng.random() < P_local:
                    # Local search
                    trial_x, trial_f = local_search(X[i], F[i], weights[i], ideal, max_ls_steps)
                else:
                    # Crossover: blend of two parents
                    alpha = rng.random()
                    trial_x = alpha * X[i] + (1 - alpha) * X[j]
                    trial_x = np.clip(trial_x, lo, hi)
                    # Mutation
                    if rng.random() < 0.3:
                        mutation = (rng.random(dim) * 2 - 1) * 0.1 * span
                        trial_x = np.clip(trial_x + mutation, lo, hi)
                    trial_f = problem(trial_x)
                    used += 1

                # Update: replace if better Tchebycheff value
                tch_old = tchebycheff_fitness(F[i], weights[i], ideal)
                tch_new = tchebycheff_fitness(trial_f, weights[i], ideal)
                
                if tch_new < tch_old:
                    X[i] = trial_x
                    F[i] = trial_f
                    # Update ideal
                    ideal = np.minimum(ideal, trial_f)
                elif not dominates(F[i], trial_f):
                    # Keep if non-dominated
                    if rng.random() < 0.5:
                        X[i] = trial_x
                        F[i] = trial_f
                        ideal = np.minimum(ideal, trial_f)

            # Periodic diversity injection
            if used < budget and (used - N) % N == 0:
                # Replace worst Tchebycheff point
                tch_vals = np.array([tchebycheff_fitness(F[i], weights[i], ideal) for i in range(N)])
                worst = int(np.argmax(tch_vals))
                X[worst] = lo + span * rng.random(dim)
                F[worst] = problem(X[worst])
                used += 1
                ideal = np.minimum(ideal, F[worst])

        return X

    def _generate_weights(self, N, n_obj):
        """Generate N weight vectors on the simplex."""
        weights = []
        # Use Dirichlet distribution for uniform sampling on simplex
        for _ in range(N):
            w = np.random.dirichlet(np.ones(n_obj))
            weights.append(w)
        return np.array(weights)
# EVOLVE-BLOCK-END
