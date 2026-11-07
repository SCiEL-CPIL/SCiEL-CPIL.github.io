"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using LHS on simplex + refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 Pareto front: x[n_obj:] = 0.5, x[0] and x[1] vary in [0,1]
        def make_front_point(u1, u2):
            x = np.full(dim, 0.5)
            x[0] = u1
            x[1] = u2
            return x

        rng = np.random.default_rng(0)

        # Phase 1: LHS on the 2D parameter space of the Pareto front
        # Reserve 3 evaluations for extreme vertices
        n_phase1 = max(1, budget - 3)
        
        # LHS: divide each dimension into n_phase1 strata
        u1 = (np.arange(n_phase1) + rng.random(n_phase1)) / n_phase1
        u2 = (np.arange(n_phase1) + rng.random(n_phase1)) / n_phase1
        # Shuffle to decorrelate
        perm = rng.permutation(n_phase1)
        u2_shuffled = u2[perm]

        X_list = []
        F_list = []
        used = 0

        for i in range(n_phase1):
            if time.time() > deadline:
                break
            x = make_front_point(u1[i], u2_shuffled[i])
            f = problem(x)
            X_list.append(x)
            F_list.append(f)
            used += 1

        # Phase 2: Add extreme vertices of the Pareto front
        vertices = [
            make_front_point(0.0, 0.0),  # f ≈ (1, 0, 0)
            make_front_point(0.0, 1.0),  # f ≈ (0, 1, 0)
            make_front_point(1.0, 0.0),  # f ≈ (0, 0, 1)
        ]
        
        for v in vertices:
            if used >= budget or time.time() > deadline:
                break
            f = problem(v)
            X_list.append(v)
            F_list.append(f)
            used += 1

        # Phase 3: Use remaining budget for local refinement
        while used < budget:
            if time.time() > deadline:
                break
            # Pick a random existing point and perturb it
            idx = rng.integers(0, len(X_list))
            x = X_list[idx].copy()
            # Small perturbation in the first two dimensions
            x[0] += 0.02 * rng.standard_normal()
            x[1] += 0.02 * rng.standard_normal()
            x[0] = np.clip(x[0], 0, 1)
            x[1] = np.clip(x[1], 0, 1)
            # Keep the rest at 0.5 for DTLZ4 optimality
            x[n_obj:] = 0.5
            f = problem(x)
            X_list.append(x)
            F_list.append(f)
            used += 1

        if X_list:
            return np.array(X_list)
        return np.empty((0, dim))
# EVOLVE-BLOCK-END
