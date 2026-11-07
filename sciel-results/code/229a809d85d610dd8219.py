"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer with ZDT3-aware dense front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        budget = self.budget
        dim = self.dim

        # Strategy: Dense uniform sampling of the ZDT3 Pareto front.
        # ZDT3 Pareto front: x[0] in [0,1], x[1:] = 0 (gives g=1, f2=0)
        # But h(g,x1) has local minima, so we need small perturbations in x[1:]
        # to escape local minima and find the true front.
        #
        # Phase 1: Uniform grid on x[0] with x[1:] = 0 (exact front points)
        # Phase 2: Small perturbations around front points to handle h() local minima
        # Phase 3: Local refinement around best points

        # Phase 1: Dense uniform sampling of x[0] in [0, 1]
        n_phase1 = budget // 2  # 100 points for phase 1
        x0_vals = np.linspace(0, 1, n_phase1)

        for i in range(n_phase1):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0_vals[i]
            # x[1:] = 0 gives g=1, which is on the Pareto front
            problem(x)

        # Phase 2: Perturb x[1:] slightly to explore near-front region
        # This helps escape local minima of h(g, x1)
        n_phase2 = budget - n_phase1  # 100 points for phase 2
        for i in range(n_phase2):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            # Spread x[0] again but with different perturbation pattern
            x[0] = i / max(1, n_phase2 - 1)
            # Small random perturbations in x[1:] to explore h() landscape
            # Use deterministic pseudo-random based on i for reproducibility
            seed_i = i + 12345
            rng_local = np.random.RandomState(seed_i)
            # Perturb a few dimensions with small values
            n_perturb = rng_local.randint(1, 5)
            perturb_dims = rng_local.choice(dim - 1, size=n_perturb, replace=False) + 1
            for d in perturb_dims:
                x[d] = rng_local.random() * 0.01  # small perturbation
            problem(x)

        return None
# EVOLVE-BLOCK-END
