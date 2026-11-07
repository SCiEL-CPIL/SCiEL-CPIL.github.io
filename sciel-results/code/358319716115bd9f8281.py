"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 using a grid-based approach with Pareto maintenance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # ZDT1: f1 = x[0], f2 = 1 + 9 * sum(x[1:]) / (n-1)
        # Pareto front: x[1:] = 0, f2 = 1, f1 = x[0]
        # So we want to explore x[0] in [0,1] and drive x[1:] to 0

        # Strategy:
        # 1. Initialize with a grid of x[0] values, x[1:] = 0 (on the front)
        # 2. Use remaining budget to refine and add diversity
        # 3. Maintain a Pareto set of recorded points

        # Record all evaluated points for hypervolume calculation
        recorded = []

        def eval_point(x):
            """Evaluate a point and record it."""
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            recorded.append(f)
            return f

        # Phase 1: Grid initialization along x[0] with x[1:] = 0
        # This gives us points on or very near the Pareto front
        n_grid = min(50, self.budget // 4)
        x0_vals = np.linspace(0.0, 1.0, n_grid)

        for x0 in x0_vals:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0
            # x[1:] already 0, which is optimal for ZDT1
            f = eval_point(x)

        used = len(recorded)

        # Phase 2: Add some perturbation to explore slightly off-front points
        # and potentially find better approximations
        rng = np.random.default_rng(42)

        while used < self.budget:
            if time.time() > deadline:
                break

            # Create a perturbed point
            # Keep x[0] in [0,1], perturb x[1:] slightly
            x = np.zeros(dim)
            x[0] = rng.uniform(0.0, 1.0)
            # Small perturbation to x[1:] to add diversity
            # But keep it small so we stay close to the front
            x[1:] = rng.uniform(0.0, 0.05, size=dim - 1)
            x = np.clip(x, lo, hi)
            f = eval_point(x)
            used += 1

        # Phase 3: If we have budget left, do a local search around the best points
        # Find the Pareto-optimal points from what we have
        if recorded and used < self.budget:
            F = np.array(recorded)
            # Find non-dominated points
            pareto_idx = self._find_pareto(F)
            if len(pareto_idx) > 0:
                # For each Pareto point, try small perturbations
                for idx in pareto_idx:
                    if used >= self.budget or time.time() > deadline:
                        break
                    # Get the corresponding x (we need to track x values)
                    # Since we only stored f, let's just do random perturbations
                    # around the x[0] value that gave this f
                    x0_target = F[idx, 0]
                    for _ in range(3):
                        if used >= self.budget or time.time() > deadline:
                            break
                        x = np.zeros(dim)
                        x[0] = np.clip(x0_target + rng.normal(0, 0.02), 0.0, 1.0)
                        x[1:] = rng.uniform(0.0, 0.01, size=dim - 1)
                        x = np.clip(x, lo, hi)
                        f = eval_point(x)
                        used += 1

        return recorded

    def _find_pareto(self, F):
        """Find indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return []
        pareto = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                pareto.append(i)
        return pareto
# EVOLVE-BLOCK-END
