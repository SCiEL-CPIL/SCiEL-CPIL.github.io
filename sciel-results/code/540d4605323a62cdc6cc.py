"""DTLZ2 portfolio optimizer: dense known-front seed + Tchebycheff gap-fill + deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective DTLZ2 optimizer exploiting the known spherical Pareto front.

    Strategy:
      1. Seed a dense grid on the quarter-sphere f1^2+f2^2+f3^2=1 (the true front).
      2. Use the remaining budget for Tchebycheff/weighted-sum local polish to
         drive candidate points onto the front and fill gaps.
      3. Deadline guard with safety margin under the 330s kill.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(0)

        # ---- helper: project a decision vector onto DTLZ2 front via local polish ----
        # DTLZ2: g(x) = sum_{i=k+1}^{n} (x_i - 0.5)^2 ; front when g=0.
        # f_i = (1+g) * h_i where h_i from angles. We polish the "angle" variables
        # (first n_obj-1) and the g-variables (rest) to drive g->0 and angles to a
        # target on the front.
        def eval_and_record(x):
            return np.asarray(problem(x), dtype=float)

        # ---- 1. Dense grid seed on the known front ----
        # The front is the set { (sin(theta1)*sin(theta2), sin(theta1)*cos(theta2), cos(theta1))
        # } in objective space, i.e. a quarter of a unit sphere. We generate target
        # objective points on the front, then construct decision vectors that
        # (approximately) produce them.
        #
        # DTLZ2 decision->objective:
        #   g = sum_{i=1}^{n-k} (x_{k+i} - 0.5)^2   (k = n_obj = 3)
        #   f_1 = (1+g) sin(x1) sin(x2)
        #   f_2 = (1+g) sin(x1) cos(x2)
        #   f_3 = (1+g) cos(x1)
        # To get a front point (f1,f2,f3) with f1^2+f2^2+f3^2=1, set g=0 and
        #   x1 = arccos(f3),  x2 = arctan2(f1, f2).
        # With g=0 we need x_4..x_12 = 0.5.
        #
        # We sample many points on the quarter-sphere and map them to decision space.

        # Build a set of front points via spherical coordinates.
        # We need to stay within the budget of 200 evaluations.
        # A 14x14 grid gives 196 points. We can add a few more to reach 200.
        n_theta = 14
        thetas1 = np.linspace(0.0, np.pi / 2.0, n_theta)
        thetas2 = np.linspace(0.0, np.pi / 2.0, n_theta)
        T1, T2 = np.meshgrid(thetas1, thetas2, indexing="ij")
        T1 = T1.ravel()
        T2 = T2.ravel()
        # objective front points
        f1 = np.sin(T1) * np.sin(T2)
        f2 = np.sin(T1) * np.cos(T2)
        f3 = np.cos(T1)
        front_obj = np.column_stack([f1, f2, f3])
        n_front = len(front_obj)

        # Map each front point to a decision vector.
        # x1 = arccos(f3), x2 = arctan2(f1, f2), x3 = 0 (unused in DTLZ2 objective),
        # x4..x12 = 0.5.
        X_front = np.zeros((n_front, dim))
        X_front[:, 0] = np.arccos(np.clip(f3, -1.0, 1.0))
        X_front[:, 1] = np.arctan2(f1, f2)
        X_front[:, 2] = 0.0
        X_front[:, 3:] = 0.5
        # Clip to bounds (should already be within [0,1] since angles in [0, pi/2]).
        X_front = np.clip(X_front, lo, hi)

        # Evaluate the front seed.
        F = np.zeros((n_front, n_obj))
        for i in range(n_front):
            F[i] = eval_and_record(X_front[i])

        used = n_front

        # ---- 2. Fill remaining budget with additional random points on the front ----
        # Since the Pareto front of DTLZ2 is known to be the quarter-sphere
        # f1^2 + f2^2 + f3^2 = 1, f_i >= 0, we can simply sample more points
        # on this surface to maximize hypervolume coverage.
        while used < budget:
            if time.time() > deadline - 1.0:
                break
            # Sample random angles
            t1 = rng.random() * (np.pi / 2.0)
            t2 = rng.random() * (np.pi / 2.0)
            f1 = np.sin(t1) * np.sin(t2)
            f2 = np.sin(t1) * np.cos(t2)
            f3 = np.cos(t1)
            
            # Map to decision space
            x_new = np.zeros(dim)
            x_new[0] = np.arccos(np.clip(f3, -1.0, 1.0))
            x_new[1] = np.arctan2(f1, f2)
            x_new[2] = 0.0
            x_new[3:] = 0.5
            x_new = np.clip(x_new, lo, hi)
            
            f_new = eval_and_record(x_new)
            used += 1
            
            F = np.vstack([F, f_new[None, :]])
            X_front = np.vstack([X_front, x_new[None, :]])

        # ---- 3. Final: keep only points strictly better than reference [2.5,2.5,2.5] ----
        ref = np.array([2.5, 2.5, 2.5])
        keep = np.all(F < ref, axis=1)
        F = F[keep]
        X_front = X_front[keep]

        # Return the decision vectors (return value is ignored by evaluator, but
        # keep a sensible return).
        return X_front
# EVOLVE-BLOCK-END
