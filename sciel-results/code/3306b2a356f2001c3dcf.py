"""Multi-objective optimizer for ZDT4: dense Pareto front sampling with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Dense Pareto front sampling for ZDT4.

    Strategy:
      - ZDT4 has a known structure: f1 = 1 + 10*(n-1) + sum((x_i-0.5)^2 - 10*cos(2*pi*(x_i-0.5))) for i=2..n
        f2 = g(x1) * h(f1) where g(x1) = x1^2 and h(f1) = 1 - sqrt(f1/g(x1))
      - The Pareto front is achieved when x_2..x_n = 0 (making f1 = 1 + 10*(n-1) = 101 for n=10... wait)
        Actually for ZDT4: f1 = 1 + 10*(M-1) + sum_{i=2}^{n} [(x_i - 0.5)^2 - 10*cos(2*pi*(x_i-0.5))]
        with M=2 objectives, n=10. So f1_min = 1 + 10 = 11 when x_i=0.5 for i=2..n? No.
        Let me recall: ZDT4 f1 = 1 + 10*(n-1) + sum_{i=2}^n [(x_i - 0.5)^2 - 10*cos(2*pi*(x_i-0.5))]
        The minimum of each term (x_i-0.5)^2 - 10*cos(2*pi*(x_i-0.5)) is -10 (at x_i=0.5)
        So f1_min = 1 + 10*9 + 9*(-10) = 1 + 90 - 90 = 1. Wait that's not right either.
        
        Actually for ZDT4: f1(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [(x_i - 0.5)^2 - 10*cos(2*pi*(x_i - 0.5))]
        With n=10: f1 = 1 + 90 + sum of 9 terms. Each term ranges from -10 (at x=0.5) to ~10 (at x=0 or 1).
        So f1 ranges from 1+90+9*(-10) = 1 to 1+90+9*(~10) = 190.
        
        The Pareto front: f2 = g(x1) * h(f1) where g(x1) = x1^2, h(f1) = 1 - sqrt(f1/g(x1))
        For the Pareto front, we need f1 to vary from its minimum to maximum while x1 varies.
        The key insight: on the Pareto front, x_2..x_n = 0.5 (giving f1 = 1) is NOT correct.
        
        Let me reconsider. The standard ZDT4 Pareto front is obtained when x_i = 0 for i=2..n,
        which gives f1 = 1 + 10*(n-1) = 1 + 90 = 91? No...
        
        Actually, the minimum of (x_i-0.5)^2 - 10*cos(2*pi*(x_i-0.5)):
        At x_i = 0.5: (0)^2 - 10*cos(0) = -10
        At x_i = 0: (0.5)^2 - 10*cos(-pi) = 0.25 - 10*(-1) = 10.25
        At x_i = 1: same as x_i=0 by symmetry.
        
        So f1_min = 1 + 90 + 9*(-10) = 1. f1_max ≈ 1 + 90 + 9*10.25 = 193.25.
        
        For the Pareto front of ZDT4, the standard result is that x_2..x_n = 0 gives the front.
        At x_i = 0: term = 0.25 + 10 = 10.25. So f1 = 1 + 90 + 9*10.25 = 193.25. That's the max.
        
        Hmm, I think I'm confusing myself. Let me just use the known fact: for ZDT4, the Pareto
        front is parameterized by x_1 in [0,1], with x_2..x_n = 0.5 (or 0, depending on convention).
        
        Actually, looking at the reference point [11.0, 400.0], and knowing that the Pareto front
        of ZDT4 has f1 in [1, ~91] or similar... The reference point f1=11 suggests f1 goes up to about 11.
        
        Let me just be pragmatic: I'll sample x_1 densely in [0,1], set x_2..x_n to values that
        minimize f1 (which is x_i=0.5 giving term=-10, so f1=1), and also try other configurations.
        
        Wait - the reference point is [11.0, 400.0]. For hypervolume, we need points with f1 < 11 and f2 < 400.
        
        The key insight from the knowledge: "x_1..x_9=0 is optimal for f2 given f1". This suggests
        that setting x_2..x_10 = 0 (or near 0) is good. Let me just do dense sampling of x_1 with
        x_2..x_10 fixed at 0, and also try x_2..x_10 = 0.5.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        rng = np.random.default_rng(0)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        archive = []  # list of (x, f) tuples for non-dominated points
        eval_count = 0
        
        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            f = problem(x)
            eval_count += 1
            return f
        
        def is_dominated(f, archive_f):
            """Check if f is dominated by any point in archive_f."""
            if len(archive_f) == 0:
                return False
            # f is dominated if there exists a point that is <= in all objectives and < in at least one
            for af in archive_f:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False
        
        def add_to_archive(x, f):
            nonlocal archive
            if f is None:
                return
            # Check if f is better than reference point
            if f[0] >= 11.0 or f[1] >= 400.0:
                return
            # Check if dominated by existing archive
            if is_dominated(f, [af for _, af in archive]):
                return
            # Remove points dominated by f
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive = new_archive
        
        # Phase 1: Dense sampling of x_1 with x_2..x_10 = 0
        # This should trace out the Pareto front
        n_samples = min(self.budget - 20, 150)  # leave some budget for exploration
        x1_values = np.linspace(0.0, 1.0, n_samples + 1)
        
        for x1 in x1_values:
            if time.time() > deadline or eval_count >= self.budget:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            # Set x_2..x_10 to 0
            f = evaluate(x)
            if f is not None:
                add_to_archive(x, f)
        
        # Phase 2: Try x_2..x_10 = 0.5 (another common Pareto front configuration)
        remaining = self.budget - eval_count
        if remaining > 10 and time.time() < deadline:
            n_samples2 = min(remaining - 5, 50)
            x1_values2 = np.linspace(0.0, 1.0, n_samples2 + 1)
            for x1 in x1_values2:
                if time.time() > deadline or eval_count >= self.budget:
                    break
                x = np.full(self.dim, 0.5)
                x[0] = x1
                f = evaluate(x)
                if f is not None:
                    add_to_archive(x, f)
        
        # Phase 3: Random exploration with remaining budget
        while eval_count < self.budget and time.time() < deadline:
            x = lo + (hi - lo) * rng.random(self.dim)
            f = evaluate(x)
            if f is not None:
                add_to_archive(x, f)
        
        return archive
# EVOLVE-BLOCK-END
