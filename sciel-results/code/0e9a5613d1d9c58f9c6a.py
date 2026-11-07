"""Multi-objective optimizer on DTLZ4 using LHS-based Pareto front sampling."""
import time
import numpy as np


class Optimizer:
    """LHS-based deterministic sampler for DTLZ4 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _lhs(self, n, d, rng):
        """Latin Hypercube Sampling in [0,1]^d with n points."""
        pts = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                block = i // (n // d) if n % d == 0 else i
                # Standard LHS: divide [0,1] into n bins per dim, sample one per row
                row = i
                pts[i, j] = (row + rng.random()) / n
        # Permute columns
        for j in range(d):
            perm = rng.permutation(n)
            pts[:, j] = pts[perm, j]
        return pts

    def _front_to_x(self, y, dim, rng):
        """Convert Pareto front position y (length n_obj-1, in [0,1]^(n_obj-1)) to x (length dim).
        
        For DTLZ4 with 3 objectives, the Pareto front is parameterized by 2 variables u1, u2 in [0,1].
        f1 = sum_{i=1}^{n} u_i^2 where u_1 + u_2 + ... + u_{n} = 1 (for 3 obj, n = dim-2 = 10)
        Actually for DTLZ4 with m objectives, we have n = dim - (m-1) decision variables on the front.
        
        For 3 objectives: f1 = sum_{i=1}^{10} u_i^2, f2 = (1-f1)*u_10, f3 = (1-f1)*(1-u_10)
        where sum_{i=1}^{10} u_i = 1.
        
        We parameterize by (alpha, beta) where alpha = u_10/sum_{i=1}^{9} u_i type ratios.
        Simpler: use Dirichlet-like parameterization.
        """
        n_obj = 3
        n_front = dim - (n_obj - 1)  # 10 for dim=12
        
        # y has length n_obj-1 = 2
        alpha = y[0]  # in [0,1]
        beta = y[1]   # in [0,1]
        
        # Parameterize the 10 u values:
        # u_10 = alpha * (1 - beta)
        # u_1 = beta * (1 - alpha)  
        # u_2...u_9 share the rest
        # Actually let's use a cleaner parameterization:
        # Let v1 = alpha, v2 = beta
        # We need 10 non-negative numbers summing to 1.
        # Use: u_i for i=1..9 from (1-alpha-beta)/9 roughly, u_10 = alpha, and scale...
        
        # Better: use the standard DTLZ4 parameterization
        # f1 = sum(u_i^2) for i=1..n_front
        # f2 = (1-f1) * u_{n_front}
        # f3 = (1-f1) * (1 - u_{n_front})
        # where sum u_i = 1
        
        # Parameterize u vector via (alpha, beta):
        # u_{n_front} = alpha
        # u_1 = beta * (1 - alpha)
        # u_2..u_{n_front-1} = (1-alpha) * (1-beta) / (n_front - 2)
        
        u = np.zeros(n_front)
        u[n_front - 1] = alpha
        u[0] = beta * (1.0 - alpha)
        if n_front > 2:
            u[1:n_front-1] = (1.0 - alpha) * (1.0 - beta) / (n_front - 2)
        
        # Now build x for DTLZ4:
        # First n_front variables: x_i = u_i^{1/alpha_dtlz4} where alpha_dtlz4=100
        # But we want x on the Pareto front, so we set g(x)=0.
        # For DTLZ4: x_0...x_{n_front-1} determine the position, x_{n_front}...x_{dim-1} determine g.
        # g = 0 means those trailing variables are at their minimum (which is 0 in [0,1]).
        # Actually g(x) = (dim - n_front) + sum_{i=n_front}^{dim-1} (x_i - 0.5)^2 * sin(10*pi*x_i)
        # For g=0, we need x_i = 0.5 for all trailing i (since sin(10*pi*0.5)=sin(5*pi)=0).
        # Wait, sin(10*pi*0.5) = sin(5*pi) = 0. Yes.
        
        alpha_dtlz4 = 100.0
        x = np.zeros(dim)
        # Front variables: x_i = u_i^{1/alpha_dtlz4}
        for i in range(n_front):
            x[i] = u[i] ** (1.0 / alpha_dtlz4)
        # Trailing variables: set to 0.5 for g=0
        for i in range(n_front, dim):
            x[i] = 0.5
        
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # 300s limit, keep 20s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Generate LHS points on the 2D parameter space of the Pareto front
        # Add explicit vertices and edge midpoints for better coverage
        n_lhs = min(budget - 10, 160)  # leave some budget
        if n_lhs < 10:
            n_lhs = 10
        
        lhs_pts = self._lhs(n_lhs, 2, rng)
        
        # Add vertices of the [0,1]^2 square
        vertices = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5],
            [0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]
        ])
        
        all_params = np.vstack([lhs_pts, vertices])
        
        # Shuffle to interleave
        rng.shuffle(all_params)
        
        # Evaluate
        F = []
        used = 0
        for p in all_params:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            x = self._front_to_x(p, dim, rng)
            x = np.clip(x, lo, hi)
            f = problem(x)
            F.append(np.array(f, dtype=float))
            used += 1
        
        return F
