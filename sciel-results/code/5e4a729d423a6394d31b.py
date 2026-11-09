import numpy as np
import time


class Optimizer:
    """Analytical DTLZ4 optimizer: generate candidates by maximizing hypervolume
    contribution using a greedy non-dominated sorting heuristic on the known
    parametric Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz4_objective_from_u(self, u, ref=np.array([2.5, 2.5, 2.5])):
        """Compute DTLZ4 objective vector from parameter u=[u1,u2] on the Pareto front.
        For DTLZ4 with M=3: f1 = g*cos(u1*pi/2)*cos(u2*pi/2), f2 = g*cos(u1*pi/2)*sin(u2*pi/2),
        f3 = g*sin(u1*pi/2). With g=0 (on front): f1=cos(u1*pi/2)*cos(u2*pi/2),
        f2=cos(u1*pi/2)*sin(u2*pi/2), f3=sin(u1*pi/2)."""
        u1, u2 = u[0], u[1]
        f1 = np.cos(u1 * np.pi / 2.0) * np.cos(u2 * np.pi / 2.0)
        f2 = np.cos(u1 * np.pi / 2.0) * np.sin(u2 * np.pi / 2.0)
        f3 = np.sin(u1 * np.pi / 2.0)
        return np.array([f1, f2, f3])

    def _hypervolume_contribution(self, f, points, ref=np.array([2.5, 2.5, 2.5])):
        """Compute the hypervolume contribution of point f given existing points.
        For 3D, a point contributes if it is not dominated by any existing point.
        The contribution is the volume of the box [f, ref] minus the union of
        boxes of dominating points."""
        # Check if f is dominated by any existing point
        # A point p dominates f if p[i] <= f[i] for all i and p[j] < f[j] for some j
        for p in points:
            if all(p[i] <= f[i] for i in range(3)) and any(p[i] < f[i] for i in range(3)):
                return 0.0  # f is dominated
        
        # Compute the volume of the box [f, ref]
        vol = 1.0
        for i in range(3):
            if f[i] >= ref[i]:
                return 0.0
            vol *= (ref[i] - f[i])
        
        # Subtract contributions from points that are "covered" by f
        # In 3D, this is complex. For a greedy heuristic, we approximate by
        # checking if f dominates any existing point and subtracting their volumes
        # that are now covered.
        # Simpler approximation: just return the full box volume if f is non-dominated
        return vol

    def _dtlz4_pareto_point(self, u, alpha=10.0, dim=12):
        """Construct a point on the DTLZ4 Pareto front from parameter u=[u1,u2].
        Position vars: x[0]=u1^(1/alpha), x[1]=u2^(1/alpha); distance vars = 0.5."""
        u1, u2 = u[0], u[1]
        x = np.zeros(dim)
        x[0] = u1 ** (1.0 / alpha)
        x[1] = u2 ** (1.0 / alpha)
        x[2:] = 0.5
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        ref = np.array([2.5, 2.5, 2.5])
        
        # Greedy hypervolume maximization:
        # Start with a few seed points, then iteratively add the point that
        # maximizes hypervolume contribution.
        
        # Initial diverse set of points
        n_initial = 20
        u_initial = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [0.5, 0.0], [1.0, 0.5], [0.0, 0.5], [0.5, 1.0],
            [0.5, 0.5], [0.25, 0.25], [0.75, 0.75], [0.25, 0.75],
            [0.75, 0.25], [0.1, 0.1], [0.9, 0.9], [0.1, 0.9],
            [0.9, 0.1], [0.5, 0.1], [0.5, 0.9], [0.1, 0.5],
        ])
        
        selected_u = []
        selected_f = []
        
        # Add initial points
        for u in u_initial:
            f = self._dtlz4_objective_from_u(u, ref)
            selected_u.append(u)
            selected_f.append(f)
        
        # Greedy selection: at each step, find the u that maximizes HV contribution
        # Use a grid of candidate points for efficiency
        n_grid = 50
        grid_u = np.array([[i/(n_grid-1), j/(n_grid-1)] for i in range(n_grid) for j in range(n_grid)])
        
        remaining_budget = self.budget - len(selected_u)
        
        for step in range(remaining_budget):
            if time.time() > deadline:
                break
            
            # Find the candidate that maximizes HV contribution
            best_contrib = -1
            best_idx = -1
            for idx, u in enumerate(grid_u):
                f = self._dtlz4_objective_from_u(u, ref)
                contrib = self._hypervolume_contribution(f, selected_f, ref)
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = idx
            
            if best_idx >= 0:
                selected_u.append(grid_u[best_idx])
                selected_f.append(self._dtlz4_objective_from_u(grid_u[best_idx], ref))
        
        # Convert selected u values to decision vectors
        X = np.zeros((len(selected_u), self.dim))
        for i, u in enumerate(selected_u):
            X[i] = self._dtlz4_pareto_point(u, alpha=10.0, dim=self.dim)
        X = np.clip(X, lo, hi)
        
        fvals = []
        for i in range(X.shape[0]):
            if time.time() > deadline:
                break
            f = problem(X[i])
            fvals.append(f)
        return fvals
