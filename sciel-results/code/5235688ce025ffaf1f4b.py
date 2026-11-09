import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    n = 8
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def score(points):
        # points: (n, 2)
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D < 1e-12:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts

    best_score = score(regular_octagon())
    best_points = regular_octagon().copy()

    # Objective: maximize product of distances, equivalently maximize sum of log distances
    # subject to all pairwise distances <= 2 (diameter constraint)
    # We'll minimize negative log product with penalty for diameter violation

    def objective_and_penalty(x, penalty_weight=1000.0):
        """x is flat array of length 16 (8 points x 2 coords)"""
        pts = x.reshape(n, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        unique_dists = dists[mask]
        
        # Penalty for coincident points
        min_d = np.min(unique_dists)
        if min_d < 1e-4:
            return 1e10
        
        log_sum = np.sum(np.log(unique_dists))
        
        # Diameter penalty
        D = np.max(dists)
        penalty = 0.0
        if D > 2.0:
            penalty = penalty_weight * (D - 2.0)**2
        
        return -log_sum + penalty

    # Use differential evolution in full 16D space with reasonable bounds
    # Points should be within [-2, 2] to keep diameter <= 2
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    # Phase 1: Differential Evolution for global exploration
    if time.time() < deadline - 40:
        try:
            de_res = differential_evolution(
                objective_and_penalty,
                bounds=bounds,
                maxiter=200,
                tol=1e-8,
                seed=42,
                popsize=20,
                mutation=(0.5, 1.5),
                recombination=0.8
            )
            x0 = de_res.x
            pts = x0.reshape(n, 2)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception as e:
            pass

    # Phase 2: Local refinement with SLSQP from multiple starting points
    # Starting points: DE result, regular octagon, and perturbed versions
    start_points = [best_points.copy()]
    
    # Add regular octagon
    start_points.append(regular_octagon())
    
    # Add perturbed regular octagons with various symmetry breakings
    rng = np.random.RandomState(42)
    for trial in range(10):
        if time.time() > deadline - 20:
            break
        base = regular_octagon()
        # Random small perturbations
        perturb = rng.normal(0, 0.1, size=(n, 2))
        candidate = base + perturb
        start_points.append(candidate)
    
    # Try SLSQP from each starting point
    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            def make_constraint(i, j):
                def constraint(x):
                    pts = x.reshape(n, 2)
                    d = np.linalg.norm(pts[i] - pts[j])
                    return 2.0 - d
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    bounds_slsqp = [(-2.0, 2.0)] * (2 * n)
    
    for sp in start_points:
        if time.time() > deadline - 5:
            break
        try:
            x0 = sp.flatten()
            res = minimize(
                objective_and_penalty,
                x0,
                method='SLSQP',
                bounds=bounds_slsqp,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
            if res.success:
                pts = res.x.reshape(n, 2)
                # Verify diameter constraint
                diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                if np.max(dists) <= 2.0 + 1e-6:
                    s = score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
        except Exception:
            pass

    # Phase 3: Try trust-constr for potentially better convergence
    if time.time() < deadline - 10:
        from scipy.optimize import LinearConstraint, NonlinearConstraint, trust_constr
        
        # Define diameter constraints as nonlinear constraints
        # g(x) = 2 - ||p_i - p_j|| >= 0 for all i < j
        def diameter_constraints(x):
            pts = x.reshape(n, 2)
            vals = []
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pts[i] - pts[j])
                    vals.append(2.0 - d)
            return np.array(vals)
        
        # Bounds
        lb = np.full(2*n, -2.0)
        ub = np.full(2*n, 2.0)
        
        # Start from current best
        x0 = best_points.flatten()
        
        try:
            res = trust_constr(
                lambda x: -objective_and_penalty(x, penalty_weight=0.0),  # Just negative log sum, no penalty
                x0,
                bounds=list(zip(lb, ub)),
                constraints=[NonlinearConstraint(diameter_constraints, 0, np.inf)],
                method='trust-constr',
                options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-10}
            )
            if res.success:
                pts = res.x.reshape(n, 2)
                diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                if np.max(dists) <= 2.0 + 1e-6:
                    s = score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
        except Exception:
            pass

    return best_points.tolist()
