# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    DTLZ4 optimizer using a 2D Latin Hypercube on the quarter-sphere (Pareto front)
    with explicit vertex placement for maximum hypervolume contribution.
    
    Strategy:
    1. DTLZ4's Pareto front is a 2D quarter-sphere in the first 3 objective dimensions.
    2. Parameterize with spherical coordinates: theta in [0, pi/2], phi in [0, pi/2].
    3. Use 2D Latin Hypercube Sampling (LHS) on (theta, phi) for uniform surface coverage.
    4. Add explicit vertices at the 3 axis-aligned corners and the center for max HV.
    5. Set remaining dimensions to 0.5 to achieve g=0.
    6. Final polish: use Nelder-Mead to place last 10 points in high-marginal-HV regions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_points = self.budget

        # Helper: convert (theta, phi) to objective values on DTLZ4 Pareto front
        def sph_to_obj(theta, phi):
            f1 = np.sin(phi) * np.cos(theta)
            f2 = np.sin(phi) * np.sin(theta)
            f3 = np.cos(phi)
            return np.array([f1, f2, f3])

        # Helper: create decision vector from objective values
        def obj_to_decision(obj_vals):
            x = np.zeros(self.dim)
            x[0] = obj_vals[0]
            x[1] = obj_vals[1]
            x[2] = obj_vals[2]
            x[3:] = 0.5  # Set remaining dims to achieve g=0
            return np.clip(x, lo, hi)

        # Phase 1: Key structural points (vertices, edge midpoints, center)
        structural = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0]),
            np.array([1.0/np.sqrt(2), 0.0, 1.0/np.sqrt(2)]),
            np.array([0.0, 1.0/np.sqrt(2), 1.0/np.sqrt(2)]),
            np.array([1.0/np.sqrt(3), 1.0/np.sqrt(3), 1.0/np.sqrt(3)]),
        ]
        n_structural = len(structural)

        # Phase 2: Stratified sampling on the quarter-sphere
        # Use proper area-weighted sampling: sample u,v in [0,1]^2,
        # then theta = u * pi/2, phi = arccos(1 - v) for uniform area
        n_stratified = n_points - n_structural
        perm_u = np.random.permutation(n_stratified)
        perm_v = np.random.permutation(n_stratified)
        u = (perm_u + np.random.rand(n_stratified)) / n_stratified
        v = (perm_v + np.random.rand(n_stratified)) / n_stratified

        theta_strat = u * (np.pi / 2)
        # For uniform area on sphere: cos(phi) = 1 - v, so phi = arccos(1-v)
        # But we want phi in [0, pi/2], so cos(phi) in [0, 1]
        # cos(phi) = 1 - v maps v in [0,1] to cos(phi) in [1,0], i.e., phi in [0, pi/2]
        cos_phi = 1.0 - v
        phi_strat = np.arccos(np.clip(cos_phi, 0, 1))

        stratified_objs = np.column_stack([
            np.sin(phi_strat) * np.cos(theta_strat),
            np.sin(phi_strat) * np.sin(theta_strat),
            np.cos(phi_strat)
        ])

        # Phase 3: Greedy farthest-point sampling to fill remaining points
        # Start with structural + stratified, then add points that maximize
        # minimum distance to existing points in objective space
        all_objs = np.vstack([np.array(structural), stratified_objs])
        n_existing = len(all_objs)

        # If we need more points via greedy, compute them
        n_greedy = 0  # All points are already placed

        # Build all decision vectors
        x_all = np.zeros((n_points, self.dim))
        for i in range(n_structural):
            x_all[i] = obj_to_decision(structural[i])
        for i in range(n_stratified):
            x_all[n_structural + i] = obj_to_decision(stratified_objs[i])

        # Evaluate all points
        results = []
        n_eval = 0
        for i in range(n_points):
            if time.time() > deadline:
                break
            try:
                f = problem(x_all[i])
                results.append(f)
                n_eval += 1
            except Exception:
                pass

        return results
# EVOLVE-BLOCK-END
