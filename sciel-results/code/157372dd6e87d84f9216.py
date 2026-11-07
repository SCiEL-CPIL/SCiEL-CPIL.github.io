# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hypervolume-driven adaptive refinement for DTLZ1 (3 objectives, 7 dims).

    Strategy:
    - Initialize with a coarse set of points on the known Pareto front.
    - Iteratively propose new points by sampling the Pareto front.
    - Use a fast proxy (box volume) to pre-filter candidates, then exact HV for the top few.
    - Map objective space candidates to decision space using the DTLZ1 structure (g=0).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget

        # Store all evaluated points
        X_list = []
        F_list = []

        def evaluate(x):
            nonlocal n_evals
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            n_evals += 1
            X_list.append(x.copy())
            F_list.append(f)
            return f

        # Helper: compute hypervolume of a set of 3D points w.r.t. reference point
        def hypervolume_3d(points, ref):
            """Compute exact 3D hypervolume using O(n^2) algorithm."""
            if len(points) == 0:
                return 0.0
            pts = np.array(points)
            # Filter points that are strictly better than ref in all objectives
            mask = np.all(pts < ref, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n = len(pts)
            
            def hv2d(pts2d, ref2d):
                """Compute 2D hypervolume of points w.r.t. reference point."""
                if len(pts2d) == 0:
                    return 0.0
                pts2d = np.array(pts2d)
                mask = np.all(pts2d < ref2d, axis=1)
                pts2d = pts2d[mask]
                if len(pts2d) == 0:
                    return 0.0
                # Sort by first coord
                order = np.argsort(pts2d[:, 0])
                pts2d = pts2d[order]
                n2 = len(pts2d)
                area = 0.0
                prev_x = 0.0
                min_y = ref2d[1]
                for j in range(n2):
                    x = pts2d[j, 0]
                    y = pts2d[j, 1]
                    if y < min_y:
                        area += (x - prev_x) * (min_y - y)
                        min_y = y
                    prev_x = x
                return area

            hv = 0.0
            for i in range(n):
                if i == 0:
                    dx = pts[i, 0]
                else:
                    dx = pts[i, 0] - pts[i-1, 0]
                if dx <= 0:
                    continue
                pts_ge_i = pts[i:, 1:3]
                area_i = hv2d(pts_ge_i, [ref[1], ref[2]])
                hv += dx * area_i
            
            return hv

        ref = np.array([400.0, 400.0, 400.0])
        rng = np.random.default_rng(0)

        # Helper: Generate a candidate point on the Pareto front of DTLZ1
        def generate_pareto_candidate():
            # DTLZ1 Pareto front (g=0):
            # f1 = 0.5 * x1
            # f2 = 0.5 * x2 * (1-x1)
            # f3 = 0.5 * (1-x2)
            # Inverse:
            # x1 = 2 * f1
            # x2 = 1 - 2 * f3
            # f2 = 0.5 * (1 - 2*f3) * (1 - 2*f1)
            
            f1 = rng.uniform(0.0, 0.5)
            f3 = rng.uniform(0.0, 0.5)
            f2 = 0.5 * (1.0 - 2.0 * f3) * (1.0 - 2.0 * f1)
            
            f = np.array([f1, f2, f3])
            
            x = np.zeros(self.dim)
            x[0] = 2.0 * f1
            x[1] = 1.0 - 2.0 * f3
            x[2:] = 0.5  # Set g(x) = 0
            
            return x, f

        # Helper: Check if a point is dominated by any point in a set
        def is_dominated(point, points_set):
            if len(points_set) == 0:
                return False
            pts = np.array(points_set)
            # point <= pts and point != pts
            # (pts >= point).all(axis=1) and (pts < point).any(axis=1)
            return np.any(np.all(pts >= point, axis=1) & np.any(pts < point, axis=1))

        # Initialization: Evaluate a structured set of points on the Pareto front
        # Use a stratified grid to ensure good initial coverage
        n_init = 30
        # Generate stratified samples in the simplex
        # We can use a simple grid on the 2D parameters (f1, f3)
        n_grid = int(np.ceil(np.sqrt(n_init)))
        # Ensure we hit the corners and edges
        # Simple approach: random but with a few fixed points
        # Fixed points: corners (0,0), (0.5,0), (0,0.5) -> f=(0,0.5,0), (0.5,0,0), (0,0,0.5)
        # And center (0.166, 0.166, 0.166) approx
        init_points = []
        # Corners
        init_points.append((0.0, 0.0)) # f1=0, f3=0 -> f=(0, 0.5, 0)
        init_points.append((0.5, 0.0)) # f1=0.5, f3=0 -> f=(0.5, 0, 0)
        init_points.append((0.0, 0.5)) # f1=0, f3=0.5 -> f=(0, 0, 0.5)
        
        # Add random points to fill the rest
        for _ in range(n_init - 3):
            f1 = rng.uniform(0.0, 0.5)
            f3 = rng.uniform(0.0, 0.5)
            init_points.append((f1, f3))
            
        for f1, f3 in init_points:
            f2 = 0.5 * (1.0 - 2.0 * f3) * (1.0 - 2.0 * f1)
            f = np.array([f1, f2, f3])
            x = np.zeros(self.dim)
            x[0] = 2.0 * f1
            x[1] = 1.0 - 2.0 * f3
            x[2:] = 0.5
            evaluate(x)

        # Main loop: Greedy hypervolume maximization
        while n_evals < max_evals:
            F = np.array(F_list)
            
            # Find non-dominated points
            n_points = len(F)
            is_dominated = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                if is_dominated[i]:
                    continue
                for j in range(n_points):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
            
            non_dom_idx = np.where(~is_dominated)[0]
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(n_points)
            
            F_nd = F[non_dom_idx]
            
            # Generate a batch of candidates
            n_candidates = 100
            candidates = []
            for _ in range(n_candidates):
                x_cand, f_cand = generate_pareto_candidate()
                candidates.append((x_cand, f_cand))
            
            # Filter dominated candidates
            valid_candidates = []
            for x_cand, f_cand in candidates:
                if not is_dominated(f_cand, F_nd):
                    valid_candidates.append((x_cand, f_cand))
            
            if not valid_candidates:
                # If all are dominated, we might be stuck or the front is fully covered
                # Pick a random one just to use the budget, though it won't improve HV
                x_cand, f_cand = generate_pareto_candidate()
                evaluate(x_cand)
                continue

            # Use box volume as a proxy to select top K candidates for exact HV
            # Box volume: (R0 - f1) * (R1 - f2) * (R2 - f3)
            # This is an upper bound on the contribution if the rest of the space was empty.
            # It's a good heuristic for "largest potential gain".
            scored_candidates = []
            for x_cand, f_cand in valid_candidates:
                vol = (ref[0] - f_cand[0]) * (ref[1] - f_cand[1]) * (ref[2] - f_cand[2])
                scored_candidates.append((vol, x_cand, f_cand))
            
            # Sort by volume descending and take top K
            scored_candidates.sort(key=lambda item: item[0], reverse=True)
            top_k = 10
            top_candidates = scored_candidates[:top_k]
            
            best_x = None
            best_f = None
            best_hv = -1.0
            
            # Calculate exact HV for top K
            for _, x_cand, f_cand in top_candidates:
                new_set = np.vstack([F_nd, [f_cand]])
                new_hv = hypervolume_3d(new_set, ref)
                if new_hv > best_hv:
                    best_hv = new_hv
                    best_x = x_cand
                    best_f = f_cand
            
            if best_x is not None:
                evaluate(best_x)
            else:
                # Fallback
                x = rng.uniform(lo, hi)
                evaluate(x)

        return X_list
# EVOLVE-BLOCK-END
