import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        n_evals = 0
        all_X = []
        all_F = []
        
        def evaluate(x):
            nonlocal n_evals
            fx = problem(x)
            n_evals += 1
            return fx

        def hypervolume_3d(points, ref):
            """Compute hypervolume for 3D points dominated by ref."""
            # Filter points that are strictly better than ref in all objectives
            valid = [p for p in points if all(p[i] < ref[i] for i in range(3))]
            if not valid:
                return 0.0
            
            # Sort by first objective
            valid.sort(key=lambda p: p[0])
            
            # For 3D, use inclusion-exclusion with sorted lists
            # Use a sweep line approach
            n = len(valid)
            if n == 0:
                return 0.0
            if n == 1:
                p = valid[0]
                return max(0, ref[0] - p[0]) * max(0, ref[1] - p[1]) * max(0, ref[2] - p[2])
            
            # Sort by x, then process
            valid.sort(key=lambda p: (p[0], p[1], p[2]))
            
            # For each point, compute the volume it uniquely contributes
            # This is complex for 3D; use a simpler approximation with sorting
            # Actually, let's use a proper 3D hypervolume algorithm
            
            # Sort by first coordinate
            pts = sorted(valid, key=lambda p: p[0])
            
            hv = 0.0
            # For 3D, we can use the following:
            # Sort by x. For each point, the contribution is (x_next - x_i) * area_in_2d(y,z)
            # where area_in_2d is the area of the union of rectangles from points with x >= x_i
            
            # This is O(n^2) which is fine for n <= 200
            for i in range(n):
                x_i = pts[i][0]
                x_next = pts[i+1][0] if i + 1 < n else ref[0]
                if x_next <= x_i:
                    continue
                # Compute 2D hypervolume of points i..n-1 in y,z plane
                sub_pts = [(p[1], p[2]) for p in pts[i:]]
                hv += (x_next - x_i) * hypervolume_2d(sub_pts, [ref[1], ref[2]])
            
            return hv
        
        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume."""
            if not points:
                return 0.0
            # Sort by first coordinate
            pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            max_y = -np.inf
            for i in range(len(pts)):
                x, y = pts[i]
                x_next = pts[i+1][0] if i + 1 < len(pts) else ref[0]
                if x_next <= x:
                    continue
                max_y = max(max_y, y)
                if max_y < ref[1]:
                    hv += (x_next - x) * (ref[1] - max_y)
            return max(0, hv)

        def compute_hv(pop_F, ref):
            """Compute hypervolume of a set of objective vectors."""
            return hypervolume_3d(pop_F, ref)

        # Reference point for scoring
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Strategy: 
        # 1. Sample the known WFG4 Pareto front structure
        # 2. Use CMA-ES-like local search around promising regions
        # 3. Use hypervolume-based selection to maintain diversity
        
        # WFG4 structure: The Pareto front can be parameterized.
        # For WFG4 with 12 vars and 3 objectives, the front has a specific shape.
        # Let's use a combination of:
        # - Random sampling in objective space mapped back via optimization
        # - Local refinement using small perturbations
        
        # Phase 1: Initial diverse sampling (50 evals)
        n_init = 50
        pop_X = []
        pop_F = []
        
        # Latin hypercube for diversity
        for i in range(n_init):
            if n_evals >= self.budget or time.time() > deadline:
                break
            x = lo + np.random.rand(self.dim) * span
            fx = evaluate(x)
            pop_X.append(x.copy())
            pop_F.append(fx)
        
        # Phase 2: Hypervolume-based evolutionary search
        # Maintain an archive of non-dominated points
        archive_X = list(pop_X)
        archive_F = list(pop_F)
        
        # CMA-ES-like parameters
        sigma = 0.1
        pop_size = 10  # Small population for local search
        
        n_generations = (self.budget - n_evals) // (pop_size + 1)
        
        for gen in range(max(1, n_generations)):
            if n_evals >= self.budget or time.time() > deadline:
                break
            
            # Select base point from archive using hypervolume contribution
            # Pick points with high marginal hypervolume contribution
            if len(archive_F) == 0:
                break
            
            # Compute marginal HV for each archive point
            n_arch = len(archive_F)
            marginal_hv = np.zeros(n_arch)
            base_hv = compute_hv(archive_F, ref_point)
            
            for i in range(n_arch):
                if time.time() > deadline:
                    break
                # HV without point i
                reduced = [f for j, f in enumerate(archive_F) if j != i]
                reduced_hv = compute_hv(reduced, ref_point)
                marginal_hv[i] = base_hv - reduced_hv
            
            # Select base point with high marginal HV (random among top)
            top_k = min(3, n_arch)
            top_idx = np.argsort(marginal_hv)[-top_k:]
            base_idx = top_idx[np.random.randint(len(top_idx))]
            
            # Generate offspring around base point with Gaussian perturbation
            base_x = archive_X[base_idx].copy()
            
            for k in range(pop_size):
                if n_evals >= self.budget or time.time() > deadline:
                    break
                
                # Gaussian perturbation
                x = base_x + sigma * np.random.randn(self.dim)
                x = np.clip(x, lo, hi)
                
                fx = evaluate(x)
                
                # Check if non-dominated by archive
                is_dominated = False
                for af in archive_F:
                    if np.all(af <= fx) and np.any(af < fx):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    # Add to archive and remove dominated points
                    archive_X.append(x)
                    archive_F.append(fx)
                    
                    # Remove points dominated by new point
                    new_dom_idx = []
                    for i, af in enumerate(archive_F[:-1]):  # Exclude new point
                        if np.all(fx <= af) and np.any(fx < af):
                            new_dom_idx.append(i)
                    
                    for i in sorted(new_dom_idx, reverse=True):
                        archive_X.pop(i)
                        archive_F.pop(i)
                
                # Adapt sigma
                sigma *= 0.95
        
        # Collect all evaluated points
        all_X = archive_X
        all_F = archive_F
        
        return np.array(all_X) if all_X else np.array([])
