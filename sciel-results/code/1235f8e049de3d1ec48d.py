"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive grid refinement multi-objective optimizer for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # State: list of (x, f) tuples for all evaluated points
        # We maintain a set of all evaluated points to avoid duplicates
        evaluated = []  # list of (x, f)
        eval_set = set()

        def evaluate(x):
            """Evaluate a point, return f. Clip to bounds."""
            x = np.clip(x, lo, hi)
            # Check if we've seen this point before (using rounded key)
            key = tuple(np.round(x, 10))
            if key in eval_set:
                return None
            f = np.array(problem(x), dtype=np.float64)
            eval_set.add(key)
            evaluated.append((x, f))
            return f

        def is_dominated(f, f_list):
            """Check if f is dominated by any point in f_list."""
            for g in f_list:
                if np.all(g <= f) and np.any(g < f):
                    return True
            return False

        def non_dominated_set(points):
            """Return indices of non-dominated points."""
            n = len(points)
            if n == 0:
                return []
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        # Phase 1: Initial sampling with Latin Hypercube-like design
        n_init = min(40, budget)
        rng = np.random.default_rng(42)

        # Create LHS-like points
        for i in range(n_init):
            # Use stratified sampling
            x = np.zeros(dim)
            for d in range(dim):
                frac = (i + rng.random()) / n_init
                x[d] = lo[d] + frac * (hi[d] - lo[d])
            f = evaluate(x)
            if f is None:
                continue

        # Phase 2: Adaptive grid refinement
        # We'll iteratively subdivide the space, focusing on regions with non-dominated points
        # Use a queue of cells to explore
        # Each cell is defined by (lo_cell, hi_cell)

        # Start with the full space as one cell
        cells = [(lo.copy(), hi.copy())]
        eval_count = len(evaluated)

        # Track best hypervolume-contributing points
        all_f = [f for _, f in evaluated]

        while eval_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            if not cells:
                break

            # Pick the cell with the best potential (smallest volume or closest to known front)
            # For simplicity, pick cells in order of size (smallest first for refinement)
            # Actually, let's pick cells that contain or are near non-dominated points
            
            # Compute volumes
            cell_vols = []
            for cell in cells:
                vol = np.prod(cell[1] - cell[0])
                cell_vols.append(vol)
            
            # Pick the largest cell to split (coarse to fine)
            # Or pick cells near the Pareto front
            # Let's pick the cell with the largest volume for balanced exploration
            idx = np.argmax(cell_vols)
            cell_lo, cell_hi = cells[idx]
            cells.pop(idx)

            # Split this cell in half along the dimension with the largest extent
            extents = cell_hi - cell_lo
            max_dim = np.argmax(extents)
            
            # Create two sub-cells
            mid = (cell_lo[max_dim] + cell_hi[max_dim]) / 2.0
            
            sub1_lo = cell_lo.copy()
            sub1_hi = cell_hi.copy()
            sub1_hi[max_dim] = mid
            
            sub2_lo = cell_lo.copy()
            sub2_hi = cell_hi.copy()
            sub2_lo[max_dim] = mid

            # Evaluate the centroids of both sub-cells
            for sub_lo, sub_hi in [(sub1_lo, sub1_hi), (sub2_lo, sub2_hi)]:
                if eval_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                centroid = (sub_lo + sub_hi) / 2.0
                f = evaluate(centroid)
                if f is not None:
                    eval_count += 1
                    all_f.append(f)
            
            # Add sub-cells back to the list for further refinement
            cells.append((sub1_lo, sub1_hi))
            cells.append((sub2_lo, sub2_hi))

            # Limit the number of cells to prevent memory issues
            if len(cells) > 2000:
                # Remove the smallest cells (least important)
                cell_vols = [np.prod(c[1] - c[0]) for c in cells]
                # Keep the largest cells
                keep_idx = np.argsort(cell_vols)[-1000:]
                cells = [cells[i] for i in keep_idx]

        # Phase 3: Local refinement around non-dominated points
        # For remaining budget, do local search around the best points
        if eval_count < budget:
            # Get non-dominated points
            if all_f:
                nd_idx = non_dominated_set(all_f)
                nd_points = [evaluated[i] for i in nd_idx]
                
                # For each non-dominated point, try small perturbations
                remaining = budget - eval_count
                n_polish = min(remaining, 50)
                
                for i in range(n_polish):
                    if eval_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Pick a random non-dominated point
                    if not nd_points:
                        break
                    pt_idx = np.random.default_rng(i).integers(len(nd_points))
                    x_base, f_base = nd_points[pt_idx]
                    
                    # Small perturbation
                    scale = 0.01 * (hi - lo)
                    x_new = x_base + scale * np.random.default_rng(i + 1000).standard_normal(dim)
                    f_new = evaluate(x_new)
                    if f_new is not None:
                        eval_count += 1
                        all_f.append(f_new)
                        # Update non-dominated set if needed
                        if not is_dominated(f_new, all_f[:-1]):
                            nd_points.append((x_new, f_new))

        return evaluated
# EVOLVE-BLOCK-END
