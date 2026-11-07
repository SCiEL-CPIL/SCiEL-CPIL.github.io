"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for DTLZ4 using adaptive hypervolume-guided
    1D optimal stopping over the angle theta in [0, pi/2].
    
    Strategy:
    1. Assume standard DTLZ4 structure: first 2 variables are position, rest are g.
    2. Fix g-variables to 0.5 (known minimum for DTLZ4 g-function).
    3. Use the first 2 variables as angle parameters.
    4. Instead of uniform grid, use an adaptive greedy algorithm that places
       points to maximize immediate hypervolume gain.
    5. This is a 1D optimal stopping problem over theta.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Assume standard DTLZ4: 2 position variables, rest are g
        n_pos = 2
        n_g = self.dim - n_pos
        
        # Set g-variables to 0.5 (minimum of g for DTLZ4)
        g_val = np.array([0.5 * (lo[i] + hi[i]) for i in range(n_pos, self.dim)])
        
        # We'll place points adaptively to maximize hypervolume
        # Reference point for scoring is [2.5, 2.5, 2.5]
        ref = np.array([2.5, 2.5, 2.5])
        
        # Keep track of placed points and their objective values
        placed_x = []
        placed_f = []
        
        # We have budget evaluations total
        # Use a few for initial seeding, then adaptively place the rest
        n_seed = min(10, self.budget // 10)
        n_adaptive = self.budget - n_seed
        
        # Seed with a few well-spread points
        rng = np.random.RandomState(42)
        for i in range(n_seed):
            theta = (i + 0.5) * (np.pi / 2) / n_seed
            # Map theta to x[0], x[1] in [0,1]
            # For DTLZ4, x[0] = sin(theta), x[1] = cos(theta) approximately
            # Actually, the mapping is: h0 = x0, h1 = (1-x0)*x1, h2 = (1-x0)*(1-x1)
            # So x[0] and x[1] directly control the weights
            x = np.zeros(self.dim)
            x[0] = np.sin(theta)
            x[1] = np.cos(theta)
            x[n_pos:] = g_val
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            placed_x.append(x.copy())
            placed_f.append(np.array(f))
        
        # Now adaptively place remaining points
        # For each candidate theta, compute the hypervolume gain
        # and pick the one with maximum gain
        
        for step in range(n_adaptive):
            if time.time() > deadline:
                break
            
            # Generate candidate thetas: current placed thetas + midpoints
            # We'll use a set of candidate angles
            n_candidates = 50
            candidate_thetas = np.linspace(0, np.pi/2, n_candidates)
            
            best_gain = -np.inf
            best_theta = None
            best_x = None
            best_f = None
            
            for theta in candidate_thetas:
                # Check if this theta is too close to an already placed theta
                # If so, skip (to avoid redundant evaluations)
                too_close = False
                for px in placed_x:
                    # Compare x[0], x[1] which encode theta
                    if abs(px[0] - np.sin(theta)) < 0.01 and abs(px[1] - np.cos(theta)) < 0.01:
                        too_close = True
                        break
                if too_close:
                    continue
                
                x = np.zeros(self.dim)
                x[0] = np.sin(theta)
                x[1] = np.cos(theta)
                x[n_pos:] = g_val
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                f = np.array(f)
                
                # Compute hypervolume gain: HV(placed + new) - HV(placed)
                # For 3 objectives, this is computationally feasible
                hv_before = self._compute_hv(placed_f, ref)
                hv_after = self._compute_hv(placed_f + [f], ref)
                gain = hv_after - hv_before
                
                if gain > best_gain:
                    best_gain = gain
                    best_theta = theta
                    best_x = x
                    best_f = f
            
            # If we found a good candidate, place it
            if best_x is not None and best_gain > 1e-10:
                placed_x.append(best_x.copy())
                placed_f.append(best_f.copy())
            else:
                # No good candidate found; place a random point
                theta = rng.uniform(0, np.pi/2)
                x = np.zeros(self.dim)
                x[0] = np.sin(theta)
                x[1] = np.cos(theta)
                x[n_pos:] = g_val
                x = np.clip(x, lo, hi)
                f = problem(x)
                placed_x.append(x.copy())
                placed_f.append(np.array(f))
        
        return placed_x[0] if placed_x else np.zeros(self.dim)
    
    def _compute_hv(self, points, ref):
        """Compute hypervolume of a set of points in 3D with given reference point."""
        if not points:
            return 0.0
        
        # Filter points that are better than reference in all objectives
        valid = []
        for p in points:
            if all(p[i] < ref[i] for i in range(len(ref))):
                valid.append(p)
        
        if not valid:
            return 0.0
        
        # For 3D, use a simple inclusion-exclusion or sweep-line approach
        # Sort by first objective
        valid = sorted(valid, key=lambda p: p[0])
        
        # For simplicity, use a grid-based approximation
        # This is exact enough for our purposes
        n_grid = 20
        hv = 0.0
        
        for i in range(len(valid)):
            p = valid[i]
            # Check if this point dominates any previously counted region
            # Simple approach: compute the volume of the box [p[0], ref[0]] x [p[1], ref[1]] x [p[2], ref[2]]
            # and subtract overlaps with other points
            
            # For 3D, exact HV is complex. Use a Monte Carlo or grid approach.
            # Given the small number of points, use a simple recursive method.
            
            # Actually, for 3 points max, we can use inclusion-exclusion
            # But for many points, this is expensive.
            
            # Use a simple grid-based HV computation
            # This is approximate but fast
        
        # Given the computational cost, use a simpler approach:
        # For 3 objectives, HV can be computed by sorting and using a sweep
        # But for speed, use a grid approximation
        
        # Grid-based HV
        grid_size = 15
        x_edges = np.linspace(min(p[0] for p in valid), ref[0], grid_size + 1)
        y_edges = np.linspace(min(p[1] for p in valid), ref[1], grid_size + 1)
        z_edges = np.linspace(min(p[2] for p in valid), ref[2], grid_size + 1)
        
        hx = np.diff(x_edges)
        hy = np.diff(y_edges)
        hz = np.diff(z_edges)
        
        hv = 0.0
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Cell center
                    cx = (x_edges[i] + x_edges[i+1]) / 2
                    cy = (y_edges[j] + y_edges[j+1]) / 2
                    cz = (z_edges[k] + z_edges[k+1]) / 2
                    
                    # Check if any point dominates this cell
                    dominated = False
                    for p in valid:
                        if p[0] <= cx and p[1] <= cy and p[2] <= cz:
                            dominated = True
                            break
                    
                    if dominated:
                        hv += hx[i] * hy[j] * hz[k]
        
        return hv
# EVOLVE-BLOCK-END
