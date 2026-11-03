"""Circle packing in perimeter-4 rectangle - Direct Physical Relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # Perimeter constraint: w + h = 2.
    # We will optimize w (width) and h = 2 - w.
    # We assume an aspect ratio near 1.5 or 1/1.5 is good, but let the optimizer find it.
    # Actually, for 21 circles, a 3x7 or 4x5-ish arrangement is common.
    # Let's fix the rectangle shape to reduce dimensionality? 
    # No, the task says "variable rectangle". 
    # However, optimizing 21*2 + 1 = 43 variables with Nelder-Mead is feasible if fast.
    
    # A better approach: 
    # 1. Fix a good aspect ratio w (e.g., 1.4, 0.6).
    # 2. Optimize positions (x,y) for N circles.
    # 3. Calculate max radius r.
    # 4. Iterate over a few aspect ratios.
    
    # Since we want a structural change, let's use a "Soft Overlap" energy minimization.
    # Energy E = sum(overlap_penalty) + boundary_penalty
    # We minimize E. If E -> 0, we have a valid packing.
    # Then we maximize r by scaling? No, r is determined by the min distance.
    
    # Alternative: 
    # Minimize -r + Penalty(overlaps, boundaries)
    # Variables: x_i, y_i, w (h=2-w)
    # This is what the current code does with L-BFGS-B.
    
    # New Strategy: 
    # Use a "Repulsive Force" simulation (Molecular Dynamics style) to untangle circles,
    # then use the resulting configuration as a start for a precise local search.
    # This avoids getting stuck in bad lattice local minima.
    
    def run_md_initialization(w, h, n_steps=2000, dt=0.001, k_rep=100, k_wall=100):
        """
        Simulate repulsive forces to spread circles out.
        Returns positions.
        """
        # Initial random positions
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        # Start with a rough grid to help convergence
        rows, cols = 3, 7
        xs = np.linspace(0.1*w, 0.9*w, cols)
        ys = np.linspace(0.1*h, 0.9*h, rows)
        X, Y = np.meshgrid(xs, ys)
        pos = np.column_stack([X.ravel(), Y.ravel()])
        
        # Add small noise
        pos += rng.uniform(-0.01, 0.01, pos.shape)
        
        # Target radius guess (will be small initially)
        r_guess = min(w, h) / 10.0
        
        for step in range(n_steps):
            forces = np.zeros_like(pos)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pos[i] - pos[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-6:
                        # Jitter
                        diff = rng.uniform(-0.01, 0.01, 2)
                        dist = np.linalg.norm(diff)
                    
                    # Repel if closer than 2*r_guess * 1.5 (interaction range)
                    # Actually, for packing, we want them to touch.
                    # Let's use a potential that pushes them apart if dist < 2*r_target
                    # But we don't know r_target. 
                    # Let's just minimize the sum of squared distances to a "soft" contact?
                    # Simpler: Minimize sum of (dist - 2*r)^2 for close pairs?
                    # This is tricky without r.
                    
                    # Let's use a standard Lennard-Jones-like repulsion to prevent overlap
                    # and an attraction to keep them compact?
                    # No, for packing in a box, we just need to spread them to fill space.
                    
                    # Let's use a penalty for dist < 2*r_current_estimate
                    # But r changes.
                    
                    # Let's stick to a simpler "Unfold" strategy:
                    # Minimize the sum of inverse square distances?
                    pass
            
            # This MD approach is getting complicated to implement robustly in short time.
            # Let's pivot to a different structural change: 
            # **Simulated Annealing on the positions.**
            # This is a global search method that can escape local minima better than L-BFGS-B from a single start.
            
            break # Abort MD, use SA instead
            
    def sa_packing(w, h, n_iter=5000, t_start=0.05, t_end=0.001, seed=42):
        """
        Simulated Annealing to find a good packing.
        Objective: Maximize r (min of wall dist, pair dist / 2)
        Penalty: If any constraint violated (shouldn't be if we track r correctly, 
                 but we move points, so r changes).
        
        Actually, standard SA on positions with objective = -r is non-smooth and hard.
        
        Better: 
        Use **Basin Hopping** on the positions.
        """
        pass

    # Let's try a different, robust structural change:
    # **Genetic Algorithm / Differential Evolution on the full set of coordinates.**
    # The current program uses a parametric lattice. 
    # DE explores the full 42D position space (plus w).
    # This is a different *representation* (free coordinates vs lattice params).
    
    from scipy.optimize import differential_evolution
    
    def objective_de(z):
        """
        z: [x1, y1, ..., x21, y21, w]
        """
        # Unpack
        coords = z[:2*N].reshape(N, 2)
        w = z[2*N]
        h = 2.0 - w
        
        # Check bounds
        # If any coord is out of [0,w] or [0,h], return high penalty
        if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] > w) or \
           np.any(coords[:, 1] < 0) or np.any(coords[:, 1] > h):
            return 100.0
            
        # Calculate max feasible radius
        r_wall = min(
            np.min(coords[:, 0]),
            np.min(w - coords[:, 0]),
            np.min(coords[:, 1]),
            np.min(h - coords[:, 1])
        )
        
        r_pair = np.inf
        # Vectorized pairwise distance check for speed
        # dist[i,j] = ||p_i - p_j||
        # We only need min dist for i != j
        # Using scipy.spatial.distance.pdist is efficient
        from scipy.spatial.distance import pdist
        d = pdist(coords)
        min_dist = np.min(d)
        r_pair = min_dist / 2.0
        
        r = min(r_wall, r_pair)
        
        # We want to maximize r. So minimize -r.
        # Add small penalty for "tight" packing to encourage robustness? No, just -r.
        return -r

    best_val = -np.inf
    best_z = None
    
    # Bounds for DE
    # x, y in [0, 2] (since w+h=2, max dimension is 2)
    # w in [0.5, 1.5]
    bounds = [(0.0, 2.0)] * (2*N) + [(0.5, 1.5)]
    
    # DE settings
    # popsize: 15-20
    # maxiter: 100
    # tol: 1e-6
    
    # We will run DE for a few aspect ratios to be safe, or just let it find w.
    # Let's let it find w.
    
    try:
        res = differential_evolution(
            objective_de, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15,
            tol=1e-5,
            polish=True, # Uses L-BFGS-B on the best result
            workers=1
        )
        
        if res.fun < -0.01: # Valid improvement
            best_val = -res.fun
            best_z = res.x
    except Exception as e:
        print(f"DE failed: {e}")
        
    # If DE didn't find a good solution, try a few manual aspect ratios with local polish
    if best_val < 1.0:
        for w0 in [1.2, 1.4, 1.6]:
            h0 = 2.0 - w0
            # Init with grid
            rows, cols = 3, 7
            xs = np.linspace(0.1*w0, 0.9*w0, cols)
            ys = np.linspace(0.1*h0, 0.9*h0, rows)
            X, Y = np.meshgrid(xs, ys)
            z0 = np.column_stack([X.ravel(), Y.ravel()]).flatten()
            z0 = np.append(z0, w0)
            
            # Local optimize with L-BFGS-B
            def obj_local(z):
                coords = z[:2*N].reshape(N, 2)
                w = z[2*N]
                h = 2.0 - w
                if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] > w) or \
                   np.any(coords[:, 1] < 0) or np.any(coords[:, 1] > h):
                    return 100.0
                r_wall = min(
                    np.min(coords[:, 0]),
                    np.min(w - coords[:, 0]),
                    np.min(coords[:, 1]),
                    np.min(h - coords[:, 1])
                )
                from scipy.spatial.distance import pdist
                d = pdist(coords)
                r_pair = np.min(d) / 2.0
                r = min(r_wall, r_pair)
                return -r
            
            bounds_local = [(0.0, w0)] * N + [(0.0, h0)] * N + [(w0, w0)] # Fix w for this run
            # Actually, let w vary slightly? No, fix w for stability.
            # Bounds for x: [0, w0], y: [0, h0]
            b_x = [(0.0, w0)] * N
            b_y = [(0.0, h0)] * N
            bounds_local = b_x + b_y + [(w0, w0)]
            
            try:
                res_l = minimize(obj_local, z0, method='L-BFGS-B', bounds=bounds_local, 
                                 options={'maxiter': 200, 'ftol': 1e-10})
                val = -res_l.fun
                if val > best_val:
                    best_val = val
                    best_z = res_l.x
            except:
                pass
    
    if best_z is None:
        # Fallback
        w, h = 1.4, 0.6
        rows, cols = 3, 7
        xs = np.linspace(0.5*w/cols, w - 0.5*w/cols, cols)
        ys = np.linspace(0.5*h/rows, h - 0.5*h/rows, rows)
        X, Y = np.meshgrid(xs, ys)
        coords = np.column_stack([X.ravel(), Y.ravel()])
        r = 0.01
        circles = np.column_stack([coords, np.full(N, r)])
        return circles
    
    # Extract best result
    coords = best_z[:2*N].reshape(N, 2)
    w = best_z[2*N]
    h = 2.0 - w
    
    # Calculate final radius
    r_wall = min(
        np.min(coords[:, 0]),
        np.min(w - coords[:, 0]),
        np.min(coords[:, 1]),
        np.min(h - coords[:, 1])
    )
    from scipy.spatial.distance import pdist
    d = pdist(coords)
    r_pair = np.min(d) / 2.0
    r = min(r_wall, r_pair) * 0.9999 # Slight shrink for safety
    
    circles = np.zeros((N, 3))
    circles[:, :2] = coords
    circles[:, 2] = r
    
    # Final validation and clipping
    # Ensure centers are within [r, w-r] and [r, h-r]
    circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
    circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
    
    # Re-check overlaps after clipping
    for _ in range(5):
        d = pdist(circles[:, :2])
        min_d = np.min(d)
        if min_d < 2 * circles[0, 2] * 1.0001:
            circles[:, 2] = (min_d / 2.0) * 0.9999
            circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
            circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
        else:
            break
            
    return circles

