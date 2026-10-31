import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def objective(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        Returns a scalar.
        """
        # pts: (N, D)
        # Compute all pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Get upper triangle indices to avoid diagonal and duplicates
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        
        # Penalize coincident points
        if min_d_sq < 1e-8:
            return 1e10
            
        max_d_sq = np.max(d_sq)
        
        # Ratio squared is (max_d / min_d)^2 = max_d_sq / min_d_sq
        return max_d_sq / min_d_sq

    def make_hex_cluster(n):
        """
        Generate a hexagonal lattice and pick n closest points to origin.
        """
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        
        pts = []
        # Generate a large enough grid
        for i in range(-5, 6):
            for j in range(-5, 6):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        
        # Center
        pts -= pts.mean(axis=0)
        
        # Sort by distance to center
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)
        return pts[order[:n]]

    def make_circular(n):
        """
        1 center, 6 on ring 1, 9 on ring 2.
        """
        pts = []
        pts.append([0, 0])
        r1 = 1.0
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        
        r2 = 1.8 # Approx 2*sqrt(3)/2 * something? In hex lattice, next ring is dist 2 or sqrt(3)?
        # In hex lattice with min dist 1:
        # Neighbors: 6 at dist 1.
        # Next: 6 at dist sqrt(3) (corners) and 6 at dist 2 (edges).
        # Total 12 at dist < 2.
        # We need 9 more.
        # Let's just use a heuristic radius.
        for i in range(9):
            angle = 2 * np.pi * i / 9 + np.pi / 9
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
            
        return np.array(pts, dtype=float)

    def make_square_grid(n):
        side = int(np.ceil(n ** 0.5))
        grid = []
        for i in range(side):
            for j in range(side):
                grid.append([i, j])
        return np.array(grid[:n], dtype=float)

    # Initial candidates to seed DE or use as local starts
    inits = [
        make_hex_cluster(N),
        make_circular(N),
        make_square_grid(N)
    ]
    
    # Scale inits to be within reasonable bounds for DE
    # DE bounds
    lower = np.full(N * D, -10.0)
    upper = np.full(N * D, 10.0)
    
    best_val = np.inf
    best_pts = inits[0].copy()
    
    # 1. Local optimization of initial heuristics
    for init in inits:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(
                lambda x: objective(x.reshape(N, D)),
                init.flatten(),
                method='Powell',
                options={'maxiter': 200, 'maxfev': 1000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_pts = res.x.reshape(N, D)
        except Exception:
            pass

    # 2. Differential Evolution
    # We have ~120s left.
    # DE can be slow. We'll run it with a moderate population.
    if time.time() < deadline - 10:
        try:
            # Use the best local result as one of the seeds?
            # DE handles its own population.
            # Set seed for reproducibility? No, random is fine, but maybe use best as a guide?
            # DE doesn't take a single seed, it takes bounds.
            # We can initialize DE with a population that includes our best points.
            
            # Create a population
            popsize = 20
            # differential_evolution does not easily allow custom initial population in older scipy versions
            # But we can just run it.
            
            # To save time, we might restrict the search space or use fewer iterations.
            # Let's try a quick DE run.
            de_res = differential_evolution(
                lambda x: objective(x.reshape(N, D)),
                bounds=zip(lower, upper),
                maxiter=50,
                popsize=15,
                tol=0.01,
                seed=42,
                workers=1
            )
            
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_pts = de_res.x.reshape(N, D)
                
        except Exception:
            pass

    # 3. Final Polish
    if time.time() < deadline - 5:
        try:
            # Polish the best result with Powell
            # Use a tighter maxiter since we are close
            res = minimize(
                lambda x: objective(x.reshape(N, D)),
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 1000, 'maxfev': 10000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_pts = res.x.reshape(N, D)
        except Exception:
            pass

    return best_pts
