import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit

    def _compute_radius(centers):
        """Compute the maximum valid radius for given centers."""
        if centers.shape[0] != N:
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check inside cube
        if np.any(centers < -0.5) or np.any(centers > 0.5):
            return 0.0
        # Minimum distance to boundary
        dist_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_boundary = np.min(dist_boundary)
        # Minimum pairwise distance
        n = len(centers)
        min_pair = float('inf')
        # Vectorized pairwise distance calculation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        if min_pair == 0.0:
            return 0.0
        return min(min_pair / 2.0, min_boundary)

    def _objective(params, centers_initial):
        """Objective function for optimization: negative radius (to maximize)."""
        centers = params.reshape(N, 3)
        r = _compute_radius(centers)
        return -r

    def _construct_bcc_superlattice():
        """
        Construct a distorted BCC superlattice for 50 spheres.
        BCC has higher packing density than SC.
        We create a 3x3x3 grid of BCC unit cells? No, that's 2*27=54 points.
        Let's try to fit 50 points into a BCC-like arrangement.
        
        A BCC lattice has points at (0,0,0) and (0.5, 0.5, 0.5) in a unit cell.
        If we use a grid of 4x4x4 cells, we have 64 cells.
        Total points = 2 * 64 = 128. Too many.
        
        We need 50 points.
        Let's use a 3x3x4 grid of BCC cells?
        3*3*4 = 36 cells. 2*36 = 72 points. Still too many.
        2x3x4 = 24 cells. 48 points. Close.
        2x4x4 = 32 cells. 64 points.
        
        Let's just generate a BCC lattice and take the first 50 points that fit best?
        Or, simpler: Generate a grid of 50 points in a way that mimics BCC locally.
        
        Alternative: Start with a simple cubic grid of 5x5x5 = 125 points, 
        but that's too many.
        
        Let's stick to the strategy: "Represent the packing as a lattice perturbation problem 
        by initializing 50 centers on a distorted Body-Centered Cubic (BCC) superlattice".
        
        Let's create a BCC lattice with lattice constant 'a'.
        Points are at (i*a, j*a, k*a) and ((i+0.5)*a, (j+0.5)*a, (k+0.5)*a).
        We need to choose 'a' and the range of i,j,k such that we get ~50 points inside [-0.5, 0.5].
        
        If we take i,j,k in range -2..2 (5 values), we have 5*5*5 = 125 corners.
        And 125 centers. Total 250.
        
        Let's try a smaller grid.
        i,j,k in range -1..1 (3 values).
        Corners: 3*3*3 = 27.
        Centers: 3*3*3 = 27.
        Total 54.
        
        This is very close to 50.
        We can generate these 54 points, and then remove the 4 that are "least useful" or just take the first 50?
        Or, we can generate them, and then use the optimizer to adjust all 50 (removing 4 randomly or the ones closest to boundaries?).
        
        Actually, the task requires exactly N=50 points.
        Let's generate the 54 BCC points, and then select the 50 with the largest minimum distance to the boundary? 
        Or just take the first 50?
        
        Let's generate the 54 points.
        Scale them to fit inside [-0.5, 0.5] with some margin.
        The extent of the grid is from -1.5*a to 1.5*a (for corners) and -a to a (for centers)?
        No, if i is -1, 0, 1.
        Corners at -a, 0, a.
        Centers at -0.5a, 0.5a, 1.5a? No.
        Center of cell (i,j,k) is at (i+0.5, j+0.5, k+0.5)*a.
        If i=-1, center is at -0.5a.
        If i=1, center is at 1.5a.
        So extent is -a to 1.5a for centers? And -a to a for corners?
        Max extent is 1.5a.
        We want 1.5a <= 0.5 - margin.
        Let margin = 0.05.
        1.5a = 0.45 => a = 0.3.
        
        Let's generate the points with a=0.3.
        """
        a = 0.3
        pts = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # Corner
                    pts.append([i*a, j*a, k*a])
                    # Center
                    pts.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
        
        pts = np.array(pts)
        
        # We have 54 points. We need 50.
        # Let's compute the score for all 54, and then remove the 4 that contribute least to the min distance?
        # Simpler: Just take the first 50. The symmetry might make it not matter much, 
        # but the optimizer will fix it.
        # However, taking the first 50 might break symmetry.
        # Let's try to center the grid.
        # The current grid is not centered at 0.
        # Corners: -0.3, 0, 0.3.
        # Centers: -0.15, 0.15, 0.45.
        # Min: -0.3. Max: 0.45.
        # Center of mass: (-0.3+0.45)/2 = 0.075.
        # Let's shift so it's centered at 0.
        pts -= np.mean(pts, axis=0)
        
        # Now take 50 points.
        # To be fair, let's sort by distance to origin and take the innermost 50?
        # Or just take the first 50.
        # Let's take the first 50.
        return pts[:50].copy()

    def _construct_cubic_grid():
        """Simple cubic grid as baseline."""
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def _random_initialization():
        """Random initialization inside the cube with some margin."""
        margin = 0.1
        centers = np.random.uniform(-0.5 + margin, 0.5 - margin, size=(N, 3))
        return centers

    def _optimize_from_start(centers_start, time_limit=30.0):
        """Run local optimization from a starting point."""
        start_time_opt = time.time()
        params = centers_start.flatten()
        
        # Use L-BFGS-B for constrained optimization
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        best_params = params.copy()
        best_r = _compute_radius(centers_start)
        
        try:
            result = minimize(
                _objective, 
                params, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.success or result.fun < -best_r:
                new_centers = result.x.reshape(N, 3)
                new_r = _compute_radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_params = result.x.copy()
        except Exception:
            pass
        
        elapsed = time.time() - start_time_opt
        return best_params.reshape(N, 3), best_r, elapsed

    def _simulated_annealing(centers_start, time_limit=20.0):
        """Simple simulated annealing to escape local minima."""
        start_time_sa = time.time()
        centers = centers_start.copy()
        current_r = _compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        
        T = 0.02  # Initial temperature
        cooling = 0.995
        
        while time.time() - start_time_sa < time_limit:
            if T < 1e-6:
                break
            
            # Random perturbation
            idx = np.random.randint(N)
            delta = np.random.normal(0, T * 0.1, size=3)
            new_centers = centers.copy()
            new_centers[idx] += delta
            new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
            
            new_r = _compute_radius(new_centers)
            delta_r = new_r - current_r
            
            if delta_r > 0 or np.random.random() < np.exp(delta_r / T):
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            T *= cooling
        
        return best_centers, best_r, time.time() - start_time_sa

    # Main logic
    global_start = time.time()
    deadline = time_limit
    
    best_centers = None
    best_r = 0.0
    
    # 1. Try BCC Superlattice
    if time.time() - global_start < deadline:
        centers_bcc = _construct_bcc_superlattice()
        r_bcc = _compute_radius(centers_bcc)
        best_centers = centers_bcc.copy()
        best_r = r_bcc
        
        # Optimize from BCC
        time_remaining = deadline - (time.time() - global_start)
        if time_remaining > 10:
            opt_centers, opt_r, elapsed = _optimize_from_start(centers_bcc, time_limit=min(30.0, time_remaining - 5))
            if opt_r > best_r:
                best_r = opt_r
                best_centers = opt_centers.copy()

    # 2. Try Cubic Grid
    if time.time() - global_start < deadline:
        centers_grid = _construct_cubic_grid()
        r_grid = _compute_radius(centers_grid)
        if r_grid > best_r:
            best_r = r_grid
            best_centers = centers_grid.copy()
        
        # Optimize from Grid
        time_remaining = deadline - (time.time() - global_start)
        if time_remaining > 10:
            opt_centers, opt_r, elapsed = _optimize_from_start(centers_grid, time_limit=min(20.0, time_remaining - 5))
            if opt_r > best_r:
                best_r = opt_r
                best_centers = opt_centers.copy()

    # 3. Random Restarts
    n_random_tries = 3
    for i in range(n_random_tries):
        if time.time() - global_start > deadline - 10:
            break
        
        centers_rand = _random_initialization()
        r_rand = _compute_radius(centers_rand)
        
        time_remaining = deadline - (time.time() - global_start)
        if time_remaining < 15:
            break
        
        opt_centers, opt_r, elapsed = _optimize_from_start(centers_rand, time_limit=min(15.0, time_remaining - 5))
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # 4. Simulated Annealing on the best found
    if time.time() - global_start < deadline - 10:
        time_remaining = deadline - (time.time() - global_start)
        sa_centers, sa_r, elapsed = _simulated_annealing(best_centers, time_limit=min(20.0, time_remaining - 5))
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()

    # 5. Final Optimization Pass
    if time.time() - global_start < deadline - 5:
        time_remaining = deadline - (time.time() - global_start)
        opt_centers, opt_r, elapsed = _optimize_from_start(best_centers, time_limit=min(10.0, time_remaining - 2))
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # Ensure validity
    final_centers = np.clip(best_centers, -0.5, 0.5)
    final_r = _compute_radius(final_centers)
    
    if final_r <= 0.0:
        # Fallback to cubic grid
        final_centers = _construct_cubic_grid()
    
    return final_centers

if __name__ == "__main__":
    res = construct()
    print(res.shape)
