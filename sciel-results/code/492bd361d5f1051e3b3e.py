"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree

DEADLINE = 140  # safety margin before 150s limit


def _pair_dist_matrix(c):
    """Compute pairwise distance matrix."""
    d = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return dists


def _compute_radius(c):
    """Compute the common radius for a given set of centres."""
    dists = _pair_dist_matrix(c)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _softmin_radius(c, alpha):
    """Smooth approximation of the radius using soft-min."""
    dists = _pair_dist_matrix(c)
    # Soft-min of pairwise distances / 2
    pair_softmin = np.logaddexp.reduce(np.log(dists / 2.0 + 1e-300)) / alpha
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    wall_softmin = np.logaddexp.reduce(np.log(wall_dists + 1e-300)) / alpha
    return min(pair_softmin, wall_softmin)


def _lloyd_step(c, iterations=20):
    """Voronoi-Lloyd equalization: move each point to the centroid of its Voronoi cell."""
    c = c.copy()
    for _ in range(iterations):
        try:
            # Build Delaunay triangulation for Voronoi cells
            # For 3D with 50 points, use a simpler approach: 
            # move each point away from its nearest neighbors
            tree = cKDTree(c)
            # Find 8 nearest neighbors for each point
            _, idx = tree.query(c, k=9)  # k=9 because includes self
            idx = idx[:, 1:]  # remove self
            
            # Compute repulsion forces
            forces = np.zeros_like(c)
            for i in range(N):
                for j in idx[i]:
                    diff = c[i] - c[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-10:
                        # Repulsive force proportional to 1/dist^2
                        force = diff / (dist ** 2)
                        forces[i] += force
            
            # Move points in the direction of the net force
            step_size = 0.01
            c += step_size * forces
            
            # Keep points inside the cube
            c = np.clip(c, -0.49, 0.49)
        except Exception:
            break
    return c


def _lloyd_optimize(c, time_budget=10):
    """Run Lloyd iteration with time budget."""
    start = time.time()
    c_current = c.copy()
    best_r = _compute_radius(c_current)
    
    while time.time() - start < time_budget:
        c_new = _lloyd_step(c_current, iterations=5)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            c_current = c_new
        else:
            break
    
    return c_current


def _smooth_obj(x, alpha):
    """Smooth objective function for optimization."""
    c = x.reshape(N, 3)
    return -_softmin_radius(c, alpha)


def _smooth_polish(c, alphas=[10, 20, 50, 100], time_budget=30):
    """Polish using multi-scale soft-min optimization."""
    start = time.time()
    c_current = c.copy()
    best_r = _compute_radius(c_current)
    
    for alpha in alphas:
        if time.time() - start > time_budget:
            break
        try:
            res = minimize(
                lambda x, a=alpha: _smooth_obj(x, a),
                c_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            c_opt = res.x.reshape(N, 3)
            r_opt = _compute_radius(c_opt)
            if r_opt > best_r:
                best_r = r_opt
                c_current = c_opt.copy()
        except Exception:
            pass
    
    return c_current


def _repulsion_anneal(c, time_budget=20):
    """Anneal repulsion to push spheres apart."""
    start = time.time()
    c_current = c.copy()
    best_r = _compute_radius(c_current)
    
    # Start with strong repulsion, gradually weaken
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for temp in temperatures:
        if time.time() - start > time_budget:
            break
        for _ in range(100):
            if time.time() - start > time_budget:
                break
            # Pick a random sphere
            i = np.random.randint(N)
            # Propose a move
            delta = np.random.normal(0, temp, size=3)
            c_new = c_current.copy()
            c_new[i] += delta
            c_new[i] = np.clip(c_new[i], -0.49, 0.49)
            
            # Accept if it improves or with some probability
            r_new = _compute_radius(c_new)
            if r_new > best_r or np.random.rand() < np.exp((r_new - best_r) / 0.01):
                c_current = c_new
                best_r = max(best_r, r_new)
    
    return c_current


def _metropolis_mc(c, time_budget=30):
    """Metropolis Monte Carlo to escape local minima."""
    start = time.time()
    c_current = c.copy()
    best_c = c_current.copy()
    best_r = _compute_radius(c_current)
    
    # Temperature schedule
    T0 = 0.05
    T1 = 0.001
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() - start > time_budget:
            break
        # Linear cooling
        T = T0 + (T1 - T0) * step / n_steps
        
        # Pick a random sphere
        i = np.random.randint(N)
        # Propose a move with Gaussian step
        delta = np.random.normal(0, T, size=3)
        c_new = c_current.copy()
        c_new[i] += delta
        c_new[i] = np.clip(c_new[i], -0.49, 0.49)
        
        # Compute new radius
        r_new = _compute_radius(c_new)
        r_old = _compute_radius(c_current)
        
        # Metropolis acceptance
        if r_new > r_old or np.random.rand() < np.exp((r_new - r_old) / (T + 1e-10)):
            c_current = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
    
    return best_c


def _local_search(c, time_budget=20):
    """Final local search with small perturbations."""
    start = time.time()
    c_current = c.copy()
    best_r = _compute_radius(c_current)
    
    step_sizes = [0.01, 0.005, 0.001, 0.0005]
    
    for step_size in step_sizes:
        if time.time() - start > time_budget:
            break
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            for i in range(N):
                if time.time() - start > time_budget:
                    break
                for axis in range(3):
                    for direction in [-1, 1]:
                        c_trial = c_current.copy()
                        c_trial[i, axis] += direction * step_size
                        c_trial[i, axis] = np.clip(c_trial[i, axis], -0.49, 0.49)
                        
                        r_trial = _compute_radius(c_trial)
                        if r_trial > best_r + 1e-12:
                            c_current = c_trial
                            best_r = r_trial
                            improved = True
    
    return c_current


def _seed_fcc():
    """Generate an FCC-like initial configuration."""
    # FCC structure in a cube
    points = []
    # Corner positions
    for x in [-0.4, 0.4]:
        for y in [-0.4, 0.4]:
            for z in [-0.4, 0.4]:
                points.append([x, y, z])
    # Face centers
    for x in [-0.4, 0.4]:
        for y in [-0.4, 0.4]:
            points.append([x, y, 0.0])
        for z in [-0.4, 0.4]:
            points.append([x, 0.0, z])
    for y in [-0.4, 0.4]:
        for z in [-0.4, 0.4]:
            points.append([0.0, y, z])
    
    # Add more points to reach 50
    # Body centers of sub-cubes
    for x in [-0.2, 0.2]:
        for y in [-0.2, 0.2]:
            for z in [-0.2, 0.2]:
                points.append([x, y, z])
    
    # Trim to 50 if needed
    if len(points) > 50:
        points = points[:50]
    
    # If we have fewer, add random points
    while len(points) < 50:
        points.append(np.random.uniform(-0.4, 0.4, size=3))
    
    return np.array(points[:50])


def _seed_grid():
    """Generate a 5x5x2 grid."""
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.array([-0.25, 0.25])
    
    points = []
    for zi in z:
        for yi in y:
            for xi in x:
                points.append([xi, yi, zi])
    return np.array(points)


def _seed_bcc():
    """Generate a BCC-like initial configuration."""
    points = []
    # Simple cubic grid
    for x in np.linspace(-0.4, 0.4, 4):
        for y in np.linspace(-0.4, 0.4, 4):
            for z in np.linspace(-0.4, 0.4, 3):
                points.append([x, y, z])
    
    # Add body centers
    for x in np.linspace(-0.2, 0.2, 3):
        for y in np.linspace(-0.2, 0.2, 3):
            for z in np.linspace(-0.2, 0.2, 2):
                points.append([x, y, z])
    
    # Trim to 50
    if len(points) > 50:
        points = points[:50]
    while len(points) < 50:
        points.append(np.random.uniform(-0.4, 0.4, size=3))
    
    return np.array(points[:50])


def construct():
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = [
        _seed_grid(),
        _seed_fcc(),
        _seed_bcc(),
    ]
    # Add random starts
    for _ in range(3):
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
    
    best_c = None
    best_r = -np.inf
    
    # Phase 1: Lloyd equalization on each initial config
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        c_lloyd = _lloyd_optimize(c0, time_budget=5)
        r_lloyd = _compute_radius(c_lloyd)
        if r_lloyd > best_r:
            best_r = r_lloyd
            best_c = c_lloyd.copy()
    
    if best_c is None:
        best_c = inits[0].copy()
        best_r = _compute_radius(best_c)
    
    # Phase 2: Smooth polish
    if time.time() - start_time < DEADLINE - 25:
        c_smooth = _smooth_polish(best_c, alphas=[10, 20, 50], time_budget=15)
        r_smooth = _compute_radius(c_smooth)
        if r_smooth > best_r:
            best_r = r_smooth
            best_c = c_smooth.copy()
    
    # Phase 3: Repulsion annealing
    if time.time() - start_time < DEADLINE - 20:
        c_rep = _repulsion_anneal(best_c, time_budget=10)
        r_rep = _compute_radius(c_rep)
        if r_rep > best_r:
            best_r = r_rep
            best_c = c_rep.copy()
    
    # Phase 4: Metropolis Monte Carlo
    if time.time() - start_time < DEADLINE - 15:
        c_mc = _metropolis_mc(best_c, time_budget=15)
        r_mc = _compute_radius(c_mc)
        if r_mc > best_r:
            best_r = r_mc
            best_c = c_mc.copy()
    
    # Phase 5: Final local search
    if time.time() - start_time < DEADLINE - 5:
        c_final = _local_search(best_c, time_budget=10)
        r_final = _compute_radius(c_final)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    # Ensure the result is valid
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
