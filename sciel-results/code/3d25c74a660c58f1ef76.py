"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 26

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps

def _objective_knn(params, penalty_weight, smooth_eps=1e-4, k_neighbors=8):
    """
    Vectorized objective using k-NN for overlap checking (faster than full pairwise).
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    
    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # k-NN overlaps
    centers = np.column_stack([x, y])
    try:
        tree = cKDTree(centers)
        # Query k nearest neighbors for each point
        dists, indices = tree.query(centers, k=min(k_neighbors+1, N))
        # dists[i][0] is 0 (self), so skip index 0
        for i in range(N):
            for j_idx in range(1, min(k_neighbors+1, N)):
                j = indices[i, j_idx]
                if j <= i:
                    continue  # avoid double-counting
                overlap = r[i] + r[j] - dists[i, j_idx]
                penalty += _softplus(overlap, smooth_eps)**2
    except Exception:
        # Fallback to full pairwise
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_full(params, penalty_weight, smooth_eps=1e-4):
    """
    Full pairwise objective for final polish (most accurate but slower).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    
    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # Full pairwise overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_delaunay(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective using Delaunay edges for overlap checking (faster).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # k-NN based overlap check (replaces Delaunay for speed)
    return _objective_knn(params, penalty_weight, smooth_eps, k_neighbors=10)

def _validate(xyz, tol=1e-7):
    """Check validity."""
    n = xyz.shape[0]
    x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            dist = np.sqrt(dx*dx+dy*dy)
            if dist < r[i]+r[j]-tol:
                return False
    return True

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz

def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5/cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5/side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i+0.5)/side
            y = (j+0.5)/side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    # Place some in corners
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    # Fill rest randomly
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])

def _initial_ring(seed=0):
    """Ring-shaped initial config."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.08]]
    for ring, n_circles, radius in [(1, 6, 0.10), (2, 10, 0.08), (3, 9, 0.07)]:
        for k in range(n_circles):
            angle = 2*np.pi*k/n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist*np.cos(angle)
            y = 0.5 + dist*np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1,0.9), rng.uniform(0.1,0.9), 0.06])
    return np.array(positions[:N])

def _compute_overlap_penalty(xyz):
    """Compute hard overlap penalty for a configuration."""
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    penalty = 0.0
    # Boundary
    for i in range(N):
        penalty += max(0, r[i] - x[i])**2
        penalty += max(0, x[i] + r[i] - 1.0)**2
        penalty += max(0, r[i] - y[i])**2
        penalty += max(0, y[i] + r[i] - 1.0)**2
        penalty += max(0, -r[i])**2
    # Pairwise
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap**2
    return penalty

def _parallel_tempering(xyz0, deadline, n_replicas=6, seed=42):
    """
    Parallel tempering (replica exchange) Monte Carlo for circle packing.
    Simultaneously optimizes positions and radii.
    """
    rng = np.random.default_rng(seed)
    # Temperature schedule: from hot to cold
    temps = np.array([50.0, 20.0, 8.0, 3.0, 1.0, 0.3])[:n_replicas]
    
    # Initialize replicas: replica 0 is the given config, others are perturbed
    replicas = []
    for rep in range(n_replicas):
        if rep == 0:
            replicas.append(xyz0.copy())
        else:
            xyz_rep = xyz0.copy()
            # Add random perturbation
            xyz_rep[:, 0] += rng.normal(0, 0.03, N)
            xyz_rep[:, 1] += rng.normal(0, 0.03, N)
            xyz_rep[:, 2] *= rng.uniform(0.8, 1.2, N)
            xyz_rep = _fix_boundaries(xyz_rep)
            replicas.append(xyz_rep)
    
    # Track best
    best_xyz = replicas[0].copy()
    best_score = np.sum(replicas[0][:, 2])
    
    n_steps = 5000  # total MC steps
    swap_interval = 20  # attempt swap every 20 steps
    
    for step in range(n_steps):
        if time.time() > deadline - 2.0:
            break
        
        # Annealing: gradually cool all replicas
        cooling_factor = 1.0 - 0.3 * (step / n_steps)
        current_temps = temps * cooling_factor
        
        # MC move for each replica
        for rep in range(n_replicas):
            xyz = replicas[rep]
            T = current_temps[rep]
            
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.4:
                # Move a single circle's position
                i = rng.integers(N)
                dx = rng.normal(0, 0.02)
                dy = rng.normal(0, 0.02)
                new_x = np.clip(xyz[i, 0] + dx, 0, 1)
                new_y = np.clip(xyz[i, 1] + dy, 0, 1)
                
                # Compute energy change
                old_energy = -np.sum(xyz[:, 2]) + 100.0 * _compute_overlap_penalty(xyz)
                xyz_new = xyz.copy()
                xyz_new[i, 0] = new_x
                xyz_new[i, 1] = new_y
                new_energy = -np.sum(xyz_new[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_new)
                
                delta_e = new_energy - old_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-6)):
                    replicas[rep] = xyz_new
            
            elif move_type < 0.7:
                # Adjust a single circle's radius
                i = rng.integers(N)
                dr = rng.normal(0, 0.01)
                new_r = np.clip(xyz[i, 2] + dr, 0.01, 0.45)
                
                old_energy = -np.sum(xyz[:, 2]) + 100.0 * _compute_overlap_penalty(xyz)
                xyz_new = xyz.copy()
                xyz_new[i, 2] = new_r
                new_energy = -np.sum(xyz_new[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_new)
                
                delta_e = new_energy - old_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-6)):
                    replicas[rep] = xyz_new
            
            else:
                # Move two circles (global move to escape local minima)
                i, j = rng.choice(N, size=2, replace=False)
                xyz_new = xyz.copy()
                # Swap positions with some randomness
                xyz_new[[i, j], 0] = xyz[[j, i], 0] + rng.normal(0, 0.01, 2)
                xyz_new[[i, j], 1] = xyz[[j, i], 1] + rng.normal(0, 0.01, 2)
                xyz_new[:, 0] = np.clip(xyz_new[:, 0], 0, 1)
                xyz_new[:, 1] = np.clip(xyz_new[:, 1], 0, 1)
                
                old_energy = -np.sum(xyz[:, 2]) + 100.0 * _compute_overlap_penalty(xyz)
                new_energy = -np.sum(xyz_new[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_new)
                
                delta_e = new_energy - old_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-6)):
                    replicas[rep] = xyz_new
        
        # Replica exchange attempt
        if step % swap_interval == 0:
            # Pick two adjacent replicas to potentially swap
            rep_a = rng.integers(n_replicas - 1)
            rep_b = rep_a + 1
            
            xyz_a = replicas[rep_a]
            xyz_b = replicas[rep_b]
            T_a = current_temps[rep_a]
            T_b = current_temps[rep_b]
            
            # Compute energies
            E_a = -np.sum(xyz_a[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_a)
            E_b = -np.sum(xyz_b[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_b)
            
            # Metropolis criterion for swap
            delta = (1.0/T_b - 1.0/T_a) * (E_b - E_a)
            if delta < 0 or rng.random() < np.exp(-delta):
                replicas[rep_a], replicas[rep_b] = replicas[rep_b], replicas[rep_a]
        
        # Track best (cold replica typically has best result)
        cold_score = np.sum(replicas[0][:, 2])
        if cold_score > best_score and _validate(replicas[0]):
            best_score = cold_score
            best_xyz = replicas[0].copy()
        
        # Also check all replicas for valid high-score configs
        for rep in range(n_replicas):
            rep_score = np.sum(replicas[rep][:, 2])
            if rep_score > best_score and _validate(replicas[rep]):
                best_score = rep_score
                best_xyz = replicas[rep].copy()
    
    return best_xyz, best_score

def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config: PT-MC then local polish."""
    best_params = xyz0.copy()
    best_score = np.sum(xyz0[:, 2])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 0: Parallel tempering MC (global search)
    remaining = deadline - time.time()
    if remaining > 20:
        pt_budget = deadline - 8.0  # leave 8s for local polish
        try:
            xyz_pt, score_pt = _parallel_tempering(xyz0, pt_budget, n_replicas=5, seed=int(hash(tuple(xyz0[:3].round(4)))) % 10000)
            if score_pt > best_score:
                best_params = xyz_pt.copy()
                best_score = score_pt
        except Exception:
            pass
    
    # Phase 1: k-NN based local optimization
    params = best_params.flatten().copy()
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 10.0
        try:
            res = minimize(_objective_knn, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, 10),
                          options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective_knn(params, penalty, 1e-4, 10):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: k-NN with higher penalty
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 50.0
        try:
            res = minimize(_objective_knn, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, 10),
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective_knn(params, penalty, 1e-4, 10):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: Full pairwise with high penalty
    remaining = deadline - time.time()
    if remaining > 3:
        penalty = 200.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective_full(params, penalty, 1e-5):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Even higher penalty
    remaining = deadline - time.time()
    if remaining > 2:
        penalty = 1000.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6),
                          options={'maxiter': 250, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective_full(params, penalty, 1e-6):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 1.5:
        penalty = 5000.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7),
                          options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 20})
            if res.fun < _objective_full(params, penalty, 1e-7):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params, best_score

def _ga_discrete_packing(deadline, seed=42):
    """
    Discrete Genetic Algorithm for circle packing.
    Represents circles by grid coordinates and discrete radii.
    """
    rng = np.random.default_rng(seed)
    
    # Grid resolution
    GRID_SIZE = 200
    # Discrete radii options (fine-grained)
    # Max radius is 0.5, but for 26 circles, typical radii are smaller.
    # Let's use a range of radii from 0.02 to 0.45 with step 0.005
    radii_options = np.arange(0.02, 0.45 + 1e-9, 0.005)
    n_radii = len(radii_options)
    
    def decode(genes):
        """Convert gene array to xyz array."""
        # genes: [gx0, gy0, ri0, gx1, gy1, ri1, ...]
        xyz = np.zeros((N, 3))
        for i in range(N):
            gx = genes[i*3]
            gy = genes[i*3+1]
            ri = int(genes[i*3+2])
            ri = np.clip(ri, 0, n_radii-1)
            xyz[i, 0] = (gx + 0.5) / GRID_SIZE
            xyz[i, 1] = (gy + 0.5) / GRID_SIZE
            xyz[i, 2] = radii_options[ri]
        return xyz
    
    def fitness(xyz):
        """Calculate fitness: sum of radii - penalties."""
        x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        score = np.sum(r)
        penalty = 0.0
        
        # Boundary check
        for i in range(N):
            if x[i] - r[i] < -1e-9 or x[i] + r[i] > 1 + 1e-9:
                penalty += 100.0
            if y[i] - r[i] < -1e-9 or y[i] + r[i] > 1 + 1e-9:
                penalty += 100.0
            if r[i] < 0:
                penalty += 100.0
        
        # Overlap check
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = r[i] + r[j] - dist
                if overlap > 1e-9:
                    penalty += overlap * 1000.0
        
        return score - penalty

    def mutate(genes, mutation_rate=0.1):
        """Randomly mutate genes."""
        new_genes = genes.copy()
        for i in range(N):
            if rng.random() < mutation_rate:
                # Mutate center
                if rng.random() < 0.5:
                    new_genes[i*3] = rng.integers(0, GRID_SIZE)
                    new_genes[i*3+1] = rng.integers(0, GRID_SIZE)
                else:
                    # Mutate radius
                    new_genes[i*3+2] = rng.integers(0, n_radii)
        return new_genes

    def crossover(parent1, parent2):
        """Uniform crossover."""
        child = np.zeros_like(parent1)
        mask = rng.random(N*3) < 0.5
        child[mask] = parent1[mask]
        child[~mask] = parent2[~mask]
        return child

    # Initialize population
    pop_size = 100
    population = []
    
    # Seed with some good heuristics converted to discrete
    heuristics = [_initial_hex(), _initial_corner_weighted(0), _initial_ring(0)]
    for h in heuristics:
        genes = np.zeros(N*3, dtype=int)
        for i in range(N):
            genes[i*3] = int(h[i, 0] * GRID_SIZE)
            genes[i*3+1] = int(h[i, 1] * GRID_SIZE)
            # Find closest radius
            ri = np.argmin(np.abs(radii_options - h[i, 2]))
            genes[i*3+2] = ri
        population.append(genes)
    
    # Fill rest with random
    while len(population) < pop_size:
        genes = rng.integers(0, GRID_SIZE, size=N*2)
        radii = rng.integers(0, n_radii, size=N)
        full_genes = np.zeros(N*3, dtype=int)
        full_genes[0::3] = genes[0::2]
        full_genes[1::3] = genes[1::2]
        full_genes[2::3] = radii
        population.append(full_genes)
    
    best_genes = None
    best_fitness = -np.inf
    best_xyz = None
    
    generations = 0
    while time.time() < deadline - 10.0:
        generations += 1
        
        # Evaluate fitness
        fits = []
        for genes in population:
            xyz = decode(genes)
            f = fitness(xyz)
            fits.append(f)
            if f > best_fitness:
                best_fitness = f
                best_genes = genes.copy()
                best_xyz = xyz.copy()
        
        # Sort population by fitness
        sorted_idx = np.argsort(fits)[::-1]
        population = [population[i] for i in sorted_idx]
        fits = [fits[i] for i in sorted_idx]
        
        # Elitism: keep top 10%
        elite_count = max(5, pop_size // 10)
        new_population = population[:elite_count].copy()
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                candidates = rng.integers(0, pop_size, size=3)
                best_cand = max(candidates, key=lambda c: fits[c])
                return population[best_cand]
            
            p1 = select()
            p2 = select()
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
        
        # Check time
        if time.time() > deadline - 15.0:
            break

    return best_xyz, best_fitness

def construct():
    global_start = time.time()
    deadline = global_start + 143.0  # 7s safety margin before 150s kill
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Phase 1: Discrete Genetic Algorithm
    # Use 70% of the time for GA
    ga_deadline = global_start + (deadline - global_start) * 0.7
    try:
        ga_xyz, ga_score = _ga_discrete_packing(ga_deadline, seed=123)
        if ga_xyz is not None:
            if not _validate(ga_xyz):
                # Try to fix invalid GA result
                ga_fix = ga_xyz.copy()
                for _ in range(50):
                    if _validate(ga_fix):
                        break
                    ga_fix[:, 2] *= 0.95
                    ga_fix = _fix_boundaries(ga_fix)
                ga_xyz = ga_fix
            
            if _validate(ga_xyz):
                ga_score = np.sum(ga_xyz[:, 2])
                if ga_score > best_score:
                    best_score = ga_score
                    best_xyz = ga_xyz.copy()
    except Exception as e:
        pass

    # Phase 2: Local Polish on the best discrete solution
    # Use remaining time for continuous optimization
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            # Use _optimize_single which does PT + L-BFGS
            # But _optimize_single has its own internal time checks.
            # We pass the current best as the initial guess.
            # To prevent it from taking too long, we adjust the deadline logic inside or just call it.
            # _optimize_single uses 'deadline' argument.
            xyz_opt, score = _optimize_single(best_xyz, deadline)
            
            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(100):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.95
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
            
            if _validate(xyz_opt):
                score = np.sum(xyz_opt[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz_opt.copy()
        except Exception:
            pass

    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
