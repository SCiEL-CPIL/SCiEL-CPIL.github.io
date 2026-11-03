"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps

def _d_softplus_sq(z, eps=1e-4):
    """Derivative of softplus(z, eps)^2 with respect to z."""
    sp = _softplus(z, eps)
    sig = 1.0 / (1.0 + np.exp(-z / eps))
    return 2.0 * sp * sig

def _objective(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective for N=26 circles.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = np.maximum(params[2::3], 1e-6)
    
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
    penalty += np.sum(_softplus(-params[2::3], smooth_eps)**2)
    
    # Full pairwise overlaps (N=26 is small enough for full pairwise)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _gradient(params, penalty_weight, smooth_eps=1e-4):
    """
    Analytic gradient of the objective function.
    """
    x = params[0::3]
    y = params[1::3]
    r = np.maximum(params[2::3], 1e-6)
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # Boundary gradients
    for i in range(N):
        # Left wall: z = r - x
        z = r[i] - x[i]
        dz = _d_softplus_sq(z, smooth_eps)
        grad[i*3] += -penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz
        
        # Right wall: z = x + r - 1
        z = x[i] + r[i] - 1.0
        dz = _d_softplus_sq(z, smooth_eps)
        grad[i*3] += penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz
        
        # Bottom wall: z = r - y
        z = r[i] - y[i]
        dz = _d_softplus_sq(z, smooth_eps)
        grad[i*3+1] += -penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz
        
        # Top wall: z = y + r - 1
        z = y[i] + r[i] - 1.0
        dz = _d_softplus_sq(z, smooth_eps)
        grad[i*3+1] += penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz

    # Pairwise gradients
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    
    dz = _d_softplus_sq(overlap, smooth_eps)
    ux = dx / dist
    uy = dy / dist
    
    # Accumulate into grad
    np.add.at(grad[0::3], idx_i, -penalty_weight * dz * ux)
    np.add.at(grad[0::3], idx_j, penalty_weight * dz * ux)
    np.add.at(grad[1::3], idx_i, -penalty_weight * dz * uy)
    np.add.at(grad[1::3], idx_j, penalty_weight * dz * uy)
    np.add.at(grad[2::3], idx_i, penalty_weight * dz)
    np.add.at(grad[2::3], idx_j, penalty_weight * dz)
    
    return grad

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

def _remaining_time():
    """Get remaining time before deadline."""
    if _GLOBAL_DEADLINE is None:
        return 140.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz

def _initial_voronoi():
    """
    Voronoi-based initialization: generate random points, compute Voronoi cells,
    and set radii to the inradius of each cell (distance to closest edge).
    This provides better-informed starting radii closer to feasible upper bounds.
    """
    rng = np.random.default_rng(42)
    # Generate 26 random points
    pts = rng.uniform(0.1, 0.9, (N, 2))
    
    # Compute Voronoi cells using scipy
    from scipy.spatial import Voronoi
    
    # Add boundary points to ensure cells are bounded
    # Create a larger set of points including boundary
    boundary_pts = []
    for i in range(20):
        t = i / 20.0
        boundary_pts.append([t, 0])
        boundary_pts.append([t, 1])
        boundary_pts.append([0, t])
        boundary_pts.append([1, t])
    
    all_pts = np.vstack([pts, np.array(boundary_pts)])
    
    vor = Voronoi(all_pts)
    
    radii = []
    for i in range(N):
        # Get region for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            radii.append(0.05)
            continue
        
        # Get the vertices of the cell
        cell_vertices = vor.vertices[region]
        
        if len(cell_vertices) == 0:
            radii.append(0.05)
            continue
        
        # Compute inradius: minimum distance from point to any edge of the cell
        # An edge is defined by two consecutive vertices
        min_dist = np.inf
        px, py = pts[i]
        
        for j in range(len(cell_vertices)):
            v1 = cell_vertices[j]
            v2 = cell_vertices[(j+1) % len(cell_vertices)]
            
            # Distance from point to line segment v1-v2
            # Vector from v1 to v2
            d = v2 - v1
            d_len = np.linalg.norm(d)
            if d_len < 1e-10:
                dist = np.linalg.norm(pts[i] - v1)
            else:
                # Projection of (p-v1) onto d
                t = np.dot(pts[i] - v1, d) / (d_len * d_len)
                t = np.clip(t, 0, 1)
                closest = v1 + t * d
                dist = np.linalg.norm(pts[i] - closest)
            
            min_dist = min(min_dist, dist)
        
        # Also check distance to square boundaries
        min_dist = min(min_dist, px, 1-px, py, 1-py)
        
        radii.append(min_dist * 0.95)  # Slightly shrink to ensure feasibility
    
    result = np.zeros((N, 3))
    result[:, :2] = pts
    result[:, 2] = radii
    
    return result

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

def _initial_corner_cluster():
    """
    Specific initialization placing larger circles in corners and edges.
    """
    positions = []
    # 4 corners
    positions.append([0.1, 0.1, 0.12])
    positions.append([0.9, 0.1, 0.12])
    positions.append([0.1, 0.9, 0.12])
    positions.append([0.9, 0.9, 0.12])
    
    # 4 edge midpoints
    positions.append([0.5, 0.08, 0.10])
    positions.append([0.5, 0.92, 0.10])
    positions.append([0.08, 0.5, 0.10])
    positions.append([0.92, 0.5, 0.10])
    
    # Center
    positions.append([0.5, 0.5, 0.12])
    
    # Fill remaining with smaller circles in gaps
    rng = np.random.default_rng(123)
    while len(positions) < N:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        r = rng.uniform(0.05, 0.08)
        positions.append([x, y, r])
        
    return np.array(positions[:N])

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty and analytic gradients."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Low penalty
    penalty = 5.0
    remaining = _remaining_time()
    if remaining > 10:
        try:
            res = minimize(_objective, params, jac=_gradient, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Medium penalty
    remaining = _remaining_time()
    if remaining > 8:
        penalty = 50.0
        try:
            res = minimize(_objective, params, jac=_gradient, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: High penalty
    remaining = _remaining_time()
    if remaining > 5:
        penalty = 500.0
        try:
            res = minimize(_objective, params, jac=_gradient, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective(params, penalty, 1e-5):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Very high penalty
    remaining = _remaining_time()
    if remaining > 2:
        penalty = 5000.0
        try:
            res = minimize(_objective, params, jac=_gradient, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective(params, penalty, 1e-6):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Ultra-high penalty
    remaining = _remaining_time()
    if remaining > 1:
        penalty = 50000.0
        try:
            res = minimize(_objective, params, jac=_gradient, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7),
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < _objective(params, penalty, 1e-7):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

def _pt_mc_search(initial_states, temps, steps, rng):
    """
    Parallel Tempering Monte Carlo search.
    """
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    energies = []
    
    # Calculate initial energies
    for i, s in enumerate(states):
        x, y, r = s[:, 0], s[:, 1], s[:, 2]
        
        lx = np.maximum(0, r - x)
        ux = np.maximum(0, x + r - 1)
        ly = np.maximum(0, r - y)
        uy = np.maximum(0, y + r - 1)
        bound_pen = np.sum(lx**2 + ux**2 + ly**2 + uy**2)
        
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
        overlap_pen = np.sum(overlap**2)
        
        neg_r_pen = np.sum(np.maximum(0, -r)**2)
        
        penalty = bound_pen + overlap_pen + neg_r_pen
        energy = -np.sum(r) + 100.0 * penalty
        energies.append(energy)
        
    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()
    
    for step in range(steps):
        if _remaining_time() < 5.0:
            break
            
        i = rng.integers(0, n_replicas)
        
        new_state = states[i].copy()
        
        scale_pos = 0.01 * temps[i]
        new_state[:, 0] += rng.normal(0, scale_pos, N)
        new_state[:, 1] += rng.normal(0, scale_pos, N)
        
        scale_r = 0.005 * temps[i]
        new_state[:, 2] += rng.normal(0, scale_r, N)
        
        new_state[:, 2] = np.maximum(new_state[:, 2], 0.01)
        new_state[:, 0] = np.clip(new_state[:, 0], -0.1, 1.1)
        new_state[:, 1] = np.clip(new_state[:, 1], -0.1, 1.1)
        
        x, y, r = new_state[:, 0], new_state[:, 1], new_state[:, 2]
        
        lx = np.maximum(0, r - x)
        ux = np.maximum(0, x + r - 1)
        ly = np.maximum(0, r - y)
        uy = np.maximum(0, y + r - 1)
        bound_pen = np.sum(lx**2 + ux**2 + ly**2 + uy**2)
        
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
        overlap_pen = np.sum(overlap**2)
        
        neg_r_pen = np.sum(np.maximum(0, -r)**2)
        
        penalty = bound_pen + overlap_pen + neg_r_pen
        new_energy = -np.sum(r) + 100.0 * penalty
        
        delta_e = new_energy - energies[i]
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temps[i]):
            states[i] = new_state
            energies[i] = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_state = new_state.copy()
        
        if n_replicas > 1:
            j = rng.integers(0, n_replicas - 1)
            k = j + 1
            
            inv_t_j = 1.0 / temps[j]
            inv_t_k = 1.0 / temps[k]
            
            delta_swap = (energies[j] - energies[k]) * (inv_t_k - inv_t_j)
            
            if delta_swap < 0 or rng.random() < np.exp(-delta_swap):
                states[j], states[k] = states[k], states[j]
                energies[j], energies[k] = energies[k], energies[j]
                
                if energies[j] < best_energy:
                    best_energy = energies[j]
                    best_state = states[j].copy()
                if energies[k] < best_energy:
                    best_energy = energies[k]
                    best_state = states[k].copy()
    
    return best_state

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0

    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations, including Voronoi-based
    inits = [
        _initial_voronoi(),
        _initial_grid(),
        _initial_hex(),
        _initial_corner_cluster()
    ]
    
    # Optimize from each
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 15.0:
            break
        xyz_opt, score = _optimize_single(xyz0)
        
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(50):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.98
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
        
        if not _validate(xyz_opt):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Local refinement from the best solution found
    remaining = _remaining_time()
    if remaining > 15.0:
        rng = np.random.default_rng(42)
        for scale in [0.005, 0.01, 0.02, 0.03]:
            if remaining < 5.0:
                break
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)
            
            xyz_opt, score = _optimize_single(perturbed)
            
            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(50):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.98
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
            
            if not _validate(xyz_opt):
                continue
            
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
            remaining = _remaining_time()
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    # Parallel Tempering Monte Carlo refinement
    remaining = _remaining_time()
    if remaining > 20.0:
        rng = np.random.default_rng(99)
        replicas = [best_xyz]
        perturbed = best_xyz.copy()
        perturbed += rng.normal(0, 0.01, perturbed.shape)
        perturbed = _fix_boundaries(perturbed)
        replicas.append(perturbed)
        
        temps = [1.0, 2.0, 4.0]
        
        if len(replicas) < 3:
            rep3 = best_xyz.copy()
            rep3 += rng.normal(0, 0.02, rep3.shape)
            rep3 = _fix_boundaries(rep3)
            replicas.append(rep3)
            
        replicas = replicas[:3]
        
        steps = 50000
        
        pt_result = _pt_mc_search(replicas, temps, steps, rng)
        
        if _validate(pt_result):
            pt_score = np.sum(pt_result[:, 2])
            if pt_score > best_score:
                best_xyz = pt_result
                best_score = pt_score
    
    return best_xyz

# EVOLVE-BLOCK-END
