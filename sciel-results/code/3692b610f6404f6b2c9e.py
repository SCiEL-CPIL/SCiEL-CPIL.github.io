import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    # Compute all triangle areas vectorized
    areas = []
    for i in range(N-2):
        for j in range(i+1, N-1):
            for l in range(j+1, N):
                v1 = points[j] - points[i]
                v2 = points[l] - points[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                areas.append(0.5 * abs(cross))
                
    min_area = np.min(areas)
    return min_area / hull_area

def _lattice_points(basis, offsets):
    """
    Generate points from lattice basis and offsets.
    basis: (2, 2) array - lattice basis vectors
    offsets: (N, 2) array - integer lattice coordinates for each point
    Returns: (N, 2) array of points
    """
    # points = offsets @ basis.T
    return offsets @ basis.T

def _objective_lattice(params):
    """
    Objective for lattice-based optimization.
    params: (4 + 2N,) array
      - First 4: flattened 2x2 basis matrix
      - Next 2N: flattened Nx2 offset matrix (continuous, will be rounded)
    """
    basis = params[:4].reshape(2, 2)
    offsets = params[4:].reshape(N, 2)
    
    # Round offsets to integers for lattice structure
    offsets_int = np.round(offsets)
    
    points = _lattice_points(basis, offsets_int)
    
    score = _score(points)
    if score <= 0:
        return 1.0
    return -score

def _objective_soft_lattice(params):
    """
    Soft objective: allow continuous offsets but penalize deviation from integers.
    This makes the optimization smoother.
    """
    basis = params[:4].reshape(2, 2)
    offsets = params[4:].reshape(N, 2)
    
    # Soft penalty for non-integer offsets
    int_penalty = np.sum((offsets - np.round(offsets))**2) * 0.1
    
    points = _lattice_points(basis, offsets)
    
    score = _score(points)
    if score <= 0:
        return 1.0 + int_penalty
    return -score + int_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_score = -1
    best_points = None
    
    # Strategy 1: Hexagonal lattice with subset selection
    # Standard hexagonal lattice basis
    hex_basis = np.array([[1.0, 0.0], [0.5, np.sqrt(3)/2]])
    
    # Try different lattice subsets
    # For N=14, we can try various patterns
    lattice_configs = []
    
    # Pattern 1: Compact hexagonal arrangement
    # Rows: 2, 3, 3, 3, 3 (total 14)
    config1 = np.array([
        [0, 0], [1, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [0, 3], [1, 3], [2, 3],
        [0, 4], [1, 4], [2, 4]
    ])
    lattice_configs.append(config1)
    
    # Pattern 2: More spread out
    config2 = np.array([
        [-1, -1], [0, -1], [1, -1],
        [-1, 0], [0, 0], [1, 0], [2, 0],
        [-1, 1], [0, 1], [1, 1],
        [0, 2], [1, 2], [2, 2],
        [1, 3]
    ])
    lattice_configs.append(config2)
    
    # Pattern 3: Diamond-like
    config3 = np.array([
        [0, -2],
        [-1, -1], [1, -1],
        [-2, 0], [0, 0], [2, 0],
        [-1, 1], [1, 1],
        [0, 2],
        [-1, 2], [1, 2],
        [0, 3],
        [-1, 3], [1, 3]
    ])
    lattice_configs.append(config3)
    
    # Pattern 4: Square grid 4x4 minus 2 corners
    config4 = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [0, 3], [2, 3], [3, 3]
    ])
    lattice_configs.append(config4)
    
    # Pattern 5: Hexagonal ring + center
    config5 = np.array([
        [0, 0],
        [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1],
        [2, 0], [-2, 0], [0, 2], [0, -2],
        [1, 2], [-1, 2], [2, 1]
    ])
    lattice_configs.append(config5)
    
    # Evaluate each configuration with hexagonal basis
    for config in lattice_configs:
        if time.time() > deadline - 5:
            break
        
        # Try with hexagonal basis
        points = _lattice_points(hex_basis, config)
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        
        # Try with square basis
        square_basis = np.array([[1.0, 0.0], [0.0, 1.0]])
        points_sq = _lattice_points(square_basis, config)
        s_sq = _score(points_sq)
        if s_sq > best_score:
            best_score = s_sq
            best_points = points_sq.copy()
        
        # Try with slightly rotated hexagonal basis
        for angle in [0.1, -0.1, 0.2, -0.2]:
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
            basis_rot = hex_basis @ rot.T
            points_rot = _lattice_points(basis_rot, config)
            s_rot = _score(points_rot)
            if s_rot > best_score:
                best_score = s_rot
                best_points = points_rot.copy()
    
    # Strategy 2: Optimize lattice basis for the best configuration
    if time.time() < deadline - 30:
        # Find the best config index
        best_config_idx = 0
        best_config_score = -1
        for idx, config in enumerate(lattice_configs):
            points = _lattice_points(hex_basis, config)
            s = _score(points)
            if s > best_config_score:
                best_config_score = s
                best_config_idx = idx
        
        best_config = lattice_configs[best_config_idx]
        
        # Optimize the basis vectors
        initial_params = np.concatenate([
            hex_basis.flatten(),
            best_config.flatten()
        ])
        
        try:
            res = minimize(
                _objective_soft_lattice,
                initial_params,
                method='Powell',
                options={
                    'maxiter': 200,
                    'maxfev': 5000,
                    'xtol': 1e-6,
                    'ftol': 1e-6
                }
            )
            
            if res.fun < _objective_soft_lattice(initial_params):
                cand_basis = res.x[:4].reshape(2, 2)
                cand_offsets = np.round(res.x[4:].reshape(N, 2))
                cand_points = _lattice_points(cand_basis, cand_offsets)
                cand_score = _score(cand_points)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand_points.copy()
                    
        except Exception:
            pass
    
    # Strategy 3: Local refinement of best points using push-pull
    if best_points is not None and time.time() < deadline - 10:
        current_points = best_points.copy()
        current_score = best_score
        
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            # Find the worst triangle
            min_area = float('inf')
            worst_tri = (0, 1, 2)
            for i in range(N-2):
                for j in range(i+1, N-1):
                    for l in range(j+1, N):
                        v1 = current_points[j] - current_points[i]
                        v2 = current_points[l] - current_points[i]
                        cross = v1[0]*v2[1] - v1[1]*v2[0]
                        area = 0.5 * abs(cross)
                        if area < min_area:
                            min_area = area
                            worst_tri = (i, j, l)
            
            i, j, l = worst_tri
            
            # Push vertices of worst triangle outward
            centroid = (current_points[i] + current_points[j] + current_points[l]) / 3
            for idx in [i, j, l]:
                direction = current_points[idx] - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    # Small push
                    current_points[idx] += 0.01 * direction
            
            # Try small perturbations
            rng = np.random.default_rng(iteration)
            for _ in range(10):
                trial = current_points + rng.normal(0, 0.005, size=(N, 2))
                trial_score = _score(trial)
                if trial_score > current_score:
                    current_score = trial_score
                    current_points = trial.copy()
                    break
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
    
    # Fallback: regular polygon if nothing worked
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
