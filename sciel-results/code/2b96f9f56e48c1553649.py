import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay, Voronoi

N = 26
DEADLINE = 145.0  # Safety margin before 150s kill

def _validity_check(pos, radii, tol=1e-8):
    """Check if configuration is valid."""
    for i in range(N):
        if pos[i][0] - radii[i] < -tol or pos[i][0] + radii[i] > 1 + tol:
            return False
        if pos[i][1] - radii[i] < -tol or pos[i][1] + radii[i] > 1 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _enforce_validity(pos, radii):
    """Make configuration valid by shrinking radii if necessary."""
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp positions
    pos = np.clip(pos, 0, 1)
    
    # Clamp radii to wall distances
    for i in range(N):
        wall_dist = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        radii[i] = min(radii[i], max(wall_dist, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(500):
        overlap_found = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                overlap = radii[i] + radii[j] - d
                if overlap > 1e-10:
                    overlap_found = True
                    # Shrink the smaller circle
                    if radii[i] <= radii[j]:
                        shrink = min(overlap, radii[i] - 1e-6)
                        radii[i] -= shrink
                        rem = overlap - shrink
                        if rem > 0:
                            radii[j] -= min(rem, radii[j] - 1e-6)
                    else:
                        shrink = min(overlap, radii[j] - 1e-6)
                        radii[j] -= shrink
                        rem = overlap - shrink
                        if rem > 0:
                            radii[i] -= min(rem, radii[i] - 1e-6)
        if not overlap_found:
            break
    
    # Final wall check
    for i in range(N):
        wall_dist = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        radii[i] = min(radii[i], max(wall_dist, 1e-6))
    
    return pos, radii

def _score_valid(pos, radii):
    """Return score if valid, else -inf."""
    if _validity_check(pos, radii):
        return np.sum(radii)
    return -np.inf

def _de_objective(params):
    """
    Objective for Differential Evolution.
    Minimizes negative sum of 'available space' for each circle.
    """
    pos = params.reshape(N, 2)
    total_space = 0.0
    for i in range(N):
        # Distance to walls
        d_wall = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        min_space = d_wall
        
        # Distance to other circles
        for j in range(N):
            if i == j:
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            # We approximate the available radius by half the distance to the nearest neighbor
            # This encourages uniform spacing
            if 0.5 * d < min_space:
                min_space = 0.5 * d
                
        total_space += min_space
        
    return -total_space

def _optimize_positions(pos, radii, deadline, max_iter=500):
    """Optimize positions to maximize sum of radii (fixed) subject to constraints."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    current_radii = radii.copy()
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Try to grow radii slightly
        growth = 1.001
        trial_radii = current_radii * growth
        
        if _validity_check(pos, trial_radii):
            current_radii = trial_radii
            if np.sum(current_radii) > best_score:
                best_score = np.sum(current_radii)
                best_pos = pos.copy()
                best_radii = current_radii.copy()
            continue
        
        # If not valid, optimize positions to make it valid
        x0 = pos.flatten()
        bounds = [(0, 1)] * (2 * N)
        weight = 500.0
        
        try:
            res = minimize(
                _penalty_objective,
                x0,
                args=(current_radii, weight),
                jac=lambda p: _penalty_grad(p, current_radii, weight),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_pos = res.x.reshape(N, 2)
            
            if _validity_check(new_pos, current_radii):
                pos = new_pos
                if np.sum(current_radii) > best_score:
                    best_score = np.sum(current_radii)
                    best_pos = pos.copy()
                    best_radii = current_radii.copy()
            elif _validity_check(pos, current_radii):
                pass
            else:
                pos, current_radii = _enforce_validity(pos, current_radii)
                if np.sum(current_radii) > best_score:
                    best_score = np.sum(current_radii)
                    best_pos = pos.copy()
                    best_radii = current_radii.copy()
                    
        except Exception:
            pos, current_radii = _enforce_validity(pos, current_radii)
            if np.sum(current_radii) > best_score:
                best_score = np.sum(current_radii)
                best_pos = pos.copy()
                best_radii = current_radii.copy()
    
    return best_pos, best_radii, best_score

def _simulated_annealing(pos, radii, deadline, max_iter=1000):
    """Simulated annealing to escape local minima."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    current_pos = pos.copy()
    current_radii = radii.copy()
    
    temp = 0.01
    cooling = 0.995
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        perturbation = np.random.normal(0, temp, size=current_pos.shape)
        new_pos = current_pos + perturbation
        new_pos = np.clip(new_pos, 0.01, 0.99)
        
        growth = 1.002
        trial_radii = current_radii * growth
        
        new_score = _score_valid(new_pos, trial_radii)
        current_score = _score_valid(current_pos, current_radii)
        
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / max(temp, 1e-8)):
            current_pos = new_pos
            current_radii = trial_radii
            if new_score > best_score:
                best_score = new_score
                best_pos = current_pos.copy()
                best_radii = current_radii.copy()
        else:
            current_pos, current_radii = _enforce_validity(current_pos, current_radii)
        
        temp *= cooling
        if temp < 0.001:
            break
    
    return best_pos, best_radii, best_score

def _penalty_objective(params, radii, weight):
    pos = params.reshape(N, 2)
    score = np.sum(radii)
    penalty = 0.0
    
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            overlap = radii[i] + radii[j] - d
            if overlap > 0:
                penalty += overlap**2
    
    for i in range(N):
        pen = max(0, radii[i] - pos[i][0])
        penalty += pen**2
        pen = max(0, pos[i][0] + radii[i] - 1)
        penalty += pen**2
        pen = max(0, radii[i] - pos[i][1])
        penalty += pen**2
        pen = max(0, pos[i][1] + radii[i] - 1)
        penalty += pen**2
    
    return -score + weight * penalty

def _penalty_grad(params, radii, weight):
    pos = params.reshape(N, 2)
    grad = np.zeros_like(params)
    
    for i in range(N):
        for j in range(i+1, N):
            d_vec = pos[i] - pos[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-10:
                d = 1e-10
                d_vec = np.array([1e-10, 0])
            overlap = radii[i] + radii[j] - d
            if overlap > 0:
                force = 2 * weight * overlap * (d_vec / d)
                grad[i*2:i*2+2] += force
                grad[j*2:j*2+2] -= force
    
    for i in range(N):
        pen = radii[i] - pos[i][0]
        if pen > 0:
            grad[i*2] += 2 * weight * pen
        pen = pos[i][0] + radii[i] - 1
        if pen > 0:
            grad[i*2] -= 2 * weight * pen
        pen = radii[i] - pos[i][1]
        if pen > 0:
            grad[i*2+1] += 2 * weight * pen
        pen = pos[i][1] + radii[i] - 1
        if pen > 0:
            grad[i*2+1] -= 2 * weight * pen
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # 1. Global Search with Differential Evolution
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Allocate ~60s for DE
    # We use a fixed seed for the DE itself to be reproducible, but the result is stochastic enough
    try:
        res_de = differential_evolution(
            _de_objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=False
        )
        de_pos = res_de.x.reshape(N, 2)
    except Exception:
        # Fallback to grid if DE fails
        rng = np.random.RandomState(42)
        de_pos = rng.uniform(0.1, 0.9, size=(N, 2))
        
    # Calculate initial radii for the DE result
    # Start with wall-limited radii, then enforce validity (shrink if overlaps)
    initial_radii = np.zeros(N)
    for i in range(N):
        initial_radii[i] = min(de_pos[i][0], 1-de_pos[i][0], de_pos[i][1], 1-de_pos[i][1])
    
    de_pos, de_radii = _enforce_validity(de_pos, initial_radii)
    
    # 2. Local Refinement
    pos, radii, score1 = _optimize_positions(de_pos, de_radii, deadline, max_iter=500)
    
    if score1 > best_score:
        best_score = score1
        best_pos = pos.copy()
        best_radii = radii.copy()
    
    # 3. Simulated Annealing for further exploration if time remains
    if time.time() < deadline - 10:
        pos, radii, score2 = _simulated_annealing(pos, radii, deadline, max_iter=500)
        if score2 > best_score:
            best_score = score2
            best_pos = pos.copy()
            best_radii = radii.copy()
            
    # Fallback
    if best_pos is None or not _validity_check(best_pos, best_radii):
        rng = np.random.RandomState(42)
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = np.full(N, 0.01)
        best_pos, best_radii = _enforce_validity(pos, radii)
    
    result = np.column_stack((best_pos, best_radii))
    return result
