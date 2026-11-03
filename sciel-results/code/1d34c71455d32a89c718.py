"""Physics-based circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in a rectangle of perimeter 4 (i.e., width + height = 2).
    Maximize sum of radii.
    
    Approach: Vectorized Simulated Annealing with penalty constraints.
    """
    best_score = -np.inf
    best_result = None
    
    # Try a range of aspect ratios w/h
    # w + h = 2 => h = 2 - w
    # Aspect ratio a = w/h => w = a(2-w) => w(1+a) = 2a => w = 2a/(1+a)
    aspect_ratios = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    
    for aspect in aspect_ratios:
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        result = _solve_sa(w, h, N)
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
                
    # Also try optimizing the aspect ratio itself in a coarse loop
    # Or just run SA with w as a variable? 
    # Let's stick to fixed w for stability, but maybe add a few more specific values
    # e.g. 1.1, 1.3, 1.7, 1.8, 2.2
    
    extra_aspects = [1.1, 1.3, 1.7, 1.8, 2.2]
    for aspect in extra_aspects:
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        result = _solve_sa(w, h, N)
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        w, h = 1.0, 1.0
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

def _solve_sa(w, h, N, max_iter=5000, T_start=1.0, T_end=1e-4):
    """
    Vectorized Simulated Annealing for fixed w, h.
    """
    # Initialization: Grid
    side = int(np.ceil(np.sqrt(N)))
    positions = np.zeros((N, 2))
    radii = np.full(N, 0.02)
    
    for i in range(N):
        row = i // side
        col = i % side
        positions[i, 0] = (col + 0.5) * w / side
        positions[i, 1] = (row + 0.5) * h / side
        
    # Ensure initial radii don't cause massive overlaps
    # Simple scaling
    max_r = min(w, h) / side * 0.9
    radii[:] = max_r * 0.5
    
    current_pos = positions.copy()
    current_rad = radii.copy()
    current_score = _evaluate(current_pos, current_rad, w, h)
    
    best_pos = current_pos.copy()
    best_rad = current_rad.copy()
    best_score = current_score
    
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max_iter)
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        T *= cooling
        
        # Generate move: pick random circle, perturb pos and rad
        idx = rng.integers(0, N)
        
        new_pos = current_pos.copy()
        new_rad = current_rad.copy()
        
        # Perturbation magnitude
        # Scale with T? Or fixed small step?
        step_pos = 0.05 * (1.0 + T)
        step_rad = 0.01 * (1.0 + T)
        
        new_pos[idx, 0] += rng.normal(0, step_pos)
        new_pos[idx, 1] += rng.normal(0, step_pos)
        new_rad[idx] += rng.normal(0, step_rad)
        
        # Clamp radii to positive
        new_rad[idx] = max(new_rad[idx], 0.01)
        
        # Optional: Hard clamp positions to box to reduce penalty?
        # No, let penalty handle it, but clamp to [0, w] and [0, h] roughly to help convergence
        # Actually, clamping helps a lot.
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, w)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, h)
        
        new_score = _evaluate(new_pos, new_rad, w, h)
        
        # Acceptance criterion
        # We want to maximize sum(radii). 
        # _evaluate returns -sum(radii) + penalty.
        # So lower is better.
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            current_pos = new_pos
            current_rad = new_rad
            current_score = new_score
            
            if current_score < best_score: # Lower is better
                best_score = current_score
                best_pos = current_pos.copy()
                best_rad = current_rad.copy()
                
    # Final validity check
    if _is_valid(best_pos, best_rad, w, h):
        result = np.zeros((N, 3))
        result[:, 0] = best_pos[:, 0]
        result[:, 1] = best_pos[:, 1]
        result[:, 2] = best_rad
        return result
    else:
        # Try to repair or return None
        return None

def _evaluate(pos, rad, w, h):
    """
    Vectorized evaluation: -sum(rad) + penalty.
    """
    N = len(rad)
    # Objective
    obj = -np.sum(rad)
    
    # Penalty
    pen = 0.0
    
    # Overlaps
    # Compute distance matrix
    # pos is (N, 2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    min_dist = rad[:, np.newaxis] + rad[np.newaxis, :]
    
    # Overlap amount
    overlap = np.maximum(min_dist - dist, 0)
    # Exclude self-overlap (diagonal)
    np.fill_diagonal(overlap, 0)
    
    pen += np.sum(overlap**2)
    
    # Boundary
    # x < r or x > w - r
    # y < r or y > h - r
    
    # Left/Bottom
    pen += np.sum(np.maximum(rad - pos[:, 0], 0)**2)
    pen += np.sum(np.maximum(rad - pos[:, 1], 0)**2)
    
    # Right/Top
    pen += np.sum(np.maximum(pos[:, 0] - (w - rad), 0)**2)
    pen += np.sum(np.maximum(pos[:, 1] - (h - rad), 0)**2)
    
    return obj + 10.0 * pen # Weight for penalty

def _is_valid(pos, rad, w, h, tol=1e-6):
    """
    Check if packing is valid.
    """
    N = len(rad)
    
    # Boundary
    if np.any(pos[:, 0] < rad - tol) or np.any(pos[:, 0] > w - rad + tol):
        return False
    if np.any(pos[:, 1] < rad - tol) or np.any(pos[:, 1] > h - rad + tol):
        return False
        
    # Overlaps
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    min_dist = rad[:, np.newaxis] + rad[np.newaxis, :]
    
    # Check if dist < min_dist - tol for any i != j
    np.fill_diagonal(min_dist, 0)
    
    if np.any(dist < min_dist - tol):
        return False
        
    return True

# EVOLVE-BLOCK-END
