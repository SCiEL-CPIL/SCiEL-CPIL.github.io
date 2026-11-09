"""
Strategy: Rigidity-Aware Stochastic Local Search with Active Constraint Polishing.
Unlike the seed program's global L-BFGS-B on a smooth surrogate (which tends to converge to the symmetric grid local minimum),
this approach uses a two-phase iteration:
1. Shake: Identify the 'rigid' set (spheres defining the minimum radius) and 'flexible' set.
   Apply large random perturbations to the flexible set to escape local minima.
2. Settle: Perform a short, targeted local optimization using gradient ascent on ONLY the active constraints
   (the ones currently limiting the radius). This is much faster and more directed than global gradient descent.
"""
import numpy as np
import time

N = 50
CUBE_HALF = 0.5
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _pairwise_distances(pts):
    """Compute all pairwise distances for N points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def _score_and_active(pts, eps=1e-7):
    """
    Compute the score and identify the active constraints (those within eps of the min).
    Returns: score, active_pairs (indices), active_walls (indices)
    """
    if pts.shape != (N, 3):
        return 0.0, [], []
    if not np.all(np.isfinite(pts)):
        return 0.0, [], []
    
    # Wall distances
    # dist_i_k = 0.5 - |pts[i, k]|
    wall_dists = CUBE_HALF - np.abs(pts) # shape (N, 3)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances / 2
    pdist = _pairwise_distances(pts)
    np.fill_diagonal(pdist, np.inf)
    min_pair_half = np.min(pdist) / 2.0
    
    score = min(min_wall, min_pair_half)
    
    if score <= 0:
        return 0.0, [], []
    
    # Identify active constraints
    active_walls = []
    active_pairs = []
    
    # Walls: dist < score + eps
    wall_mask = wall_dists < (score + eps)
    for i in range(N):
        for k in range(3):
            if wall_mask[i, k]:
                active_walls.append((i, k))
                
    # Pairs: dist/2 < score + eps => dist < 2*score + 2*eps
    pair_mask = pdist < (2.0 * score + 2.0 * eps)
    # Exclude diagonal
    pair_mask = pair_mask & ~np.eye(N, dtype=bool)
    
    for i in range(N):
        for j in range(i+1, N):
            if pair_mask[i, j]:
                active_pairs.append((i, j))
                
    return score, active_pairs, active_walls

def _local_polish(pts, max_iters=50, learning_rate=0.01):
    """
    Gradient ascent on the minimum radius using only active constraints.
    This is a simplified subgradient method.
    """
    current_score, active_pairs, active_walls = _score_and_active(pts)
    if current_score <= 0:
        return pts, current_score
        
    pts = pts.copy()
    
    for _ in range(max_iters):
        # Recalculate active constraints
        score, active_pairs, active_walls = _score_and_active(pts)
        
        # If no active constraints (shouldn't happen if score > 0), break
        if not active_pairs and not active_walls:
            break
            
        # Compute subgradient
        grad = np.zeros_like(pts)
        
        # Wall contributions
        for (i, k) in active_walls:
            # d(0.5 - |x|)/dx = -sign(x)
            # We want to maximize distance, so move in direction of -grad(dist) = sign(x)
            # Wait, maximize dist. grad(dist) = -sign(x). Ascent direction = -grad(dist) = sign(x).
            # Actually, if x > 0, dist = 0.5 - x. grad = -1. Ascent = +1 (move right? No, move left to increase dist? No, 0.5-x increases as x decreases).
            # Let's check: x=0.4, dist=0.1. Move x to 0.39, dist=0.11.
            # d(dist)/dx = -1. Ascent direction for dist is -(-1) = +1? No.
            # We want to increase dist. 
            # If x > 0, moving x in negative direction increases dist.
            # So gradient of dist w.r.t x is -1. 
            # We want to move in direction of +grad(dist) ? No, gradient ascent moves in direction of +grad.
            # If grad is -1, we move in -1 direction (decrease x). Correct.
            # So: grad[i, k] += -np.sign(pts[i, k])
            
            if pts[i, k] == 0:
                # Undefined, push randomly or 0
                grad[i, k] += 0 
            else:
                grad[i, k] += -np.sign(pts[i, k])
                
        # Pair contributions
        for (i, j) in active_pairs:
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            # dist/2. grad(dist/2) = 0.5 * diff / dist
            # We want to maximize dist/2.
            # For point i: move in direction +grad = 0.5 * diff / dist
            # For point j: move in direction -grad = -0.5 * diff / dist
            grad[i] += 0.5 * diff / dist
            grad[j] -= 0.5 * diff / dist
            
        # Normalize gradient to prevent steps that are too large or small
        norm = np.linalg.norm(grad)
        if norm > 1e-12:
            grad = grad / norm
            
        # Update
        new_pts = pts + learning_rate * grad
        # Project to cube
        new_pts = np.clip(new_pts, -CUBE_HALF, CUBE_HALF)
        
        new_score, _, _ = _score_and_active(new_pts)
        
        if new_score > score:
            pts = new_pts
            score = new_score
        else:
            # If it got worse, maybe try a smaller step or break?
            # For simplicity, break if we stall
            if _ > max_iters * 0.8:
                break
                
    return pts, score

def construct():
    """
    Construct 50 sphere centers using Rigidity-Aware Stochastic Local Search.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initial guess: Simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_pts = grid[:N].astype(float)
    best_score, _, _ = _score_and_active(best_pts)
    
    rng = np.random.RandomState(42)
    
    # Parameters for Simulated Annealing
    T = 0.05 # Initial temperature (scale of perturbation)
    T_min = 0.001
    cooling_rate = 0.995
    
    current_pts = best_pts.copy()
    current_score = best_score
    
    while time.time() < deadline - 2.0: # Leave 2s for final polish
        # 1. Identify Rigid vs Flexible
        score, active_pairs, active_walls = _score_and_active(current_pts)
        
        # Spheres involved in active constraints are "rigid"
        rigid_spheres = set()
        for (i, j) in active_pairs:
            rigid_spheres.add(i)
            rigid_spheres.add(j)
        for (i, k) in active_walls:
            rigid_spheres.add(i)
            
        flexible_spheres = [i for i in range(N) if i not in rigid_spheres]
        
        # 2. Shake (Perturb flexible spheres)
        if flexible_spheres:
            # Large random perturbation for flexible spheres
            # Scale perturbation by temperature
            perturbations = rng.randn(len(flexible_spheres), 3) * T
            current_pts[flexible_spheres] += perturbations
            current_pts = np.clip(current_pts, -CUBE_HALF, CUBE_HALF)
            
        # 3. Settle (Local Polish)
        # Polish the whole system, but the flexible ones will move more initially
        new_pts, new_score = _local_polish(current_pts, max_iters=20, learning_rate=0.05)
        
        # 4. Acceptance Criteria (Simulated Annealing)
        delta = new_score - current_score
        if delta > 0:
            current_pts = new_pts
            current_score = new_score
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept worse solution with probability exp(delta / T)
            if T > 1e-9:
                prob = np.exp(delta / T)
                if rng.rand() < prob:
                    current_pts = new_pts
                    current_score = new_score
            else:
                # If T is very low, only accept improvements
                pass
        
        # Cool down
        T *= cooling_rate
        
        # If we've cooled down enough, restart with a new random perturbation to escape deep local minima
        if T < T_min:
            # Restart: Perturb all points slightly
            current_pts = best_pts + rng.randn(N, 3) * 0.01
            current_pts = np.clip(current_pts, -CUBE_HALF, CUBE_HALF)
            current_score, _, _ = _score_and_active(current_pts)
            T = 0.02 # Reset temperature for new phase
            
    # Final Polish: Use a tighter local optimization on the best found
    if time.time() < deadline - 0.5:
        # Run a longer polish on the best point
        final_pts, final_score = _local_polish(best_pts, max_iters=200, learning_rate=0.01)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
            
        # One more pass with even smaller steps
        final_pts2, final_score2 = _local_polish(best_pts, max_iters=100, learning_rate=0.001)
        if final_score2 > best_score:
            best_pts = final_pts2
            best_score = final_score2

    return best_pts

# EVOLVE-BLOCK-END
