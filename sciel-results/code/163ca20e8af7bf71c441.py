"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Use a Direct Search / Simulated Annealing approach with a hard minimum-area objective.
   This avoids the smoothing bias of surrogate functions and the local minima traps of
   gradient-based methods on non-convex landscapes.
2. The objective is the exact minimum area of any triangle formed by 3 points.
3. We use a multi-start strategy:
   a. Greedy initialization (max-min distance).
   b. Structured initialization (symmetric patterns).
   c. Random initialization.
4. Simulated Annealing:
   - Perturb a random point by a random vector.
   - Project back into the triangle if necessary.
   - Accept if improvement, or with probability exp(delta / T) if worse.
   - Cool down temperature over time.
5. Final refinement:
   - Take the best configuration found by SA.
   - Run Nelder-Mead on the hard min-area objective for high-precision local optimization.
   - This step is crucial to polish the solution to machine precision.
"""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        # Clamp x to [0, 1]
        x = np.clip(x, 0.0, 1.0)
        
        # Calculate max y for given x
        # Triangle vertices: (0,0), (1,0), (0.5, S3H)
        # Left edge: y = sqrt(3) * x
        # Right edge: y = sqrt(3) * (1 - x)
        
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        
        # Clamp y to [0, max_y]
        y = np.clip(y, 0.0, max_y)
        
        pts[i] = [x, y]
    return pts

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    """
    pts = []
    p0 = np.array([0.5, S3H/2]) # Start near center
    pts.append(p0)
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        # Sample many candidates to find a good spot
        for _ in range(1000):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            x = u + 0.5 * v
            y = S3H * v
            cand = np.array([x, y])
            
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def _structured_init():
    """Create a structured initialization based on known Heilbronn-like configurations."""
    # Symmetric configuration:
    # 1 center
    # 6 points forming a hexagon around center
    # 4 points near corners? Or just 10 points in a grid?
    # Let's try a simple grid-like structure projected to triangle
    
    # A better structured init:
    # 3 vertices
    # 3 edge midpoints
    # 3 points on medial triangle vertices
    # 2 interior points
    
    pts = []
    # Vertices
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.5, S3H])
    
    # Edge midpoints
    pts.append([0.5, 0.0])
    pts.append([0.25, S3H/2])
    pts.append([0.75, S3H/2])
    
    # Medial triangle vertices (centroids of corner sub-triangles)
    # Centroid of (0,0)-(0.5,0)-(0.25,S3H/2) -> (0.25, S3H/6)
    pts.append([0.25, S3H/6])
    pts.append([0.75, S3H/6])
    pts.append([0.5, S3H/3])
    
    # Two more interior points
    pts.append([0.5, S3H/4])
    pts.append([0.5, S3H/2])
    
    return np.array(pts[:N])

def _random_init(rng):
    """Generate random points inside the triangle."""
    pts = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        while u + v > 1:
            u = rng.random()
            v = rng.random()
        x = u + 0.5 * v
        y = S3H * v
        pts.append([x, y])
    return np.array(pts)

def _simulated_annealing(init_pts, rng, time_budget):
    """
    Perform Simulated Annealing to maximize the minimum triangle area.
    """
    pts = init_pts.copy()
    current_score = _min_triangle_area(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Initial temperature
    T = 0.01
    cooling_rate = 0.9995
    
    # Step size for perturbations
    step_size = 0.1
    
    while time.time() - start_time < time_budget:
        # Perturb a random point
        idx = rng.randint(0, N - 1)
        new_pts = pts.copy()
        
        # Random direction and magnitude
        angle = rng.uniform(0, 2 * math.pi)
        magnitude = rng.uniform(0, step_size)
        dx = magnitude * math.cos(angle)
        dy = magnitude * math.sin(angle)
        
        new_pts[idx][0] += dx
        new_pts[idx][1] += dy
        
        # Project back to triangle
        new_pts = _project_to_triangle(new_pts)
        
        new_score = _min_triangle_area(new_pts)
        
        # Acceptance criterion
        if new_score > current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    pts = new_pts
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
        # Adjust step size: decrease as we cool
        step_size *= 0.9999
        
    return best_pts, best_score

def _refine_with_nelder_mead(pts, time_budget):
    """
    Refine the solution using Nelder-Mead on the hard min-area objective.
    """
    start_time = time.time()
    
    def nm_obj(pts_flat):
        pts_tmp = pts_flat.reshape(N, 2)
        pts_tmp = _project_to_triangle(pts_tmp)
        return -_min_triangle_area(pts_tmp)
    
    try:
        result = minimize(
            nm_obj,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        
        if result.success:
            pts_opt = result.x.reshape(N, 2)
            pts_opt = _project_to_triangle(pts_opt)
            
            if _check_inside(pts_opt):
                score = _min_triangle_area(pts_opt)
                return pts_opt, score
    except Exception:
        pass
    
    return pts, _min_triangle_area(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Multi-start Simulated Annealing
    # We will run SA from several different initializations
    
    initializations = []
    
    # 1. Greedy
    try:
        p_greedy = _greedy_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
    
    # 2. Structured
    try:
        p_struct = _structured_init()
        p_struct = _project_to_triangle(p_struct)
        initializations.append(p_struct)
    except Exception:
        pass
    
    # 3. Random
    try:
        p_rand = _random_init(rng)
        initializations.append(p_rand)
    except Exception:
        pass
    
    # 4. More random starts
    for _ in range(3):
        try:
            p_rand2 = _random_init(rng)
            initializations.append(p_rand2)
        except Exception:
            pass
    
    # Allocate time budget for SA
    # We have ~145s total. Let's reserve 10s for final refinement.
    # So ~135s for SA.
    # If we have 5 initializations, we can give each ~27s.
    # But we want to be safe. Let's do 3 initializations with more time each.
    # Or just run SA until time is up, checking all inits in parallel? No, sequential.
    
    # Let's run SA for the top 3 initializations
    # Greedy is usually best. Structured is good. Random is fallback.
    
    sa_time_per_start = 20.0  # 20s per start * 3 starts = 60s
    # If we have time, we can do more starts or longer runs.
    
    for i, init_pts in enumerate(initializations[:3]):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Run SA
        sa_pts, sa_score = _simulated_annealing(init_pts, rng, sa_time_per_start)
        
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
            
    # Phase 2: Refine the best solution with Nelder-Mead
    if best_pts is not None:
        if time.time() - start_time < DEADLINE - 5:
            refined_pts, refined_score = _refine_with_nelder_mead(best_pts, 10.0)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
                
    # If no optimization succeeded, fall back to the best initial guess
    if best_pts is None:
        for init_pts in initializations:
            score = _min_triangle_area(init_pts)
            if score > best_score:
                best_score = score
                best_pts = init_pts.copy()
                
    if best_pts is None:
        best_pts = _project_to_triangle(_random_init(rng))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
