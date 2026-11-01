import time
import numpy as np
from scipy.spatial.distance import cdist

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _generate_icosahedron_grid(num_levels=5):
    """
    Generate a high-resolution set of points on the unit sphere based on
    icosahedral symmetry. This provides a much better initial distribution
    for spherical packing problems than random or Fibonacci spirals,
    as it respects the underlying symmetry of the optimal solution.
    """
    # Golden ratio
    phi = (1 + 5 ** 0.5) / 2
    
    # Base icosahedron vertices
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1], [0, phi, -1], [0, -phi, 1], [0, -phi, -1],
        [1, 0, phi], [1, 0, -phi], [-1, 0, phi], [-1, 0, -phi],
        [phi, 1, 0], [phi, -1, 0], [-phi, 1, 0], [-phi, -1, 0]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # We will perform a search on this fixed set of directions.
    # For N=14, we need to select 14 points from this set (or a refined version).
    # However, a simple 20-point icosahedron is too coarse.
    # We generate a denser grid by subdividing faces or just using a very dense
    # Fibonacci sphere but sorting by angular distance to icosahedral vertices?
    # Actually, for N=14, the optimal structure is likely a subset of the 
    # vertices of a polyhedron or a perturbation thereof.
    # Let's generate a dense set of candidates.
    
    # Use a dense Fibonacci sphere for candidates, but we will do a 
    # combinatorial selection.
    n_candidates = 200
    i = np.arange(n_candidates)
    phi_angle = np.arccos(1 - 2 * (i + 0.5) / n_candidates)
    theta_angle = np.pi * (1 + 5 ** 0.5) * i
    
    x = np.sin(phi_angle) * np.cos(theta_angle)
    y = np.sin(phi_angle) * np.sin(theta_angle)
    z = np.cos(phi_angle)
    
    candidates = np.column_stack([x, y, z])
    return candidates

def _local_refine(pts, time_limit=2.0):
    """
    A very fast, lightweight local refinement using a simple repulsion-based 
    gradient step. This is used to polish the discrete selection.
    """
    start_time = time.time()
    pts = pts.copy()
    step_size = 0.05
    decay = 0.99
    
    while time.time() - start_time < time_limit:
        # Compute pairwise distances
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        
        # Calculate forces: repulsion for close points, attraction for far points?
        # To minimize max/min ratio, we want to equalize distances.
        # A simple heuristic: move points apart if too close, together if too far.
        # But "too far" is relative to the current min.
        
        min_d = d.min()
        max_d = d.max()
        
        if min_d < 1e-6:
            return pts
            
        forces = np.zeros_like(pts)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = d[i, j]
                
                # If dist is close to min_d, push apart strongly.
                # If dist is close to max_d, pull together slightly.
                # We want to reduce the spread (max - min).
                
                # Gradient of (max/min) is complex. 
                # Let's use a potential energy approach:
                # V = sum (1/d^2) for d < target, and (d - target)^2 for d > target?
                # Simpler: just minimize variance of distances?
                
                # Let's try a simple gradient descent on the objective (max/min)^2.
                # Since objective is non-differentiable at max/min, we use subgradients.
                # For now, a simple repulsion is safer to avoid collisions.
                
                # Repulsion force
                f_rep = diff / (dist ** 2)
                
                # Attraction force to keep them compact (prevent flying apart)
                # Pull towards center of mass? No, that's handled by centering.
                # Pull towards average distance?
                
                # Let's just use repulsion and centering.
                forces[i] += f_rep
        
        # Update
        pts += step_size * forces
        pts = pts - pts.mean(axis=0)
        
        # Re-normalize to unit sphere? 
        # The problem is in Euclidean space, not on a sphere. 
        # However, for minimal ratio, points tend to lie on a sphere or similar surface.
        # Let's project to the sphere of radius 1 to maintain scale invariance 
        # (since score is scale invariant).
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        pts = pts / norms
        
        step_size *= decay
        if step_size < 1e-6:
            break
            
    return pts

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    # 1. Generate a dense set of candidate directions
    candidates = _generate_icosahedron_grid()
    
    best_pts = None
    best_score = np.inf
    
    # 2. Combinatorial Search: Select N points from candidates
    # To make this tractable, we don't brute force all C(200, 14).
    # We use a greedy approach with random restarts and local refinement.
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() - _t0 > DEADLINE - 5:
            break
            
        # Randomly select N candidates
        # To improve quality, we can bias selection towards "spread out" points
        # but random selection from a dense sphere is a good baseline.
        indices = rng.choice(len(candidates), size=N, replace=False)
        init_pts = candidates[indices]
        
        # Center and normalize
        init_pts = init_pts - init_pts.mean(axis=0)
        norms = np.linalg.norm(init_pts, axis=1, keepdims=True)
        init_pts = init_pts / norms
        
        # Local Refinement
        refined_pts = _local_refine(init_pts, time_limit=2.0)
        
        # Evaluate
        score = _true_score(refined_pts)
        
        if score < best_score:
            best_score = score
            best_pts = refined_pts.copy()
            
        # Adaptive: if we find a very good score, we can try to perturb it
        # and refine again.
        if best_score < 2.0: # Arbitrary threshold for "good"
             # Try a few more perturbations around the best found
             for _ in range(5):
                 if time.time() - _t0 > DEADLINE - 2:
                     break
                 perturbed = best_pts + rng.normal(0, 0.1, size=(N, D))
                 # Project to sphere
                 perturbed = perturbed - perturbed.mean(axis=0)
                 norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
                 perturbed = perturbed / norms
                 
                 refined_pert = _local_refine(perturbed, time_limit=1.0)
                 score_pert = _true_score(refined_pert)
                 if score_pert < best_score:
                     best_score = score_pert
                     best_pts = refined_pert.copy()

    # 3. Final Polish: A more careful local optimization on the best candidate
    if best_pts is not None:
        # Use a slightly longer refinement time for the final best
        final_pts = _local_refine(best_pts, time_limit=5.0)
        final_score = _true_score(final_pts)
        if final_score < best_score:
            best_pts = final_pts
            best_score = final_score

    # Fallback if something went wrong
    if best_pts is None:
        # Return a simple regular grid or sphere
        i = np.arange(N)
        phi = np.arccos(1 - 2*(i+0.5)/N)
        theta = np.pi*(1+5**0.5)*i
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)
        best_pts = np.column_stack([x, y, z])

    return best_pts
