import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    if len(pts) < 2:
        return np.inf
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _generate_fibonacci_sphere(n_points):
    """
    Generate n_points on a sphere using the Fibonacci lattice method.
    This provides a quasi-uniform distribution which is a good starting point.
    """
    i = np.arange(n_points)
    # Golden angle
    phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _greedy_subset_selection(candidates, k):
    """
    Greedily select k points from candidates to maximize the minimum pairwise distance.
    This helps find a 'spread out' configuration.
    """
    selected = [0] # Start with index 0
    min_dists = np.full(len(candidates), np.inf)
    
    for _ in range(k - 1):
        # Calculate min distance from new candidate to all selected points
        # For efficiency, we could maintain this, but for ~1000 candidates and k=14, 
        # recalculating is fast enough if done smartly.
        # Actually, let's just do a simple loop.
        
        # Distance from all unselected to all selected
        # selected_pts = candidates[selected]
        # dists = cdist(candidates, selected_pts)
        # min_to_selected = dists.min(axis=1)
        
        # Mask already selected
        # mask = np.ones(len(candidates), dtype=bool)
        # mask[selected] = False
        
        # Find the candidate with max min_to_selected among unselected
        # This is O(N * K * K) roughly. With N=1000, K=14, it's fine.
        
        min_to_selected = np.full(len(candidates), np.inf)
        for idx in selected:
            d = np.linalg.norm(candidates - candidates[idx], axis=1)
            min_to_selected = np.minimum(min_to_selected, d)
        
        min_to_selected[selected] = -1 # Mark selected as invalid
        
        next_idx = np.argmax(min_to_selected)
        selected.append(next_idx)
        
    return selected

def _local_refine_indices(candidates, selected_indices, num_iterations=50):
    """
    Try to improve the set by swapping one selected index with one unselected index
    if it improves the score (minimizes the ratio).
    """
    current_score = _true_score(candidates[selected_indices])
    n_candidates = len(candidates)
    unselected = [i for i in range(n_candidates) if i not in selected_indices]
    
    improved = True
    while improved and num_iterations > 0:
        improved = False
        # Try all swaps
        # To save time, we can sample swaps or limit iterations
        # Let's do a few random swaps per iteration or just one pass
        for i_idx in range(len(selected_indices)):
            for j_idx in unselected:
                new_selected = selected_indices.copy()
                new_selected[i_idx] = j_idx
                # Quick check: ensure distinctness (indices are unique, so points are distinct)
                score = _true_score(candidates[new_selected])
                if score < current_score - 1e-6:
                    current_score = score
                    selected_indices = new_selected
                    # Update unselected list
                    unselected.remove(j_idx)
                    unselected.append(selected_indices[i_idx])
                    improved = True
                    break
            if improved:
                break
        num_iterations -= 1
        
    return selected_indices, current_score

def _refine_coordinates(points):
    """
    Fine-tune the coordinates of the selected points using gradient descent
    on the squared ratio objective.
    """
    def obj(x):
        pts = x.reshape(-1, 3)
        # Keep points on sphere? 
        # The problem doesn't strictly require points on a sphere, but 
        # for maximizing min distance / minimizing max distance, spherical symmetry is often optimal.
        # However, allowing off-sphere movement might help break symmetries.
        # Let's project back to sphere after each step to maintain scale invariance implicitly 
        # or just optimize raw coords.
        # Actually, scaling invariance means we can fix one point or fix the scale.
        # Let's just optimize raw coordinates.
        
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        mn = d.min()
        mx = d.max()
        if mn < 1e-9:
            return 1e10
        return (mx / mn) ** 2

    # Use L-BFGS-B from a good start
    try:
        res = minimize(obj, points.flatten(), method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
        if res.success:
            best_pts = res.x.reshape(-1, 3)
            best_score = _true_score(best_pts)
            if best_score < _true_score(points):
                return best_pts
    except Exception:
        pass
    return points

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    best_score = np.inf
    best_pts = None
    
    # Strategy:
    # 1. Generate a large candidate set (Fibonacci sphere)
    # 2. Use greedy selection to find a good subset
    # 3. Use local search (swaps) on the indices to improve the subset
    # 4. Use continuous optimization to fine-tune the coordinates of the best subset
    
    # Candidate set size: 2000 points is a good balance between resolution and speed
    # cdist on 2000x2000 is fast.
    n_candidates = 2000
    candidates = _generate_fibonacci_sphere(n_candidates)
    
    # 1. Greedy Selection
    selected = _greedy_subset_selection(candidates, N)
    current_pts = candidates[selected]
    current_score = _true_score(current_pts)
    
    if current_score < best_score:
        best_score = current_score
        best_pts = current_pts.copy()
    
    # 2. Local Search on Indices
    # We have time budget. Let's do a limited number of swap passes.
    # Each pass checks all (N * (N_cand - N)) swaps.
    # 14 * 1986 ~ 27k swaps. Each swap involves cdist(14,14) which is very fast.
    # So one pass is very fast. We can do many passes.
    
    max_swap_passes = 200
    for _ in range(max_swap_passes):
        if time.time() - _t0 > DEADLINE - 10:
            break
            
        selected, score = _local_refine_indices(candidates, selected, num_iterations=1)
        if score < best_score:
            best_score = score
            best_pts = candidates[selected].copy()
            current_pts = best_pts.copy()
        else:
            # If no improvement in a full pass, break? 
            # The function above breaks after first improvement. 
            # We should check if it converged.
            # Let's modify _local_refine_indices to return a flag or just check score change.
            # For now, if score didn't improve, we might be in a local optimum of the discrete space.
            # Break to save time for coordinate refinement.
            break
            
    # 3. Continuous Refinement
    # Try to refine the best discrete solution found.
    if time.time() - _t0 < DEADLINE - 5:
        refined_pts = _refine_coordinates(best_pts)
        refined_score = _true_score(refined_pts)
        if refined_score < best_score:
            best_score = refined_score
            best_pts = refined_pts
    
    # 4. Try a few random restarts of the greedy process if time permits
    # Or just perturb the best points and re-optimize.
    if time.time() - _t0 < DEADLINE - 2:
        rng = np.random.default_rng(42)
        # Perturb best points slightly
        for _ in range(10):
            if time.time() - _t0 > DEADLINE - 2:
                break
            noise = rng.normal(0, 0.1, best_pts.shape)
            perturbed = best_pts + noise
            # Normalize to unit sphere to keep scale reasonable? 
            # Or just let optimizer handle it.
            # Let's project to sphere to avoid drift
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            perturbed = perturbed / norms
            
            refined = _refine_coordinates(perturbed)
            score = _true_score(refined)
            if score < best_score:
                best_score = score
                best_pts = refined

    # Fallback
    if best_pts is None:
        best_pts = _generate_fibonacci_sphere(N)
        
    return best_pts
