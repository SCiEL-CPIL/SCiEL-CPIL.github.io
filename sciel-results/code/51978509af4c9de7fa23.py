import numpy as np
import time

def _greedy_select(A, scores, N):
    """Greedy selection: pick vertices in descending score order, skip neighbors of picked."""
    is_picked = np.zeros(N, dtype=bool)
    indices = np.argsort(-scores)
    count = 0
    for v in indices:
        if not is_picked[v]:
            count += 1
            is_picked[v] = True
            is_picked[A[v]] = True
    return count, is_picked

def _local_refine(A, is_picked, N, max_iter=50):
    """Local refinement: try to add unpicked vertices that don't conflict with picked ones."""
    improved = True
    count = int(is_picked.sum())
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        # Find unpicked vertices that are not neighbors of any picked vertex
        blocked = np.zeros(N, dtype=bool)
        picked_indices = np.where(is_picked)[0]
        if len(picked_indices) > 0:
            for v in picked_indices:
                blocked[A[v]] = True
            blocked[picked_indices] = True
        
        candidates = np.where(~is_picked & ~blocked)[0]
        if len(candidates) == 0:
            break
        
        # Try adding one candidate at a time (greedy local search)
        for v in candidates:
            if not is_picked[v] and not blocked[v]:
                # Check if we can add this vertex
                can_add = True
                for u in np.where(A[v])[0]:
                    if is_picked[u]:
                        can_add = False
                        break
                if can_add:
                    is_picked[v] = True
                    count += 1
                    improved = True
                    # Update blocked
                    blocked[A[v]] = True
                    blocked[v] = True
    return count, is_picked

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin
    
    A = np.asarray(adjacency, dtype=bool)
    N = A.shape[0]
    
    # 1. Compute Features
    # Degree
    D = A.sum(axis=1).astype(float)
    
    # Sum of Neighbor Degrees
    SND = (A @ D)
    
    # Number of triangles containing each vertex
    Tri = np.diag(A @ A)
    
    # 2. Define Weight Grid
    # Strategy: low degree is good, but also want to block high-degree neighbors
    # Score = -w1*D + w2*SND - w3*Tri
    # w1 > 0 (penalize high degree), w2 > 0 (reward blocking high-degree hubs), w3 > 0 (penalize triangles)
    
    w1_vals = [0.5, 1.0, 2.0, 5.0]
    w2_vals = [0.1, 0.5, 1.0, 2.0]
    w3_vals = [0.0, 0.1, 0.5]
    
    best_size = -1
    best_scores = None
    
    # Fallback: Seed strategy (Degree only)
    fallback_scores = -D
    
    for w1 in w1_vals:
        if time.time() > DEADLINE:
            break
        for w2 in w2_vals:
            if time.time() > DEADLINE:
                break
            for w3 in w3_vals:
                if time.time() > DEADLINE:
                    break
                
                # Compute scores: low D is good, high SND is good (blocks hubs), low Tri is good
                scores = -w1 * D + w2 * SND - w3 * Tri
                
                # Greedy Selection
                count, is_picked = _greedy_select(A, scores, N)
                
                # Local refinement
                if time.time() < DEADLINE - 5.0:
                    count_refined, is_picked_refined = _local_refine(A, is_picked, N, max_iter=30)
                    if count_refined > count:
                        count = count_refined
                
                if count > best_size:
                    best_size = count
                    best_scores = scores
    
    if best_scores is None:
        return fallback_scores
        
    return best_scores
