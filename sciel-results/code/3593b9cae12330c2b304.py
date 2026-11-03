import numpy as np
import time
import random

N = 21
PERIMETER = 4.0

def _max_radius_for_circle(i, pos, radii, w, h):
    """
    Calculate the maximum radius for circle i at pos[i] 
    given fixed positions and radii of other circles.
    """
    x, y = pos[i]
    # Wall constraints
    r_max = min(x, w - x, y, h - y)
    
    if r_max <= 0:
        return 0.0

    # Circle-circle constraints
    for j in range(N):
        if i == j:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        # The circle i can grow until it touches circle j
        # dist = r_i + r_j  => r_i = dist - r_j
        r_allowed = dist - radii[j]
        if r_allowed < r_max:
            r_max = r_allowed
    
    if r_max < 0:
        return 0.0
    return r_max

def _update_all_radii(pos, radii, w, h):
    """
    Update all radii to their maximum valid values given current positions.
    This is done iteratively because increasing one radius might restrict others.
    We iterate until convergence or a max number of steps.
    """
    for _ in range(10):
        changed = False
        for i in range(N):
            new_r = _max_radius_for_circle(i, pos, radii, w, h)
            if abs(new_r - radii[i]) > 1e-9:
                radii[i] = new_r
                changed = True
        if not changed:
            break
    return np.sum(radii)

def _local_search_positions(pos, radii, w, h, steps, step_size, time_limit):
    """
    Perform local search on positions to maximize sum of radii.
    We perturb positions, then recalculate max radii.
    """
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    start_time = time.time()
    
    for step in range(steps):
        if time.time() > time_limit:
            break
            
        # Generate a perturbed position
        # We only move one circle at a time to ensure stability
        i = random.randint(0, N - 1)
        dx = random.uniform(-step_size, step_size)
        dy = random.uniform(-step_size, step_size)
        
        new_pos = pos.copy()
        new_pos[i, 0] += dx
        new_pos[i, 1] += dy
        
        # Clamp to walls (centers must be inside, radius can be 0 if center is on wall, 
        # but usually we want centers strictly inside to allow some radius)
        # Actually, if center is outside, radius is 0. 
        # It's better to clamp center to [0, w] and [0, h] so radius can be non-zero.
        # But if we clamp, we might get stuck on walls. 
        # Let's allow centers to go slightly out? No, radius becomes 0.
        # Standard approach: Clamp center to [0, w] and [0, h].
        new_pos[i, 0] = np.clip(new_pos[i, 0], 0, w)
        new_pos[i, 1] = np.clip(new_pos[i, 1], 0, h)
        
        # Update radii for this new configuration
        new_radii = radii.copy()
        # First, update the moved circle's radius
        new_radii[i] = _max_radius_for_circle(i, new_pos, new_radii, w, h)
        
        # Then, propagate changes to others (simple one-pass update for speed, 
        # or full convergence if time permits? One-pass is faster, might be less optimal locally 
        # but the next step will fix it. Let's do a quick full update for the moved circle's neighbors?
        # Actually, just doing a full _update_all_radii is safer and still fast for N=21.
        _update_all_radii(new_pos, new_radii, w, h)
        
        new_score = np.sum(new_radii)
        
        if new_score > best_score:
            best_score = new_score
            best_pos = new_pos
            best_radii = new_radii
            pos = new_pos
            radii = new_radii
            # Shrink step size slightly on success? Or keep it? 
            # Let's keep it constant for simplicity, maybe reduce if we want to converge faster.
            # step_size *= 0.99 
        else:
            # If we fail, we can try a smaller step or just continue
            pass
            
    return best_pos, best_radii, best_score

def _initialize_pos(w, h):
    """
    Initialize circle positions randomly within the rectangle.
    """
    pos = np.random.rand(N, 2) * np.array([w, h])
    radii = np.zeros(N)
    # Initial radii can be small, _update_all_radii will fix them
    return pos, radii

def _solve_for_aspect(w, h, time_limit):
    """
    Solve the packing problem for a specific rectangle aspect ratio.
    Returns best (pos, radii, score).
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    # Number of restarts depends on available time
    # Estimate time per restart
    # Local search takes most time.
    # Let's try 3-5 restarts if time permits.
    
    num_restarts = 5
    time_per_restart = (time_limit - 1) / num_restarts
    
    for r in range(num_restarts):
        restart_deadline = time.time() + time_per_restart
        if time.time() > time_limit - 0.5:
            break
            
        pos, radii = _initialize_pos(w, h)
        
        # Initial radius update
        _update_all_radii(pos, radii, w, h)
        score = np.sum(radii)
        
        # Local search
        # Start with larger steps, reduce over time
        step_size = 0.1 * min(w, h)
        
        # Run local search for a fraction of the restart time
        search_deadline = restart_deadline - 0.1
        pos, radii, score = _local_search_positions(pos, radii, w, h, 1000, step_size, search_deadline)
        
        if score > best_score:
            best_score = score
            best_pos = pos
            best_radii = radii
            
    if best_pos is None:
        # Fallback: single random
        pos, radii = _initialize_pos(w, h)
        _update_all_radii(pos, radii, w, h)
        best_score = np.sum(radii)
        best_pos = pos
        best_radii = radii
        
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_global_score = 0
    best_global_result = None
    
    # Aspect ratios: w + h = 2
    # w ranges from 0.5 to 1.5
    # Square is w=1.0
    # Let's sample a range of aspect ratios.
    # Based on typical circle packing, near-square is often optimal, 
    # but slight elongation can help for odd numbers of circles.
    
    # We will spend more time on promising aspect ratios.
    # Strategy: 
    # 1. Quick scan of many aspect ratios with few restarts.
    # 2. Refine the best few with more restarts.
    
    # Phase 1: Coarse Scan
    aspect_ratios = np.linspace(0.6, 1.4, 17)
    # Ensure 1.0 is included
    if 1.0 not in aspect_ratios:
        aspect_ratios = np.insert(aspect_ratios, 0, 1.0)
    
    results = []
    
    time_per_scan = (deadline - start_time - 20) / len(aspect_ratios) # Leave 20s for refinement
    
    for ar in aspect_ratios:
        if time.time() > deadline - 25:
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        # In scan phase, use fewer restarts and shorter search
        # _solve_for_aspect uses fixed restart count, so we need to adapt it or make a lighter version.
        # For now, let's just call it with a very tight time limit.
        
        t_start = time.time()
        # We need a way to control the "effort" inside _solve_for_aspect.
        # Let's modify _solve_for_aspect to accept a time_limit directly.
        
        pos, radii, score = _solve_for_aspect(w, h, t_start + time_per_scan)
        
        if pos is not None:
            results.append((score, w, h, pos, radii))
            
    if not results:
        # Emergency fallback
        w, h = 1.0, 1.0
        pos, radii = _initialize_pos(w, h)
        _update_all_radii(pos, radii, w, h)
        return np.column_stack([pos, radii])
        
    # Sort results by score
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Refinement
    # Take the top 3 aspect ratios and spend more time on them.
    top_results = results[:3]
    
    remaining_time = deadline - time.time() - 5
    time_per_refine = remaining_time / len(top_results) if top_results else 0
    
    for score, w, h, pos_init, radii_init in top_results:
        if time.time() > deadline - 5:
            break
            
        # We can initialize with the previous best pos/radii to jumpstart
        # Or just run _solve_for_aspect again with more time.
        # _solve_for_aspect currently does random init. 
        # Let's create a version that accepts initial pos? 
        # For simplicity, we'll just run it again. The random init might find a different local opt.
        
        pos, radii, score = _solve_for_aspect(w, h, time.time() + time_per_refine)
        
        if pos is not None and score > best_global_score:
            best_global_score = score
            best_global_result = np.column_stack([pos, radii])
            
    # If no result was updated in Phase 2 (e.g. time ran out immediately), use best from Phase 1
    if best_global_result is None and results:
        score, w, h, pos, radii = results[0]
        best_global_score = score
        best_global_result = np.column_stack([pos, radii])
        
    if best_global_result is None:
        # Absolute fallback
        w, h = 1.0, 1.0
        pos, radii = _initialize_pos(w, h)
        _update_all_radii(pos, radii, w, h)
        best_global_result = np.column_stack([pos, radii])
        
    return best_global_result

if __name__ == "__main__":
    res = construct()
    print("Score:", np.sum(res[:, 2]))
