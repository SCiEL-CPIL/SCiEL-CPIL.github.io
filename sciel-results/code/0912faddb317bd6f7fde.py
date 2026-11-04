"""
Strategy: Best-Fit Insertion + Simulated Annealing
- Initialize with a greedy best-fit insertion of circles.
- Optimize using Simulated Annealing with random perturbations.
- Use exact validity checks (no smooth approximations).
- Multi-start with different random seeds.
"""

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _is_valid(centers, radii):
    """
    Check if the packing is valid (no overlaps, inside square).
    Returns (sum_radii, is_valid).
    """
    n = len(radii)
    total_r = np.sum(radii)
    
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r or x > 1.0 - r or y < r or y > 1.0 - r:
            return -np.inf, False
            
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                return -np.inf, False
                
    return total_r, True

def _best_fit_insertion(n, time_limit, seed):
    """
    Greedy insertion of circles.
    Starts with empty square, inserts circles one by one.
    For each new circle, tries to place it at a position that maximizes its radius
    while maintaining non-overlap.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    centers = []
    radii = []
    
    # First circle: large, in center
    centers.append(np.array([0.5, 0.5]))
    radii.append(0.5)
    
    for k in range(1, n):
        # Try to insert a new circle
        # We want to maximize the radius of the new circle.
        # The max radius is limited by distance to walls and existing circles.
        
        # Candidate positions:
        # 1. Random points
        # 2. Points near existing circles (tangent)
        
        best_r = 0.0
        best_pos = None
        
        # Generate candidate positions
        candidates = []
        
        # Random points
        for _ in range(200):
            candidates.append(np.random.rand(2))
            
        # Points near existing circles
        for i in range(len(centers)):
            # Try points around existing circle i
            for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                for dist in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    pos = centers[i] + dist * np.array([np.cos(angle), np.sin(angle)])
                    if 0 <= pos[0] <= 1 and 0 <= pos[1] <= 1:
                        candidates.append(pos)
                        
        # For each candidate, compute max possible radius
        for pos in candidates:
            # Max radius limited by walls
            max_r = min(pos[0], 1.0 - pos[0], pos[1], 1.0 - pos[1])
            if max_r <= 0:
                continue
                
            # Max radius limited by existing circles
            for i in range(len(centers)):
                dist = np.linalg.norm(pos - centers[i])
                r_limit = dist - radii[i]
                if r_limit < max_r:
                    max_r = r_limit
                    
            if max_r > best_r:
                best_r = max_r
                best_pos = pos
        
        if best_pos is not None and best_r > 0.001:
            centers.append(best_pos)
            radii.append(best_r)
        else:
            # Fallback: tiny circle at random position
            pos = np.random.rand(2)
            r = 0.001
            centers.append(pos)
            radii.append(r)
            
        # Check time
        if time.time() > time_limit:
            break
            
    # Pad with tiny circles if we didn't reach n
    while len(centers) < n:
        pos = np.random.rand(2)
        r = 0.001
        centers.append(pos)
        radii.append(r)
        
    return np.array(centers[:n]), np.array(radii[:n])

def _sa_packing(centers, radii, time_limit, seed):
    """
    Simulated Annealing for circle packing.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n = len(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_score, _ = _is_valid(best_centers, best_radii)
    
    current_centers = centers.copy()
    current_radii = radii.copy()
    current_score, _ = _is_valid(current_centers, current_radii)
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit and T > T_min:
        # Propose a move
        move_type = random.choice(['jiggle', 'grow', 'shrink', 'swap'])
        
        new_centers = current_centers.copy()
        new_radii = current_radii.copy()
        
        i = random.randint(0, n-1)
        
        if move_type == 'jiggle':
            # Move circle i slightly
            delta = np.random.randn(2) * 0.05 * T
            new_centers[i] = current_centers[i] + delta
        elif move_type == 'grow':
            # Grow circle i
            new_radii[i] = current_radii[i] * (1 + 0.1 * T)
        elif move_type == 'shrink':
            # Shrink circle i
            new_radii[i] = current_radii[i] * (1 - 0.1 * T)
        elif move_type == 'swap':
            # Swap radii of two circles
            j = random.randint(0, n-1)
            if i != j:
                new_radii[i], new_radii[j] = new_radii[j], new_radii[i]
        
        # Evaluate new state
        new_score, new_valid = _is_valid(new_centers, new_radii)
        
        if new_valid:
            if new_score > current_score:
                # Always accept improvement
                current_centers = new_centers
                current_radii = new_radii
                current_score = new_score
                
                if current_score > best_score:
                    best_centers = current_centers.copy()
                    best_radii = current_radii.copy()
                    best_score = current_score
            else:
                # Accept worse solution with probability exp(-delta/T)
                delta = current_score - new_score
                if delta < 0:
                    delta = 0
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_centers = new_centers
                    current_radii = new_radii
                    current_score = new_score
        else:
            # Invalid solution: accept with very low probability
            prob = np.exp(-10 / T)
            if random.random() < prob:
                current_centers = new_centers
                current_radii = new_radii
                current_score = -np.inf
        
        # Cool down
        T *= cooling
        
    return best_centers, best_radii

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    best_centers = None
    best_radii = None
    best_score = -np.inf
    
    # Multi-start
    for trial in range(5):
        if time.time() > deadline:
            break
            
        seed = trial * 123 + 7
        
        # Initialize with best-fit insertion
        centers, radii = _best_fit_insertion(N, time_limit=10.0, seed=seed)
        
        # Optimize with simulated annealing
        c_opt, r_opt = _sa_packing(centers, radii, time_limit=20.0, seed=seed)
        
        score, valid = _is_valid(c_opt, r_opt)
        
        if valid and score > best_score:
            best_score = score
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
            
        # If we have time, do a second SA pass on the best found so far
        if best_centers is not None and time.time() < deadline - 15:
            c_opt2, r_opt2 = _sa_packing(best_centers, best_radii, time_limit=10.0, seed=seed+1000)
            score2, valid2 = _is_valid(c_opt2, r_opt2)
            if valid2 and score2 > best_score:
                best_score = score2
                best_centers = c_opt2.copy()
                best_radii = r_opt2.copy()
                
    if best_centers is None:
        # Fallback: grid
        side = 5
        best_centers = np.zeros((N, 2))
        best_radii = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            best_centers[i, 0] = (gx + 0.5) / side
            best_centers[i, 1] = (gy + 0.5) / side
            best_radii[i] = 0.5 / side - 0.01
            
    # Final validity check and repair
    result_c = best_centers.copy()
    result_r = best_radii.copy()
    
    for i in range(N):
        x, y, ri = result_c[i, 0], result_c[i, 1], result_r[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if ri > max_r:
            ri = max(0.001, max_r)
        result_r[i] = ri
        result_c[i, 0] = np.clip(x, ri, 1.0 - ri)
        result_c[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    # Resolve overlaps by shrinking smaller circles
    for _ in range(20):
        modified = False
        for i in range(N):
            for j in range(i+1, N):
                dx = result_c[i, 0] - result_c[j, 0]
                dy = result_c[i, 1] - result_c[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = result_r[i] + result_r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    if result_r[i] < result_r[j]:
                        result_r[i] = max(0.001, result_r[i] - overlap/2)
                    else:
                        result_r[j] = max(0.001, result_r[j] - overlap/2)
                    modified = True
        if not modified:
            break
            
    result = np.column_stack([result_c, result_r])
    
    return result
# EVOLVE-BLOCK-END
