"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 16
NUM_POINTS = N * N
POINTS = [(x, y) for x in range(N) for y in range(N)]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

# Precompute squared distances
xs = np.array([p[0] for p in POINTS])
ys = np.array([p[1] for p in POINTS])
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST2 = (dx**2 + dy**2).astype(np.int32)

# Precompute all pairs (i, j, d) where d = DIST2[i, j] > 0
# For each distance value, store the list of pairs at that distance
dist_to_pairs = defaultdict(list)
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        d = int(DIST2[i, j])
        if d > 0:
            dist_to_pairs[d].append((i, j))

# Also precompute for each point p, the distance to each other point
# dist_from_p[p] = array of length NUM_POINTS with distances from p
# But DIST2 already gives us this: DIST2[p, :]

def check_validity(mask):
    """
    Check if the set of points given by mask (boolean array of length NUM_POINTS)
    is a valid no-isosceles set.
    
    Returns True if valid, False otherwise.
    """
    selected = np.where(mask)[0]
    n_sel = len(selected)
    if n_sel <= 2:
        return True
    
    # Check 1: For each selected point p, all distances to other selected points must be unique
    for idx in range(n_sel):
        p = selected[idx]
        dists = np.array([int(DIST2[p, q]) for q in selected if q != p])
        if len(dists) != len(np.unique(dists)):
            return False
    
    # Check 2: No isosceles triangle
    # For each pair (a, b) in selected, check if there exists c in selected such that
    # dist(a,c) == dist(a,b) or dist(b,c) == dist(a,b)
    # More efficiently: for each point a in selected, and each distance d from a to some b in selected,
    # check if any other point c in selected (c != a, c != b) has dist(a,c) == d.
    # This is the same as Check 1 above actually.
    # Wait, Check 1 already ensures that for each a, all distances to other selected points are unique.
    # That means for each a, there's no c != a, c != b with dist(a,c) == dist(a,b).
    # So Check 1 covers the "vertex at a" case.
    # But we also need to check: for triangle (a, b, c), if dist(a,b) == dist(b,c), that's vertex at b.
    # Which is covered by Check 1 for point b.
    # So Check 1 is sufficient!
    
    return True

def check_maximality(mask):
    """
    Check if no point can be added to the set without creating an isosceles triangle.
    """
    selected_set = set(np.where(mask)[0].tolist())
    for p in range(NUM_POINTS):
        if p in selected_set:
            continue
        # Check if p can be added
        # Temporarily add p and check validity
        new_mask = mask.copy()
        new_mask[p] = True
        if check_validity(new_mask):
            return False
        new_mask[p] = False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 295
    
    best_size = NUM_POINTS + 1
    best_mask = None
    
    # Phase 1: Greedy construction with multiple restarts
    num_greedy_restarts = 200
    
    for restart in range(num_greedy_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7 + 13)
        
        # Random order for greedy
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        current_mask = np.zeros(NUM_POINTS, dtype=bool)
        
        for p in order:
            if time.time() > deadline:
                break
            trial = current_mask.copy()
            trial[p] = True
            if check_validity(trial):
                current_mask = trial
        
        # Phase 2: Minimization - try to remove points while preserving maximality
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                break
            
            # Try removing points in random order
            to_remove = list(np.where(current_mask)[0])
            random.shuffle(to_remove)
            
            for p in to_remove:
                if time.time() > deadline:
                    break
                trial = current_mask.copy()
                trial[p] = False
                # Check if trial is still valid and maximal
                if check_validity(trial) and check_maximality(trial):
                    current_mask = trial
                    changed = True
                    break
        
        if np.sum(current_mask) < best_size:
            best_size = int(np.sum(current_mask))
            best_mask = current_mask.copy()
    
    # Phase 3: Simulated Annealing on the mask
    if best_mask is not None:
        current_mask = best_mask.copy()
        current_size = int(np.sum(current_mask))
        best_mask_sa = current_mask.copy()
        best_size_sa = current_size
        
        # Temperature schedule
        T = 10.0
        T_min = 0.1
        alpha = 0.995
        max_steps = 500000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Choose a move: flip a random point
            p = random.randint(0, NUM_POINTS - 1)
            trial = current_mask.copy()
            trial[p] = not trial[p]
            
            trial_size = int(np.sum(trial))
            
            # Evaluate the move
            if trial[p]:
                # Adding a point: check validity
                if not check_validity(trial):
                    continue
                # Check maximality (expensive)
                # For SA, we can skip maximality check and just optimize size + validity
                # But we need the final result to be maximal
                delta = 1  # size increases, which is bad
            else:
                # Removing a point: check validity (always valid if removing)
                # But we need to ensure the remaining set is still maximal? No, that's the constraint
                # Actually, removing a point always keeps the no-isosceles property.
                # But it may break maximality.
                delta = -1  # size decreases, which is good
            
            # For SA, we only accept valid moves. Maximality is checked at the end.
            # So we just optimize size among valid configurations.
            
            if delta < 0:
                current_mask = trial
                current_size = trial_size
                if trial_size < best_size_sa:
                    best_size_sa = trial_size
                    best_mask_sa = trial.copy()
            else:
                # delta > 0 (adding), accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta / T):
                    current_mask = trial
                    current_size = trial_size
            
            T *= alpha
            if T < T_min:
                T = T_min
    
    # Final: ensure best result is maximal
    if best_mask is not None:
        if not check_maximality(best_mask):
            # Try to fix by adding points to make it maximal
            trial = best_mask.copy()
            for p in range(NUM_POINTS):
                if time.time() > deadline:
                    break
                if not trial[p]:
                    new_trial = trial.copy()
                    new_trial[p] = True
                    if check_validity(new_trial):
                        trial = new_trial
            if check_maximality(trial):
                best_mask = trial
                best_size = int(np.sum(best_mask))
    
    if best_mask is None:
        # Fallback: deterministic greedy
        current_mask = np.zeros(NUM_POINTS, dtype=bool)
        for p in range(NUM_POINTS):
            trial = current_mask.copy()
            trial[p] = True
            if check_validity(trial):
                current_mask = trial
        best_mask = current_mask
        best_size = int(np.sum(best_mask))
    
    return [POINTS[i] for i in np.where(best_mask)[0]]

# EVOLVE-BLOCK-END
