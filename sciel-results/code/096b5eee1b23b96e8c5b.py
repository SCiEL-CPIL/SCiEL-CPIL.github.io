"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

DEADLINE = 325  # seconds, safety margin before 330s kill

# Precomputation
POINTS = list(itertools.product(range(N), repeat=2))
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}
NUM_POINTS = len(POINTS)

# Precompute squared distances
xs = np.array([p[0] for p in POINTS])
ys = np.array([p[1] for p in POINTS])
dx = xs[:, np.newaxis] - xs[np.newaxis, :]
dy = ys[:, np.newaxis] - ys[np.newaxis, :]
DIST = (dx * dx + dy * dy).astype(np.int32)

def _can_add(p_idx, chosen_idxs):
    """Check if p_idx can be added to chosen_idxs without creating an isosceles triangle."""
    if len(chosen_idxs) < 2:
        return True
    
    # Distances from p to all chosen
    d_p = DIST[p_idx, chosen_idxs]
    
    # Check 1: No two chosen points are equidistant from p (apex at p)
    if len(np.unique(d_p)) < len(d_p):
        return False
    
    # For each pair (i, j) in chosen, check if dist(p,i) == dist(i,j) or dist(p,j) == dist(i,j)
    n = len(chosen_idxs)
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = DIST[chosen_idxs[i], chosen_idxs[j]]
            if d_p[i] == d_ij or d_p[j] == d_ij:
                return False
    return True

def _is_maximal(chosen_idxs):
    """Check if no point outside chosen can be added."""
    chosen_set = set(chosen_idxs)
    for p_idx in range(NUM_POINTS):
        if p_idx in chosen_set:
            continue
        if _can_add(p_idx, chosen_idxs):
            return False
    return True

def _greedy_random(rng):
    """Randomized greedy: pick points in random order, add if possible."""
    order = list(range(NUM_POINTS))
    rng.shuffle(order)
    chosen = []
    for p_idx in order:
        if _can_add(p_idx, chosen):
            chosen.append(p_idx)
    return chosen

def _try_remove(chosen_idxs, rng, max_iters=200):
    """Try to remove points while maintaining maximality."""
    chosen = list(chosen_idxs)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        indices = list(range(len(chosen)))
        rng.shuffle(indices)
        for idx in indices:
            candidate = chosen[:idx] + chosen[idx+1:]
            if _is_maximal(candidate):
                chosen = candidate
                improved = True
                break
    return chosen

def _swap_search(chosen_idxs, rng, max_iters=50):
    """Try to swap points to reduce size while maintaining maximality."""
    chosen = list(chosen_idxs)
    chosen_set = set(chosen)
    remaining = [i for i in range(NUM_POINTS) if i not in chosen_set]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if not remaining:
            break
        
        chosen_indices = list(range(len(chosen)))
        rng.shuffle(chosen_indices)
        
        for ci in chosen_indices:
            without_ci = chosen[:ci] + chosen[ci+1:]
            without_ci_set = set(without_ci)
            
            # Find points that can be added to without_ci
            addable = []
            for p_idx in remaining:
                if p_idx in without_ci_set:
                    continue
                if _can_add(p_idx, without_ci):
                    addable.append(p_idx)
            
            if not addable:
                continue
            
            rng.shuffle(addable)
            for p_idx in addable[:30]:
                candidate = without_ci + [p_idx]
                if _is_maximal(candidate):
                    chosen = candidate
                    chosen_set = set(chosen)
                    remaining = [i for i in range(NUM_POINTS) if i not in chosen_set]
                    improved = True
                    break
            
            if improved:
                break
    
    return chosen

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best = None
    best_size = NUM_POINTS + 1
    
    # Phase 1: Multiple randomized greedy starts
    while time.time() - start_time < DEADLINE * 0.25:
        candidate = _greedy_random(rng)
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
    
    # Phase 2: Local search on best and near-best candidates
    if best is not None:
        # Keep a few good candidates for diversity
        candidates = [best]
        
        while time.time() - start_time < DEADLINE * 0.85:
            # Try removal on current best
            improved = _try_remove(best, rng, max_iters=150)
            if len(improved) < best_size:
                best = improved
                best_size = len(best)
            
            # Try swap search
            swapped = _swap_search(best, rng, max_iters=40)
            if len(swapped) < best_size:
                best = swapped
                best_size = len(best)
            
            # Occasionally try a fresh random start + removal
            if time.time() - start_time < DEADLINE * 0.6:
                candidate = _greedy_random(rng)
                candidate = _try_remove(candidate, rng, max_iters=80)
                if len(candidate) < best_size:
                    best = candidate
                    best_size = len(candidate)
            
            # Try removal on other good candidates
            if len(candidates) > 1 and time.time() - start_time < DEADLINE * 0.7:
                for c in candidates[1:]:
                    if len(c) <= best_size + 2:
                        improved_c = _try_remove(c, rng, max_iters=50)
                        if len(improved_c) < best_size:
                            best = improved_c
                            best_size = len(best)
                            break
    
    # Final: ensure maximality of best result
    if best is not None and time.time() - start_time < DEADLINE - 5:
        if not _is_maximal(best):
            remaining = [i for i in range(NUM_POINTS) if i not in set(best)]
            rng.shuffle(remaining)
            for p_idx in remaining:
                if _can_add(p_idx, best):
                    best.append(p_idx)
                if _is_maximal(best):
                    break
    
    # Convert indices back to (x, y) tuples
    return [POINTS[i] for i in best] if best is not None else []
# EVOLVE-BLOCK-END
