import time
import random
import math
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 330s limit
    
    num_points = N * N
    coords = np.array([(i // N, i % N) for i in range(num_points)], dtype=np.int32)
    
    # Precompute squared distance matrix
    # dist_matrix[i, j] = (coords[i] - coords[j])^2
    dx = coords[:, 0:1] - coords[1:2, 0] # This won't work directly for full matrix easily without broadcasting issues or memory blowup?
    # Let's do it properly:
    # coords shape (num_points, 2)
    # diff = coords[:, None, :] - coords[None, :, :]
    # dist2 = np.sum(diff**2, axis=-1)
    # This creates a (4096, 4096) matrix. 4096^2 = 16.7M floats. 16.7M * 8 bytes = 134 MB. Acceptable.
    
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff**2, axis=-1)
    
    # State
    selected_indices = []
    selected_set = set()
    
    # Helper to check if a point can be added
    # p_idx: int
    # Returns True if p_idx can be added to current selected_set
    def _can_add(p_idx):
        if p_idx in selected_set:
            return False
        
        if not selected_indices:
            return True
            
        # Get distances from p_idx to all selected points
        sel_arr = np.array(selected_indices, dtype=np.int32)
        d_p_to_sel = dist2[p_idx, sel_arr]
        
        # Check 1: Distances from p must be unique
        # Sort and check for duplicates
        if len(d_p_to_sel) > 1:
            # Vectorized duplicate check: sort and compare adjacent
            sorted_d = np.sort(d_p_to_sel)
            if np.any(sorted_d[1:] == sorted_d[:-1]):
                return False
                
        # Check 2: For each selected point q, the distance d(p, q) must not have appeared before in q's distance set.
        # We need to maintain dist_sets for each selected point.
        # To make this vectorized, we can store dist_sets as a list of sets? No, that's slow.
        # Alternative: Store a global "used distances" structure? No, it's per-vertex.
        
        # Let's stick to the incremental structure but optimize the inner loop if possible.
        # Actually, for the SA loop, we will do many checks.
        # Let's maintain dist_sets as a dict of sets.
        
        for q_idx in selected_indices:
            d = dist2[p_idx, q_idx]
            if d in dist_sets[q_idx]:
                return False
                
        return True

    # We need to initialize dist_sets
    dist_sets = {}
    
    def _add(p_idx):
        selected_set.add(p_idx)
        selected_indices.append(p_idx)
        dist_sets[p_idx] = set()
        
        for q_idx in selected_indices[:-1]:
            d = dist2[p_idx, q_idx]
            dist_sets[p_idx].add(d)
            dist_sets[q_idx].add(d)
            
    def _remove(p_idx):
        selected_set.remove(p_idx)
        # Remove from list
        try:
            idx = selected_indices.index(p_idx)
            selected_indices.pop(idx)
        except ValueError:
            pass
            
        for q_idx in selected_indices:
            d = dist2[p_idx, q_idx]
            dist_sets[q_idx].discard(d) # discard to avoid error if not present
            
        del dist_sets[p_idx]

    # Initialization: Greedy
    rng = np.random.default_rng(42)
    candidates = np.arange(num_points)
    rng.shuffle(candidates)
    
    for p_idx in candidates:
        if _can_add(p_idx):
            _add(p_idx)
            
    best_size = len(selected_indices)
    best_selected = selected_indices[:]
    best_set = selected_set.copy()
    best_dist_sets = {k: set(v) for k, v in dist_sets.items()}
    
    # Simulated Annealing
    # Temperature schedule
    T = 1.0
    cooling = 0.999
    
    # Precompute a random permutation for swaps? Or just use rng.
    # We will iterate until deadline
    
    # To speed up _can_add, we can try to vectorize the "d in dist_sets[q_idx]" check?
    # Hard to do with sets. But we can try to batch check multiple candidates?
    # For now, let's optimize the Python loop by reducing overhead.
    
    # Optimization: Instead of checking all selected points in _can_add, 
    # we can pre-filter candidates based on distance to a few "anchor" points?
    # Or just run the SA loop efficiently.
    
    # Let's try a different strategy for SA:
    # Move: Pick a random point p from all points.
    # If p in set: Try to remove it? No, we want to maximize size.
    # If p not in set: Try to add it. If success, accept.
    # If we fail to add, try a swap: Remove a random q from set, try to add p. If success, accept.
    
    # This is what the current code does roughly, but SA allows "bad" moves?
    # Since the constraint is hard (no isosceles), we can only accept valid moves.
    # So it's effectively a random walk on the space of valid sets, biased towards larger sets?
    # If we only accept valid moves, we never decrease size (unless we swap).
    # Swap keeps size same. Add increases size.
    # So we should primarily look for Adds. If stuck, do Swaps to rearrange.
    
    # The current code does this. The bottleneck is `_can_add`.
    # `_can_add` iterates over `selected_indices`.
    # If `selected_indices` is ~50, it's fast.
    # If it's 100, it's slower.
    
    # Let's try to improve the initialization with multiple random restarts and keep the best.
    # Then use the SA loop to improve the best.
    
    # Re-initialize with a few seeds
    best_size = 0
    best_selected = []
    best_set = set()
    best_dist_sets = {}
    
    for seed in [42, 123, 456, 789, 1011]:
        # Reset state
        selected_indices.clear()
        selected_set.clear()
        dist_sets.clear()
        
        rng_init = np.random.default_rng(seed)
        candidates = np.arange(num_points)
        rng_init.shuffle(candidates)
        
        for p_idx in candidates:
            if _can_add(p_idx):
                _add(p_idx)
                
        if len(selected_indices) > best_size:
            best_size = len(selected_indices)
            best_selected = selected_indices[:]
            best_set = selected_set.copy()
            best_dist_sets = {k: set(v) for k, v in dist_sets.items()}
            
    # Restore best state
    selected_indices[:] = best_selected
    selected_set.clear()
    selected_set.update(best_set)
    dist_sets.clear()
    for k, v in best_dist_sets.items():
        dist_sets[k] = set(v)
        
    # SA Loop
    # Use a random generator for the loop
    rng_sa = np.random.default_rng(12345)
    
    # Precompute list of all points for random selection
    all_points = np.arange(num_points)
    
    while time.time() < deadline:
        # Try to add
        added = False
        # Try a batch of adds
        for _ in range(100):
            p_idx = rng_sa.integers(0, num_points)
            if p_idx in selected_set:
                continue
            if _can_add(p_idx):
                _add(p_idx)
                added = True
                if len(selected_indices) > best_size:
                    best_size = len(selected_indices)
                    best_selected = selected_indices[:]
                    best_set = selected_set.copy()
                    best_dist_sets = {k: set(v) for k, v in dist_sets.items()}
                break # Restart batch if added? Or continue?
        
        if added:
            continue
            
        # Try a swap
        if selected_indices:
            # Remove a random point
            q_idx = selected_indices[rng_sa.integers(0, len(selected_indices))]
            _remove(q_idx)
            
            # Try to add a random point
            swap_success = False
            for _ in range(50):
                p_idx = rng_sa.integers(0, num_points)
                if p_idx in selected_set:
                    continue
                if _can_add(p_idx):
                    _add(p_idx)
                    swap_success = True
                    break
                    
            if not swap_success:
                # Revert
                _add(q_idx)
                
    # Return best
    result = []
    for idx in best_selected:
        x = idx // N
        y = idx % N
        result.append((x, y))
        
    return result
# EVOLVE-BLOCK-END
