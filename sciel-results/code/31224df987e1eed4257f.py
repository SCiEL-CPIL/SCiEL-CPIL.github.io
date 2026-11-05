"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import random
import time

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def _precompute_sq_dist(all_points):
    """Precompute squared distances between all pairs."""
    pts = np.array(all_points, dtype=np.int32)
    diff = pts[:, None, :] - pts[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)
    return sq_dist


def _check_isosceles(p1, p2, p3, sq_dist, idx1, idx2, idx3):
    """Check if three points form an isosceles triangle."""
    d12 = sq_dist[idx1, idx2]
    d13 = sq_dist[idx1, idx3]
    d23 = sq_dist[idx2, idx3]
    # Degenerate case: all same point
    if d12 == 0 and d13 == 0 and d23 == 0:
        return True
    # Isosceles if any two distances are equal (and not all zero)
    if d12 == d13 and d12 != 0:
        return True
    if d12 == d23 and d12 != 0:
        return True
    if d13 == d23 and d13 != 0:
        return True
    return False


def _can_add_point(p_idx, chosen_indices, sq_dist, all_points):
    """Check if adding p_idx maintains no-isosceles property."""
    for i in range(len(chosen_indices)):
        for j in range(i + 1, len(chosen_indices)):
            a = chosen_indices[i]
            b = chosen_indices[j]
            if _check_isosceles(all_points[p_idx], all_points[a], all_points[b], 
                                sq_dist, p_idx, a, b):
                return False
    return True


def _is_maximal(chosen_indices, sq_dist, all_points):
    """Check if the set is maximal (no point can be added)."""
    chosen_set = set(chosen_indices)
    n = len(all_points)
    for p_idx in range(n):
        if p_idx in chosen_set:
            continue
        if _can_add_point(p_idx, chosen_indices, sq_dist, all_points):
            return False
    return True


def _backtrack_minimal(chosen_indices, remaining, sq_dist, all_points, deadline):
    """
    Backtracking to find a minimal extension that makes the set maximal.
    Returns the extended set or None if timeout.
    """
    if time.time() > deadline:
        return None
    
    if _is_maximal(chosen_indices, sq_dist, all_points):
        return list(chosen_indices)
    
    # Find points that can be added
    addable = []
    for p_idx in remaining:
        if _can_add_point(p_idx, chosen_indices, sq_dist, all_points):
            addable.append(p_idx)
    
    if not addable:
        # Can't add any point, but not maximal? That shouldn't happen
        return None
    
    # Try adding each addable point (sorted by some heuristic)
    # Heuristic: prefer points that block more other points
    def blocking_score(p_idx):
        count = 0
        for q_idx in remaining:
            if q_idx == p_idx:
                continue
            if _check_isosceles(all_points[p_idx], all_points[q_idx], all_points[q_idx], 
                                sq_dist, p_idx, q_idx, q_idx):
                count += 1
        return count
    
    addable.sort(key=blocking_score, reverse=True)
    
    for p_idx in addable:
        new_chosen = chosen_indices + [p_idx]
        new_remaining = [p for p in remaining if p != p_idx]
        result = _backtrack_minimal(new_chosen, new_remaining, sq_dist, all_points, deadline)
        if result is not None:
            return result
    
    return None


def _local_search_swap(chosen_indices, sq_dist, all_points, deadline):
    """
    Try to reduce size by swapping: remove one point, add another that doesn't create isosceles.
    """
    n = len(all_points)
    chosen_set = set(chosen_indices)
    improved = True
    best_size = len(chosen_indices)
    best_set = list(chosen_indices)
    
    while improved and time.time() < deadline:
        improved = False
        for i, p_idx in enumerate(list(best_set)):
            # Try removing p_idx
            without_p = [x for x in best_set if x != p_idx]
            # Check if still no-isosceles (it should be, since removing can't create isosceles)
            # Now try adding other points to see if we can keep it maximal with same or smaller size
            # Actually, we want to see if we can find a different point to replace p_idx
            # that results in a smaller or equal size maximal set
            
            # Simpler: just try to see if the set without p_idx is still maximal
            # If not, we need to add points back, potentially leading to same size
            pass
        
        # Different approach: try pairwise swaps
        for i in range(len(best_set)):
            for j in range(len(best_set)):
                if i == j:
                    continue
                p_remove = best_set[i]
                # Try adding each non-chosen point
                for p_add in range(n):
                    if p_add in best_set:
                        continue
                    # Try swap: remove p_remove, add p_add
                    new_set = [x for x in best_set if x != p_remove]
                    if p_add in new_set:
                        continue
                    new_set.append(p_add)
                    
                    # Check no-isosceles
                    valid = True
                    for a in range(len(new_set)):
                        for b in range(a + 1, len(new_set)):
                            for c in range(b + 1, len(new_set)):
                                if _check_isosceles(all_points[new_set[a]], all_points[new_set[b]], 
                                                    all_points[new_set[c]], sq_dist, 
                                                    new_set[a], new_set[b], new_set[c]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Check maximality
                    if _is_maximal(new_set, sq_dist, all_points):
                        if len(new_set) < best_size:
                            best_size = len(new_set)
                            best_set = new_set
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return best_set


def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    n_points = len(all_points)
    start = time.time()
    deadline = start + 290  # safety margin before 330s kill
    
    # Precompute squared distances
    sq_dist = _precompute_sq_dist(all_points)
    
    best = None
    best_size = n_points + 1
    
    random.seed(42)
    
    # Strategy 1: Random greedy with different seeds
    for attempt in range(300):
        if time.time() > deadline - 60:
            break
        order = list(range(n_points))
        random.shuffle(order)
        
        chosen = []
        for p_idx in order:
            if _can_add_point(p_idx, chosen, sq_dist, all_points):
                chosen.append(p_idx)
        
        if _is_maximal(chosen, sq_dist, all_points):
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = chosen
    
    # Strategy 2: Try to improve best via local search
    if best is not None and time.time() < deadline - 30:
        improved = _local_search_swap(best, sq_dist, all_points, deadline - 10)
        if len(improved) < best_size:
            best_size = len(improved)
            best = improved
    
    # Strategy 3: Backtracking from small seeds
    if time.time() < deadline - 40:
        # Try starting from various small initial sets
        initial_seeds = []
        
        # Single points
        for p in range(n_points):
            initial_seeds.append([p])
        
        # Pairs that don't form isosceles with themselves (always true for distinct points)
        for i in range(0, n_points, 10):  # Sample pairs
            for j in range(i + 1, min(i + 20, n_points)):
                initial_seeds.append([i, j])
        
        random.shuffle(initial_seeds)
        
        for seed in initial_seeds[:50]:
            if time.time() > deadline - 40:
                break
            
            # Check seed is valid (no isosceles)
            valid_seed = True
            for i in range(len(seed)):
                for j in range(i + 1, len(seed)):
                    if _check_isosceles(all_points[seed[i]], all_points[seed[j]], 
                                        all_points[seed[j]], sq_dist, 
                                        seed[i], seed[j], seed[j]):
                        valid_seed = False
                        break
                if not valid_seed:
                    break
            
            if not valid_seed:
                continue
            
            # Use backtracking to extend minimally
            remaining = [p for p in range(n_points) if p not in seed]
            result = _backtrack_minimal(seed, remaining, sq_dist, all_points, deadline - 40)
            
            if result is not None and _is_maximal(result, sq_dist, all_points):
                if len(result) < best_size:
                    best_size = len(result)
                    best = result
    
    # Strategy 4: More random attempts with time remaining
    while time.time() < deadline - 5:
        order = list(range(n_points))
        random.shuffle(order)
        
        chosen = []
        for p_idx in order:
            if _can_add_point(p_idx, chosen, sq_dist, all_points):
                chosen.append(p_idx)
        
        if _is_maximal(chosen, sq_dist, all_points):
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = chosen
    
    if best is None:
        # Fallback: simple greedy
        order = list(range(n_points))
        best = []
        for p_idx in order:
            if _can_add_point(p_idx, best, sq_dist, all_points):
                best.append(p_idx)
    
    return [all_points[i] for i in best]
# EVOLVE-BLOCK-END
