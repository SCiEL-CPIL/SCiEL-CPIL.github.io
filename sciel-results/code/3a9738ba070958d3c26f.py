"""Combined: bitmask conflict tracking + minimum-degree greedy + intensive local search.
Uses precomputed pair-conflict bitmasks for O(1) addability checks, enabling
much more aggressive local search refinement within the time budget."""

import time
import random
import itertools
import numpy as np

N = 16
DEADLINE_SECONDS = 290  # safety margin before 330s kill

_all_pts = list(itertools.product(range(N), repeat=2))
_pt_idx = {p: i for i, p in enumerate(_all_pts)}
_np = len(_all_pts)

# Precompute pair_conflict: pair_conflict[i][j] = bitmask of k where {i,j,k} is isosceles
pair_conflict = [[0] * _np for _ in range(_np)]
for i in range(_np):
    xi, yi = _all_pts[i]
    for j in range(i + 1, _np):
        xj, yj = _all_pts[j]
        d_ij = (xi - xj) ** 2 + (yi - yj) ** 2
        mask = 0
        for k in range(_np):
            if k == i or k == j:
                continue
            xk, yk = _all_pts[k]
            d_ik = (xi - xk) ** 2 + (yi - yk) ** 2
            d_jk = (xj - xk) ** 2 + (yj - yk) ** 2
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        pair_conflict[i][j] = mask
        pair_conflict[j][i] = mask


def _compute_blocked(S_indices):
    """Compute the blocked bitmask for a set S_indices."""
    blocked = 0
    n = len(S_indices)
    for i in range(n):
        for j in range(i + 1, n):
            blocked |= pair_conflict[S_indices[i]][S_indices[j]]
    return blocked


def _is_maximal(S_indices, blocked):
    """Check if no point outside S_indices can be added."""
    S_set = set(S_indices)
    for p in range(_np):
        if p not in S_set and not (blocked & (1 << p)):
            return False
    return True


def _is_valid(S_indices):
    """Verify no isosceles triangles in S_indices."""
    n = len(S_indices)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                xi, yi = _all_pts[S_indices[i]]
                xj, yj = _all_pts[S_indices[j]]
                xk, yk = _all_pts[S_indices[k]]
                d1 = (xi - xj) ** 2 + (yi - yj) ** 2
                d2 = (xi - xk) ** 2 + (yi - yk) ** 2
                d3 = (xj - xk) ** 2 + (yj - yk) ** 2
                if d1 == d2 or d1 == d3 or d2 == d3:
                    return False
    return True


def _greedy_construct(seed):
    """
    Minimum-degree greedy: at each step, pick the addable point that blocks
    the fewest other addable points. Uses bitmask for O(1) addability checks.
    """
    rng = random.Random(seed)
    S_indices = []
    blocked = 0
    
    while True:
        # Find all addable points
        addable = []
        S_set = set(S_indices)
        for p in range(_np):
            if p not in S_set and not (blocked & (1 << p)):
                addable.append(p)
        
        if not addable:
            break
        
        # Score each addable point by how many others it blocks
        best_p = None
        best_score = float('inf')
        rng.shuffle(addable)
        
        for p in addable:
            # Compute how many currently-addable points p would block
            new_blocked = 0
            for s in S_indices:
                new_blocked |= pair_conflict[p][s]
            # Count how many of the current addable set get blocked
            block_count = 0
            for q in addable:
                if q != p and (new_blocked & (1 << q)):
                    block_count += 1
            
            score = block_count + rng.random() * 0.01
            if score < best_score:
                best_score = score
                best_p = p
        
        if best_p is None:
            break
        
        S_indices.append(best_p)
        for s in S_indices[:-1]:
            blocked |= pair_conflict[best_p][s]
    
    # Ensure maximality: add any remaining valid points
    changed = True
    while changed:
        changed = False
        S_set = set(S_indices)
        for p in range(_np):
            if p not in S_set and not (blocked & (1 << p)):
                S_indices.append(p)
                for s in S_indices[:-1]:
                    blocked |= pair_conflict[p][s]
                changed = True
    
    return S_indices, blocked


def _local_search(S_indices, seed, time_limit):
    """
    Local search: try removing points and check if the set remains maximal.
    Also try 1-1 swaps to escape local optima.
    """
    rng = random.Random(seed)
    S_indices = list(S_indices)
    blocked = _compute_blocked(S_indices)
    best_S = list(S_indices)
    best_size = len(S_indices)
    start_time = time.time()
    
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        if len(S_indices) <= best_size:
            # Try removals
            S_shuffled = list(S_indices)
            rng.shuffle(S_shuffled)
            
            improved = False
            for remove_p in S_shuffled:
                if len(S_indices) - 1 >= best_size:
                    continue
                
                S_indices.remove(remove_p)
                new_blocked = _compute_blocked(S_indices)
                
                if _is_maximal(S_indices, new_blocked):
                    best_S = list(S_indices)
                    best_size = len(S_indices)
                    blocked = new_blocked
                    improved = True
                    break
                
                S_indices.append(remove_p)
            
            if improved:
                continue
        
        # Try 1-1 swaps: remove one, add another
        if len(S_indices) >= best_size:
            S_shuffled = list(S_indices)
            rng.shuffle(S_shuffled)
            
            for remove_p in S_shuffled:
                if time.time() - start_time > time_limit:
                    break
                
                S_indices.remove(remove_p)
                new_blocked = _compute_blocked(S_indices)
                
                # Find a point that can be added
                S_set = set(S_indices)
                added = False
                for q in range(_np):
                    if q not in S_set and not (new_blocked & (1 << q)):
                        S_indices.append(q)
                        for s in S_indices[:-1]:
                            new_blocked |= pair_conflict[q][s]
                        added = True
                        break
                
                if added and len(S_indices) < best_size:
                    best_S = list(S_indices)
                    best_size = len(S_indices)
                    blocked = new_blocked
                else:
                    if added:
                        S_indices.remove(S_indices[-1])
                    S_indices.append(remove_p)
                    new_blocked = _compute_blocked(S_indices)
    
    return best_S


def construct():
    start_time = time.time()
    
    best_S = None
    best_size = float('inf')
    
    # Baseline: simple row-major greedy
    S_base, _ = _greedy_construct(42)
    if S_base:
        best_S = S_base
        best_size = len(S_base)
    
    # Multiple greedy constructions with local search
    for seed in range(200):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 10:
            break
        
        S, blocked = _greedy_construct(seed)
        if not S:
            continue
        
        if len(S) < best_size:
            best_S = S
            best_size = len(S)
        
        # Apply local search with remaining time budget
        remaining = DEADLINE_SECONDS - 5 - (time.time() - start_time)
        if remaining > 3:
            time_per_search = min(remaining / max(1, 200 - seed), 15)
            S_refined = _local_search(S, seed, time_per_search)
            if len(S_refined) < best_size:
                best_size = len(S_refined)
                best_S = S_refined
    
    # Final validation
    if best_S is not None:
        if not _is_valid(best_S):
            # Fallback: rebuild from scratch with validation
            S_valid, _ = _greedy_construct(0)
            if S_valid and _is_valid(S_valid):
                best_S = S_valid
    
    return [_all_pts[i] for i in best_S] if best_S else [_all_pts[0]]


if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
