"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 5.0
    
    # Generate all candidate vectors with entries in {-1, 0, 1}
    # Weight 1: norm 1, weight 2: norm sqrt(2), weight 3: norm sqrt(3)
    
    # Strategy: Use vectors of weight 2 (norm sqrt(2)) and weight 1 (norm 1)
    # For mixed norms, constraint is ||vi - vj|| >= max(||vi||, ||vj||)
    
    # Let's focus on a single norm class first for simplicity, then try mixing.
    
    # Generate all weight-2 vectors: 4 * C(11,2) = 220
    candidates = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Also add weight-1 vectors: 2 * 11 = 22
    for i in range(D):
        for si in (1, -1):
            v = [0] * D
            v[i] = si
            candidates.append(v)
    
    # Also add weight-3 vectors: 8 * C(11,3) = 660
    for i, j, k in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    n = len(candidates)
    cand_arr = np.array(candidates, dtype=np.int64)
    
    # Compute norms squared
    norms_sq = np.sum(cand_arr ** 2, axis=1)
    
    # Compute all pairwise dot products
    # dot[i,j] = candidates[i] . candidates[j]
    dots = cand_arr @ cand_arr.T
    
    # For each pair (i,j), check if they can coexist:
    # ||vi - vj||^2 = ni + nj - 2*dot >= max(ni, nj)
    # ni + nj - 2*dot >= max(ni, nj)
    
    # Build conflict matrix: conflict[i,j] = True if they CANNOT coexist
    max_norms = np.maximum(norms_sq[:, None], norms_sq[None, :])
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < max_norms)
    np.fill_diagonal(conflict, False)
    
    # Now find maximum independent set in this graph
    # Use greedy with multiple restarts
    
    best_set = set()
    best_size = 0
    
    def greedy_order(order):
        """Greedy: process in given order, add if no conflict with already added."""
        selected = set()
        for idx in order:
            if idx in selected:
                continue
            # Check if idx conflicts with any in selected
            can_add = True
            for s in selected:
                if conflict[idx, s]:
                    can_add = False
                    break
            if can_add:
                selected.add(idx)
        return selected
    
    def greedy_mrv():
        """Minimum remaining values heuristic."""
        remaining = set(range(n))
        selected = set()
        while remaining:
            # For each candidate, count how many remaining it conflicts with
            # Pick the one with minimum conflicts (or maximum compatibility)
            best_idx = None
            best_score = -1
            for idx in remaining:
                # Count conflicts with other remaining
                conflicts = sum(1 for j in remaining if j != idx and conflict[idx, j])
                score = len(remaining) - 1 - conflicts  # number of compatible
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.add(best_idx)
            remaining.discard(best_idx)
            # Remove all that conflict with best_idx
            to_remove = set()
            for j in remaining:
                if conflict[best_idx, j]:
                    to_remove.add(j)
            remaining -= to_remove
        return selected
    
    # Try greedy with different orderings
    rng = random.Random(42)
    
    # First: simple greedy with natural order
    selected = greedy_order(range(n))
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = selected
    
    # Try MRV greedy
    if time.time() < deadline - margin:
        selected = greedy_mrv()
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
    
    # Multi-start randomized greedy
    for trial in range(50):
        if time.time() > deadline - margin:
            break
        order = list(range(n))
        rng.shuffle(order)
        selected = greedy_order(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
    
    # Local improvement: try to add more vectors by removing one and adding multiple
    if time.time() < deadline - margin:
        for _ in range(100):
            if time.time() > deadline - margin:
                break
            improved = False
            for remove_idx in list(best_set):
                new_set = best_set - {remove_idx}
                # Try to add any vector not in new_set that doesn't conflict
                added = 0
                for add_idx in range(n):
                    if add_idx in new_set:
                        continue
                    can_add = True
                    for s in new_set:
                        if conflict[add_idx, s]:
                            can_add = False
                            break
                    if can_add:
                        new_set.add(add_idx)
                        added += 1
                if len(new_set) > len(best_set):
                    best_set = new_set
                    best_size = len(new_set)
                    improved = True
                    break
            if not improved:
                break
    
    result = np.array([candidates[i] for i in sorted(best_set)], dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
