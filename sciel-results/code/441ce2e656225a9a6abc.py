import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s limit
    
    # Generate all integer vectors with squared norm 4
    # These are: 2*e_i, -2*e_i, and (±1,±1,±1,±1,0,...,0) for 4-combinations
    base_points = []
    for i in range(d):
        p1 = np.zeros(d, dtype=np.int64)
        p1[i] = 2
        base_points.append(p1)
        
        p2 = np.zeros(d, dtype=np.int64)
        p2[i] = -2
        base_points.append(p2)
    
    # Generate all 4-combinations of indices
    combos = list(itertools.combinations(range(d), 4))
    n_combos = len(combos)
    
    # Precompute intersection sizes using bitmask for speed
    combo_masks = []
    for c in combos:
        mask = 0
        for idx in c:
            mask |= (1 << idx)
        combo_masks.append(mask)
    
    # Precompute intersection sizes matrix
    inter_sizes = np.zeros((n_combos, n_combos), dtype=np.int8)
    for i in range(n_combos):
        for j in range(i + 1, n_combos):
            inter = combo_masks[i] & combo_masks[j]
            sz = bin(inter).count('1')
            inter_sizes[i, j] = sz
            inter_sizes[j, i] = sz
            
    # Precompute neighbors (conflicts) for faster greedy
    neighbors = [[] for _ in range(n_combos)]
    for i in range(n_combos):
        for j in range(i + 1, n_combos):
            if inter_sizes[i, j] > 2:
                neighbors[i].append(j)
                neighbors[j].append(i)
    
    best_selected = None
    best_count = 0
    
    # Try multiple random orderings with greedy selection
    n_trials = 2000
    for trial in range(n_trials):
        if time.time() > deadline - 120: # Leave 120s for local search
            break
            
        if trial == 0:
            # Sort by ascending degree
            degrees = [len(neighbors[i]) for i in range(n_combos)]
            order = sorted(range(n_combos), key=lambda x: degrees[x])
        elif trial == 1:
            # Sort by descending degree
            degrees = [len(neighbors[i]) for i in range(n_combos)]
            order = sorted(range(n_combos), key=lambda x: -degrees[x])
        elif trial == 2:
            # Weighted: prefer low degree but break ties by index
            degrees = [len(neighbors[i]) for i in range(n_combos)]
            order = sorted(range(n_combos), key=lambda x: (degrees[x], x))
        elif trial == 3:
            # Reverse index order
            order = list(range(n_combos - 1, -1, -1))
        else:
            rng = np.random.default_rng(trial * 7919 + 13)
            order = rng.permutation(n_combos)
        
        selected = []
        forbidden = set()
        
        for ci in order:
            if ci not in forbidden:
                selected.append(ci)
                # Add all neighbors to forbidden
                for nb in neighbors[ci]:
                    forbidden.add(nb)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]
    
    # Local improvement: 1-for-1 swaps, 2-for-2 swaps, and removals
    if best_selected is not None and best_count > 0:
        if time.time() < deadline - 20:
            best_selected = best_selected[:]
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                best_set = set(best_selected)
                
                # Try 1-for-1 swap
                for si in range(len(best_selected)):
                    if time.time() > deadline - 10:
                        break
                    removed = best_selected[si]
                    remaining = best_set - {removed}
                    for cj in range(n_combos):
                        if cj in remaining:
                            continue
                        valid = True
                        for sj in remaining:
                            if inter_sizes[cj, sj] > 2:
                                valid = False
                                break
                        if valid:
                            best_selected[si] = cj
                            best_set = set(best_selected)
                            improved = True
                            break
                    if improved:
                        break
                
                # Try 2-for-2 swap
                if not improved and len(best_selected) >= 2:
                    for si1 in range(len(best_selected) - 1):
                        if time.time() > deadline - 10:
                            break
                        for si2 in range(si1 + 1, len(best_selected)):
                            if time.time() > deadline - 10:
                                break
                            r1 = best_selected[si1]
                            r2 = best_selected[si2]
                            remaining = best_set - {r1, r2}
                            # Try pairs of unselected combos
                            for cj1 in range(n_combos):
                                if cj1 in remaining or cj1 in best_set:
                                    continue
                                for cj2 in range(cj1 + 1, n_combos):
                                    if cj2 in remaining or cj2 in best_set:
                                        continue
                                    if inter_sizes[cj1, cj2] > 2:
                                        continue
                                    valid = True
                                    for sj in remaining:
                                        if inter_sizes[cj1, sj] > 2 or inter_sizes[cj2, sj] > 2:
                                            valid = False
                                            break
                                    if valid:
                                        best_selected[si1] = cj1
                                        best_selected[si2] = cj2
                                        best_set = set(best_selected)
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                
                # Try removing one to free up space (might help with subsequent additions)
                if not improved and len(best_selected) > 2:
                    for si in range(len(best_selected)):
                        if time.time() > deadline - 10:
                            break
                        removed = best_selected[si]
                        new_set = best_set - {removed}
                        new_selected = [x for i, x in enumerate(best_selected) if i != si]
                        # Try to add multiple combos to the freed space
                        added = 0
                        for cj in range(n_combos):
                            if cj in new_set:
                                continue
                            valid = True
                            for sj in new_set:
                                if inter_sizes[cj, sj] > 2:
                                    valid = False
                                    break
                            if valid:
                                new_selected.append(cj)
                                new_set.add(cj)
                                added += 1
                        if added > 1:
                            best_selected = new_selected
                            best_set = set(best_selected)
                            improved = True
                            break
    
    # Build points array
    points = list(base_points)
    
    if best_selected is not None:
        for ci in best_selected:
            c = combos[ci]
            indices = list(c)
            for mask in range(16):
                p = np.zeros(d, dtype=np.int64)
                for k, idx in enumerate(indices):
                    if (mask >> k) & 1:
                        p[idx] = -1
                    else:
                        p[idx] = 1
                points.append(p)
    
    points_array = np.array(points, dtype=np.int64)
    
    return points_array
