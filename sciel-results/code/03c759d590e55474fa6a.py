import numpy as np
import time
from itertools import combinations
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580.0
    
    # ============================================================
    # Strategy: Find maximum independent set of 4-sets in 11D
    # where two 4-sets conflict if they share 3+ indices.
    # Each valid 4-set contributes 16 points (all sign combos).
    # All points have norm^2 = 4, and min pairwise dist^2 >= 4.
    # ============================================================
    
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)
    
    # Precompute conflict matrix: two 4-sets conflict if intersection size >= 3
    conflict = np.zeros((n_sets, n_sets), dtype=np.uint8)
    for i in range(n_sets):
        si = set(all_4sets[i])
        for j in range(i + 1, n_sets):
            inter = len(si & set(all_4sets[j]))
            if inter >= 3:
                conflict[i, j] = 1
                conflict[j, i] = 1
    
    # Build adjacency list for faster checking
    adj = [np.where(conflict[i] > 0)[0] for i in range(n_sets)]
    
    def count_conflicts(selection):
        """Count conflicting pairs in a selection list."""
        cnt = 0
        sel_set = set(selection)
        for i in range(len(selection)):
            for j in adj[selection[i]]:
                if j in sel_set and j > selection[i]:
                    cnt += 1
        return cnt
    
    # Simulated annealing / greedy with local search
    rng = random.Random(42)
    
    best_selection = []
    best_size = 0
    
    # Phase 1: Greedy construction with random ordering
    n_trials = 2000
    for trial in range(n_trials):
        if time.time() > deadline - 120:
            break
        
        order = list(range(n_sets))
        rng.shuffle(order)
        
        selected = []
        selected_set = set()
        
        for idx in order:
            # Check if idx conflicts with any selected
            ok = True
            for s in selected:
                if conflict[idx, s]:
                    ok = False
                    break
            if ok:
                selected.append(idx)
                selected_set.add(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selection = selected[:]
    
    # Phase 2: Local search - single replacement moves
    improved = True
    while improved and time.time() < deadline - 60:
        improved = False
        selected_set = set(best_selection)
        n_sel = len(best_selection)
        
        # Try each position for removal and each candidate for insertion
        for pos in range(n_sel):
            if time.time() > deadline - 60:
                break
            removed = best_selection[pos]
            remaining = [x for x in best_selection if x != removed]
            rem_set = set(remaining)
            
            # Find candidates that don't conflict with remaining
            for cand in range(n_sets):
                if cand in rem_set:
                    continue
                # Check cand doesn't conflict with any in remaining
                ok = True
                for r in remaining:
                    if conflict[cand, r]:
                        ok = False
                        break
                if ok:
                    # Adding cand gives same size; but we removed one, so net 0
                    # We need to find a size-increasing move, which requires
                    # finding a set that can be added without removing anything.
                    pass
        
        # Better approach: try to add any unselected set that doesn't conflict
        for cand in range(n_sets):
            if time.time() > deadline - 60:
                break
            if cand in selected_set:
                continue
            ok = True
            for s in best_selection:
                if conflict[cand, s]:
                    ok = False
                    break
            if ok:
                best_selection.append(cand)
                selected_set.add(cand)
                improved = True
                break
    
    # Phase 3: Pair replacement - remove one, add two
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        selected_set = set(best_selection)
        
        for pos in range(len(best_selection)):
            if time.time() > deadline - 30:
                break
            removed = best_selection[pos]
            remaining = [x for x in best_selection if x != removed]
            rem_set = set(remaining)
            
            # Find all candidates compatible with remaining
            compatible = []
            for cand in range(n_sets):
                if cand in rem_set:
                    continue
                ok = True
                for r in remaining:
                    if conflict[cand, r]:
                        ok = False
                        break
                if ok:
                    compatible.append(cand)
            
            # Try to find two compatible candidates that don't conflict with each other
            found_pair = False
            for i in range(len(compatible)):
                if found_pair:
                    break
                for j in range(i + 1, len(compatible)):
                    if not conflict[compatible[i], compatible[j]]:
                        best_selection = remaining + [compatible[i], compatible[j]]
                        improved = True
                        found_pair = True
                        break
        
        # Also try single add again
        if not improved:
            selected_set = set(best_selection)
            for cand in range(n_sets):
                if time.time() > deadline - 30:
                    break
                if cand in selected_set:
                    continue
                ok = True
                for s in best_selection:
                    if conflict[cand, s]:
                        ok = False
                        break
                if ok:
                    best_selection.append(cand)
                    improved = True
                    break
    
    # Generate points from best_selection
    points = []
    for idx in best_selection:
        s = all_4sets[idx]
        i, j, k, l = s
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
    
    if not points:
        # Fallback: 220 norm-2 vectors
        for i, j in combinations(range(d), 2):
            for si, sj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
