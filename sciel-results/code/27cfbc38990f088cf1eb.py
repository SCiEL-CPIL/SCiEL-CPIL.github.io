import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 570
    
    four_subsets = list(combinations(range(d), 4))
    n4 = len(four_subsets)  # 330
    
    # Conflict: two 4-subsets conflict if |intersection| >= 3
    conflict = np.zeros((n4, n4), dtype=bool)
    for i in range(n4):
        si = four_subsets[i]
        for j in range(i + 1, n4):
            sj = four_subsets[j]
            common = len(set(si) & set(sj))
            if common >= 3:
                conflict[i, j] = True
                conflict[j, i] = True
    
    rng = np.random.default_rng(42)
    best_selected = []
    best_count = 0
    
    while time.time() < deadline - 15:
        # Random greedy
        order = rng.permutation(n4)
        used = np.zeros(n4, dtype=bool)
        selected = []
        selected_set = set()
        
        for idx in order:
            if not np.any(conflict[idx] & used):
                used[idx] = True
                selected.append(idx)
                selected_set.add(idx)
        
        # Local search: try to add, then try swaps
        changed = True
        while changed:
            changed = False
            # Try to add any compatible unselected
            for idx in range(n4):
                if idx not in selected_set and not np.any(conflict[idx] & used):
                    used[idx] = True
                    selected.append(idx)
                    selected_set.add(idx)
                    changed = True
            
            # Try 1-for-2 swaps: remove one, add two
            for i in range(len(selected)):
                if time.time() > deadline - 15:
                    break
                rem = selected[i]
                used[rem] = False
                selected_set.discard(rem)
                
                added = []
                for j in range(n4):
                    if j in selected_set or j == rem:
                        continue
                    if not np.any(conflict[j] & used):
                        added.append(j)
                        used[j] = True
                        selected_set.add(j)
                        if len(added) == 2:
                            break
                
                if len(added) >= 2:
                    selected = [x for x in selected if x != rem]
                    selected.extend(added)
                    changed = True
                else:
                    # Undo
                    for a in added:
                        used[a] = False
                        selected_set.discard(a)
                    used[rem] = True
                    selected_set.add(rem)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = list(selected)
        
        if best_count >= 22:
            break
    
    points = []
    for idx in best_selected:
        support = four_subsets[idx]
        for signs in range(16):
            v = np.zeros(d, dtype=np.int64)
            for k, pos in enumerate(support):
                v[pos] = 1 if (signs >> k) & 1 else -1
            points.append(v)
    
    if len(points) < 2:
        points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
    
    return np.around(np.array(points, dtype=np.int64)).astype(np.int64)
