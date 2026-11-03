import time
import itertools
import numpy as np
from math import log

def compute_score(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    n = len(U)
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    ratio = diffset_size / sumset_size
    if ratio <= 1:
        return 1.0
    return 1.0 + log(ratio) / log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 105  # 105s deadline, 45s safety margin
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    def update_best(U, s):
        nonlocal best_U, best_score
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Try structured families
    # Family 1: {0, 1, 3, 7, ..., 2^k - 1}
    for k in range(2, 20):
        if time.time() > deadline:
            break
        U = [2**i - 1 for i in range(k)]
        s = compute_score(U)
        update_best(U, s)
    
    # Family 2: {0, 1, 3, 9, 27, ...} powers of 3
    for k in range(2, 15):
        if time.time() > deadline:
            break
        U = [3**i for i in range(k)]
        s = compute_score(U)
        update_best(U, s)
    
    # Family 3: Geometric progressions with various bases
    for base in range(2, 10):
        for k in range(2, 12):
            if time.time() > deadline:
                break
            U = [base**i for i in range(k)]
            s = compute_score(U)
            update_best(U, s)
    
    # Family 4: Arithmetic-geometric hybrid
    for a in range(1, 5):
        for r in range(2, 6):
            for k in range(2, 10):
                if time.time() > deadline:
                    break
                U = [a * (r**i) for i in range(k)]
                s = compute_score(U)
                update_best(U, s)
    
    # Local search with add, remove, and replace operations
    def local_search(U, max_val_limit=50000):
        U = sorted(set(U))
        current_score = compute_score(U)
        improved = True
        iterations = 0
        max_iterations = 5000
        while improved and time.time() < deadline and iterations < max_iterations:
            improved = False
            iterations += 1
            U_set = set(U)
            
            # Try adding elements
            candidates = []
            # Small range
            if max(U) < 1000:
                candidates.extend(range(1, min(max_val_limit, max(U)*2 + 20)))
            else:
                candidates.extend(range(max(U)-10, max(U)+10))
            # Multiplicative candidates
            for mult in [2, 3, 5, 7, 10, 13, 17]:
                c = max(U) * mult
                if c < max_val_limit:
                    candidates.append(c)
            # Geometric candidates
            for base in [2, 3, 4, 5]:
                for exp in range(1, 15):
                    c = base**exp
                    if c < max_val_limit and c not in U_set:
                        candidates.append(c)
            
            candidates = sorted(set(c for c in candidates if c not in U_set and c > 0))
            
            best_add = None
            best_add_score = current_score
            for c in candidates[:50]:  # Limit candidates per iteration
                new_U = U + [c]
                new_score = compute_score(new_U)
                if new_score > best_add_score:
                    best_add_score = new_score
                    best_add = c
            
            if best_add is not None:
                U = U + [best_add]
                current_score = best_add_score
                improved = True
                continue
            
            # Try removing elements (except 0)
            best_remove = None
            best_remove_score = current_score
            for i, u in enumerate(U):
                if u == 0:
                    continue
                if len(U) <= 3:
                    continue
                new_U = U[:i] + U[i+1:]
                new_score = compute_score(new_U)
                if new_score > best_remove_score:
                    best_remove_score = new_score
                    best_remove = u
            
            if best_remove is not None:
                U = [u for u in U if u != best_remove]
                current_score = best_remove_score
                improved = True
                continue
            
            # Try replacing elements
            best_replace = None
            best_replace_score = current_score
            for i, old_u in enumerate(U):
                if old_u == 0:
                    continue
                if len(U) <= 3:
                    continue
                # Try replacing with nearby values
                for delta in [-5, -3, -2, -1, 1, 2, 3, 5, 10, 20]:
                    new_u = old_u + delta
                    if new_u <= 0 or new_u in U_set:
                        continue
                    if new_u > max_val_limit:
                        continue
                    new_U = U[:i] + [new_u] + U[i+1:]
                    new_score = compute_score(new_U)
                    if new_score > best_replace_score:
                        best_replace_score = new_score
                        best_replace = (i, new_u)
            
            if best_replace is not None:
                i, new_u = best_replace
                U[i] = new_u
                current_score = best_replace_score
                improved = True
                continue
        
        return U, current_score
    
    # Try local search from multiple starting points
    starts = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 9],
        [0, 1, 2, 5],
        [0, 1, 4, 9],
        [0, 1, 3, 10],
        [0, 2, 5, 13],
    ]
    
    # Also add the current best as a starting point
    starts.append(best_U[:])
    
    for start_U in starts:
        if time.time() > deadline:
            break
        U, s = local_search(start_U)
        update_best(U, s)
    
    # Final polish: try more aggressive local search on best
    if time.time() < deadline - 5:
        U, s = local_search(best_U, max_val_limit=100000)
        update_best(U, s)
    
    return best_U
