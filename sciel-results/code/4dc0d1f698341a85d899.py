import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 115.0

    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    neighbors = []
    for i, c in enumerate(words):
        neigh = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors.append(neigh)
    
    # Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    while uncovered.any():
        best_idx = -1
        best_count = 0
        for j in range(N):
            count = 0
            for nb in neighbors[j]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = j
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered[nb] = False
            
        if time.time() > deadline:
            break
    
    # Build coverage counts
    coverage_count = np.zeros(N, dtype=np.int32)
    for j in selected:
        for nb in neighbors[j]:
            coverage_count[nb] += 1

    def is_redundant(j, cov):
        for nb in neighbors[j]:
            if cov[nb] == 1:
                return False
        return True

    def try_reduce(selected_set, cov, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(selected_set)):
                j = selected_set[idx]
                if is_redundant(j, cov):
                    for nb in neighbors[j]:
                        cov[nb] -= 1
                    selected_set.pop(idx)
                    changed = True
                    break
        return selected_set

    best_selected = list(selected)
    best_cov = coverage_count.copy()
    best_selected = try_reduce(best_selected, best_cov, deadline)
    best_size = len(best_selected)
    
    random.seed(42)
    
    # Local search: add-then-remove with multiple candidates per iteration
    max_iters = 2000
    iter_count = 0
    no_improve_streak = 0
    max_no_improve = 200
    
    while time.time() < deadline - 3.0 and iter_count < max_iters and no_improve_streak < max_no_improve:
        iter_count += 1
        
        # Try several random additions in this iteration
        improved_this_iter = False
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
                
            # Pick a random candidate not in the set
            add_idx = random.randint(0, N - 1)
            if add_idx in best_selected:
                continue
            
            new_selected = list(best_selected)
            new_selected.append(add_idx)
            
            new_cov = best_cov.copy()
            for nb in neighbors[add_idx]:
                new_cov[nb] += 1
            
            new_selected = try_reduce(new_selected, new_cov, deadline)
            
            if len(new_selected) < len(best_selected):
                best_selected = new_selected
                best_cov = new_cov
                best_size = len(best_selected)
                no_improve_streak = 0
                improved_this_iter = True
                break  # Restart with new best
        
        if not improved_this_iter:
            no_improve_streak += 1
        
        # Swap heuristic: remove one, then try to add one that allows reduction
        if time.time() < deadline - 3.0 and len(best_selected) > 2:
            remove_idx = random.randint(0, len(best_selected) - 1)
            removed_word = best_selected[remove_idx]
            
            # Tentatively remove
            temp_selected = list(best_selected)
            temp_selected.pop(remove_idx)
            temp_cov = best_cov.copy()
            for nb in neighbors[removed_word]:
                temp_cov[nb] -= 1
            
            # Find words that were uniquely covered by removed_word
            # These need to be re-covered
            need_cover = []
            for nb in neighbors[removed_word]:
                if temp_cov[nb] == 0:
                    need_cover.append(nb)
            
            if need_cover:
                # Try to add a word that covers many of these
                best_add = -1
                best_cover_count = 0
                for candidate in range(N):
                    if candidate in temp_selected:
                        continue
                    cover_count = 0
                    for nb in neighbors[candidate]:
                        if temp_cov[nb] == 0:
                            cover_count += 1
                    if cover_count > best_cover_count:
                        best_cover_count = cover_count
                        best_add = candidate
                
                if best_add != -1 and best_cover_count > 0:
                    temp_selected.append(best_add)
                    for nb in neighbors[best_add]:
                        temp_cov[nb] += 1
                    
                    temp_selected = try_reduce(temp_selected, temp_cov, deadline)
                    
                    if len(temp_selected) < len(best_selected):
                        best_selected = temp_selected
                        best_cov = temp_cov
                        best_size = len(best_selected)
                        no_improve_streak = 0
    
    # Final reduction pass
    best_selected = try_reduce(best_selected, best_cov, deadline)
    
    codewords = [words[i] for i in best_selected]
    return codewords
