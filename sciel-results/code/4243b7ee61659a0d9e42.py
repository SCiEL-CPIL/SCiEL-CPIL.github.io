import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        return _construct_ils_3_7_1()
    return _greedy_cover(q, n, radius)

def _construct_ils_3_7_1():
    start_time = time.time()
    time_limit = 115.0
    
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = []
    for w in all_words:
        nb = [word_to_idx[w]]
        for i in range(7):
            for d in [1, 2]:
                new_w = list(w)
                new_w[i] = (w[i] + d) % 3
                nb.append(word_to_idx[tuple(new_w)])
        neighbors.append(nb)
        
    # Helper: Local Search (Remove/Swap)
    def local_search(code_set, counts, uncovered):
        code_list = list(code_set)
        random.shuffle(code_list)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(code_list)):
                c = code_list[i]
                # Try remove
                for nb in neighbors[c]:
                    counts[nb] -= 1
                    if counts[nb] == 0:
                        uncovered.add(nb)
                
                if len(uncovered) == 0:
                    code_set.remove(c)
                    code_list.pop(i)
                    improved = True
                    break
                else:
                    # Try swap if uncovered is small
                    if len(uncovered) <= 3:
                        candidates = None
                        for w in uncovered:
                            if candidates is None:
                                candidates = set(neighbors[w])
                            else:
                                candidates &= set(neighbors[w])
                        
                        if candidates:
                            valid_cands = [c_new for c_new in candidates if c_new not in code_set]
                            if valid_cands:
                                c_new = random.choice(valid_cands)
                                for nb in neighbors[c_new]:
                                    counts[nb] += 1
                                    if counts[nb] == 1:
                                        uncovered.discard(nb)
                                
                                code_set.remove(c)
                                code_set.add(c_new)
                                code_list[i] = c_new
                                improved = True
                                break
                            else:
                                for nb in neighbors[c]:
                                    counts[nb] += 1
                                    if counts[nb] == 1:
                                        uncovered.discard(nb)
                                continue
                        else:
                            for nb in neighbors[c]:
                                counts[nb] += 1
                                if counts[nb] == 1:
                                    uncovered.discard(nb)
                            continue
                    else:
                        for nb in neighbors[c]:
                            counts[nb] += 1
                            if counts[nb] == 1:
                                uncovered.discard(nb)
                        continue
            
        return code_set, counts, uncovered

    # Helper: Randomized Greedy Init
    def randomized_greedy_init():
        uncovered = set(range(N))
        code_set = set()
        counts = np.zeros(N, dtype=np.int32)
        
        while uncovered:
            if time.time() - start_time > time_limit:
                break
                
            target = random.choice(list(uncovered))
            cands = neighbors[target]
            
            best_cand = -1
            best_cover = -1
            
            for c in cands:
                if c in code_set:
                    continue
                cover_count = 0
                for nb in neighbors[c]:
                    if nb in uncovered:
                        cover_count += 1
                if cover_count > best_cover:
                    best_cover = cover_count
                    best_cand = c
                    
            if best_cand == -1:
                break
                
            code_set.add(best_cand)
            for nb in neighbors[best_cand]:
                counts[nb] += 1
                if counts[nb] == 1:
                    uncovered.discard(nb)
                    
        return code_set, counts, uncovered

    # Main ILS Loop
    best_code_set = None
    best_size = float('inf')
    
    # Initial solution
    code_set, counts, uncovered = randomized_greedy_init()
    code_set, counts, uncovered = local_search(code_set, counts, uncovered)
    
    if len(code_set) < best_size:
        best_size = len(code_set)
        best_code_set = set(code_set)
        
    while time.time() - start_time < time_limit:
        # Perturbation
        current_code = list(best_code_set)
        if not current_code:
            break
            
        num_remove = max(1, int(len(current_code) * 0.15))
        to_remove = random.sample(current_code, num_remove)
        
        perturbed_code_set = set(best_code_set)
        perturbed_code_set.difference_update(to_remove)
        
        # Recompute counts and uncovered
        counts = np.zeros(N, dtype=np.int32)
        for c in perturbed_code_set:
            for nb in neighbors[c]:
                counts[nb] += 1
        
        uncovered = set(np.where(counts == 0)[0])
        
        # Repair Phase
        while uncovered:
            if time.time() - start_time > time_limit:
                break
                
            target = random.choice(list(uncovered))
            cands = neighbors[target]
            
            best_cand = -1
            best_cover = -1
            
            for c in cands:
                if c in perturbed_code_set:
                    continue
                cover_count = 0
                for nb in neighbors[c]:
                    if nb in uncovered:
                        cover_count += 1
                if cover_count > best_cover:
                    best_cover = cover_count
                    best_cand = c
                    
            if best_cand == -1:
                break
                
            perturbed_code_set.add(best_cand)
            for nb in neighbors[best_cand]:
                counts[nb] += 1
                if counts[nb] == 1:
                    uncovered.discard(nb)
                    
        # Local Search on perturbed solution
        perturbed_code_set, counts, uncovered = local_search(perturbed_code_set, counts, uncovered)
        
        if len(perturbed_code_set) < best_size:
            best_size = len(perturbed_code_set)
            best_code_set = set(perturbed_code_set)

    if best_code_set is None:
        return [list(all_words[0])]
        
    return [list(all_words[idx]) for idx in best_code_set]

def _greedy_cover(q, n, radius):
    # Fallback for other parameters (not used in this specific task instance)
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    covers = []
    for w in all_words:
        covered = set()
        covered.add(word_to_idx[w])
        for r in range(1, radius + 1):
            for positions in itertools.combinations(range(n), r):
                for deltas in itertools.product(range(1, q), repeat=r):
                    new_w = list(w)
                    for pos, d in zip(positions, deltas):
                        new_w[pos] = (w[pos] + d) % q
                    covered.add(word_to_idx[tuple(new_w)])
        covers.append(covered)
    
    uncovered = set(range(N))
    code = []
    while uncovered:
        best = None
        best_count = -1
        for i in range(N):
            count = len(covers[i] & uncovered)
            if count > best_count:
                best_count = count
                best = i
        code.append(list(all_words[best]))
        uncovered -= covers[best]
    
    return code
