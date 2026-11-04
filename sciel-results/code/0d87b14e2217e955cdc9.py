import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    N = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Precompute neighbors (balls of radius 1)
    neighbors = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbors.append(neigh)

    # Greedy Set Cover with randomization
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 30:
                break
            best_idx = -1
            max_gain = 0
            
            for i in range(N):
                if i in code_set:
                    continue
                gain = np.sum(uncovered[neighbors[i]])
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            if max_gain == 0:
                break
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbors[best_idx]] = False
        
        return code, code_set

    # Local search with deletion/swap
    def local_search(code, code_set, coverage):
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Deletion pass
            to_remove = []
            for c in code:
                removable = True
                for p in neighbors[c]:
                    if coverage[p] == 1:
                        removable = False
                        break
                if removable:
                    to_remove.append(c)
            
            if to_remove:
                improved = True
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    coverage[neighbors[c]] -= 1
            
            # Swap pass
            for c in list(code):
                if time.time() > deadline - 5:
                    break
                crit_points = [p for p in neighbors[c] if coverage[p] == 1]
                if not crit_points:
                    continue
                
                candidate_set = set(neighbors[crit_points[0]])
                for p in crit_points[1:]:
                    candidate_set &= set(neighbors[p])
                candidates = [x for x in candidate_set if x not in code_set]
                
                if not candidates:
                    continue
                
                best_candidate = -1
                best_score = -1
                for w in candidates:
                    score = 0
                    for p in neighbors[w]:
                        if coverage[p] == 1 and p not in neighbors[c]:
                            score += 1
                    for p in neighbors[c]:
                        if coverage[p] == 2 and p not in neighbors[w]:
                            score -= 1
                    if score > best_score:
                        best_score = score
                        best_candidate = w
                
                if best_candidate != -1 and best_score > 0:
                    code.remove(c)
                    code_set.remove(c)
                    coverage[neighbors[c]] -= 1
                    code.append(best_candidate)
                    code_set.add(best_candidate)
                    coverage[neighbors[best_candidate]] += 1
                    improved = True
                    # Quick cleanup after swap
                    to_remove = []
                    for c2 in code:
                        removable = True
                        for p in neighbors[c2]:
                            if coverage[p] == 1:
                                removable = False
                                break
                        if removable:
                            to_remove.append(c2)
                    for c2 in to_remove:
                        code.remove(c2)
                        code_set.remove(c2)
                        coverage[neighbors[c2]] -= 1
        return code, code_set, coverage

    # Kick search
    def try_kick(code, code_set, coverage, k):
        if len(code) < k:
            return False, code, code_set, coverage
        original_code = list(code)
        original_set = set(code_set)
        original_coverage = coverage.copy()
        
        removed = random.sample(code, k)
        for c in removed:
            code.remove(c)
            code_set.remove(c)
            coverage[neighbors[c]] -= 1
        
        # Add k random new codewords
        added = []
        for _ in range(k):
            new_c = random.randrange(N)
            if new_c not in code_set:
                code.append(new_c)
                code_set.add(new_c)
                coverage[neighbors[new_c]] += 1
                added.append(new_c)
        
        # Local search
        code, code_set, coverage = local_search(code, code_set, coverage)
        
        if len(code) < len(original_code):
            return True, code, code_set, coverage
        else:
            # Revert
            for c in code:
                code_set.discard(c)
            code.clear()
            for c in original_code:
                code.append(c)
                code_set.add(c)
            for c in original_code:
                coverage[neighbors[c]] += 1
            for c in added:
                coverage[neighbors[c]] -= 1
            coverage = original_coverage
            return False, code, code_set, coverage

    # Try multiple seeds for greedy
    best_code = []
    best_set = set()
    best_size = float('inf')
    
    for seed in range(3):
        if time.time() > deadline - 30:
            break
        c, cs = greedy_cover(seed)
        if len(c) < best_size:
            best_size = len(c)
            best_code = c
            best_set = cs
    
    code = best_code
    code_set = best_set
    coverage = np.zeros(N, dtype=np.int32)
    for idx in code:
        coverage[neighbors[idx]] += 1

    # Local search
    code, code_set, coverage = local_search(code, code_set, coverage)

    # Kick search
    kick_sizes = [1, 2]
    while time.time() < deadline - 5:
        k = random.choice(kick_sizes)
        improved, code, code_set, coverage = try_kick(code, code_set, coverage, k)
        if improved:
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
                best_set = set(code_set)

    # Final cleanup
    code, code_set, coverage = local_search(code, code_set, coverage)

    return [list(words[i]) for i in code]
