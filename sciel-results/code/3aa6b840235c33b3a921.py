import time
import random
import math
import heapq
import itertools
import numpy as np


def construct(q, n, radius):
    deadline = time.monotonic() + 110.0
    total = q ** n
    rng = random.Random(42)

    # Precompute neighbors (ball of radius 1)
    neigh = []
    for idx in range(total):
        neighbors = [idx]
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neighbors.append(new_idx)
        neigh.append(neighbors)

    # Build ball_matrix as uint8 for fast vectorized operations
    ball_matrix = np.zeros((total, total), dtype=np.uint8)
    for c in range(total):
        for u in neigh[c]:
            ball_matrix[c, u] = 1

    def greedy_init(seed, time_limit):
        """Vectorized greedy set cover with random tie-breaking."""
        r = random.Random(seed)
        counts = np.zeros(total, dtype=np.int32)
        code = []
        
        while time.monotonic() < time_limit:
            remaining_mask = (counts == 0).astype(np.uint8)
            if remaining_mask.sum() == 0:
                break
            
            gains = ball_matrix @ remaining_mask
            best_gain = int(gains.max())
            if best_gain <= 0:
                break
            
            candidates = np.where(gains == best_gain)[0]
            if len(candidates) > 1:
                best_c = int(candidates[r.randint(0, len(candidates) - 1)])
            else:
                best_c = int(candidates[0])
            
            code.append(best_c)
            counts += ball_matrix[best_c]
        
        # Fallback: cover any remaining uncovered words
        remaining_mask = (counts == 0)
        if remaining_mask.any():
            code_set = set(code)
            while remaining_mask.any() and time.monotonic() < time_limit:
                uncovered = np.where(remaining_mask)[0]
                u = int(uncovered[0])
                covering = np.where(ball_matrix[:, u] == 1)[0]
                for nc in covering:
                    nc = int(nc)
                    if nc not in code_set:
                        code.append(nc)
                        code_set.add(nc)
                        counts += ball_matrix[nc]
                        remaining_mask = (counts == 0)
                        break
        
        return code, counts

    def independent_set_init(seed, time_limit):
        """
        Build initial code via greedy independent set on distance-2 graph.
        Two words are at distance 2 if they differ in exactly 2 positions.
        An independent set in this graph has no pairs at distance 2,
        which structurally avoids easily-removable redundant pairs.
        """
        r = random.Random(seed)
        # Precompute distance-2 neighbors for each index
        # dist2[i] = list of indices at Hamming distance exactly 2 from i
        dist2 = [[] for _ in range(total)]
        for i in range(total):
            for j in range(i + 1, total):
                # Count differing positions
                diff = 0
                for k in range(n):
                    pi = (i // (q ** k)) % q
                    pj = (j // (q ** k)) % q
                    if pi != pj:
                        diff += 1
                        if diff > 2:
                            break
                if diff == 2:
                    dist2[i].append(j)
                    dist2[j].append(i)
        
        # Greedy independent set: repeatedly pick a random unselected vertex,
        # then remove it and all its distance-2 neighbors from candidates
        candidates = set(range(total))
        code = []
        while candidates and time.monotonic() < time_limit:
            # Pick random candidate
            c = r.choice(list(candidates))
            code.append(c)
            candidates.discard(c)
            # Remove all distance-2 neighbors
            for nb in dist2[c]:
                candidates.discard(nb)
        
        # Now cover any remaining uncovered words
        counts = np.zeros(total, dtype=np.int32)
        code_set = set(code)
        for c in code:
            counts += ball_matrix[c]
        
        remaining_mask = (counts == 0)
        if remaining_mask.any():
            while remaining_mask.any() and time.monotonic() < time_limit:
                uncovered = np.where(remaining_mask)[0]
                u = int(uncovered[0])
                covering = np.where(ball_matrix[:, u] == 1)[0]
                for nc in covering:
                    nc = int(nc)
                    if nc not in code_set:
                        code.append(nc)
                        code_set.add(nc)
                        counts += ball_matrix[nc]
                        remaining_mask = (counts == 0)
                        break
        
        return code, counts

    def remove_redundant(code, counts, time_limit):
        """Aggressively remove all redundant codewords."""
        code_set = set(code)
        changed = True
        while changed and time.monotonic() < time_limit:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                ball_words = neigh[c]
                if np.all(counts[ball_words] > 1):
                    code.pop(i)
                    code_set.discard(c)
                    counts[ball_matrix[c] == 1] -= 1
                    changed = True
        return code, counts, code_set

    def local_search_k_swap(code, counts, code_set, k, time_limit, rng):
        """
        Iterated Local Search with k-swap perturbations.
        Remove k random codewords, add k new ones, then do aggressive redundancy removal.
        """
        best_code = code[:]
        best_size = len(code)
        best_counts = counts.copy()
        
        while time.monotonic() < time_limit:
            if len(code) < k:
                break
            
            remove_indices = rng.sample(range(len(code)), k)
            removed = []
            for idx in sorted(remove_indices, reverse=True):
                c = code[idx]
                code.pop(idx)
                code_set.discard(c)
                counts[ball_matrix[c] == 1] -= 1
                removed.append(c)
            
            new_codewords = []
            for _ in range(k):
                if rng.random() < 0.6:
                    if removed:
                        base = removed[rng.randint(0, len(removed) - 1)]
                        candidates = [x for x in neigh[base] if x not in code_set]
                        if candidates:
                            new_c = candidates[rng.randint(0, len(candidates) - 1)]
                        else:
                            new_c = rng.randint(0, total - 1)
                    else:
                        new_c = rng.randint(0, total - 1)
                else:
                    new_c = rng.randint(0, total - 1)
                
                if new_c not in code_set:
                    code.append(new_c)
                    code_set.add(new_c)
                    counts[ball_matrix[new_c] == 1] += 1
                    new_codewords.append(new_c)
            
            code, counts, code_set = remove_redundant(code, counts, time_limit)
            
            zero_mask = (counts == 0)
            if zero_mask.any():
                uncovered = np.where(zero_mask)[0]
                for u in uncovered:
                    u = int(u)
                    covering = np.where(ball_matrix[:, u] == 1)[0]
                    added = False
                    for nc in covering:
                        nc = int(nc)
                        if nc not in code_set:
                            code.append(nc)
                            code_set.add(nc)
                            counts[ball_matrix[nc] == 1] += 1
                            added = True
                            break
                    if not added:
                        break
                
                code, counts, code_set = remove_redundant(code, counts, time_limit)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                best_counts = counts.copy()
        
        return best_code, best_counts, best_size

    def sa_polish(code, counts, time_limit):
        """Simulated annealing polish for fine-tuning."""
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        best_counts = counts.copy()
        
        temp = 5.0
        cool_rate = 0.9995
        min_temp = 0.01
        
        while time.monotonic() < time_limit and temp > min_temp:
            move_type = rng.random()
            
            if move_type < 0.4:
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    ball_words = neigh[c]
                    if np.all(counts[ball_words] > 1):
                        code.pop(idx)
                        code_set.discard(c)
                        counts[ball_matrix[c] == 1] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                            best_counts = counts.copy()
                        
            elif move_type < 0.9:
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    if rng.random() < 0.7:
                        new_c = neigh[old_c][rng.randint(1, len(neigh[old_c]) - 1)]
                    else:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    old_ball = ball_matrix[old_c]
                    new_ball = ball_matrix[new_c]
                    unique_old = (old_ball == 1) & (new_ball == 0) & (counts == 1)
                    uncovered = int(unique_old.sum())
                    
                    if uncovered == 0:
                        counts[old_ball == 1] -= 1
                        counts[new_ball == 1] += 1
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                            best_counts = counts.copy()
                    else:
                        if uncovered <= 3 and rng.random() < math.exp(-uncovered / temp):
                            counts[old_ball == 1] -= 1
                            counts[new_ball == 1] += 1
                            code[idx] = new_c
                            code_set.discard(old_c)
                            code_set.add(new_c)
                            zero_mask = (counts == 0)
                            if zero_mask.any():
                                zero_indices = np.where(zero_mask)[0]
                                for u in zero_indices:
                                    u = int(u)
                                    covering = np.where(ball_matrix[:, u] == 1)[0]
                                    for nc in covering:
                                        nc = int(nc)
                                        if nc not in code_set:
                                            code.append(nc)
                                            code_set.add(nc)
                                            counts[ball_matrix[nc] == 1] += 1
                                            break
            else:
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    if rng.random() < math.exp(-1.5 / temp):
                        code.append(new_c)
                        code_set.add(new_c)
                        counts[ball_matrix[new_c] == 1] += 1
            
            temp *= cool_rate
        
        code, counts, code_set = remove_redundant(code, counts, time_limit)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            best_counts = counts.copy()
        
        return best_code, best_counts, best_size

    # Multi-restart strategy
    overall_best_code = None
    overall_best_size = total + 1
    
    # Phase 1: Multiple initializations with different seeds and methods
    time_per_init = 2.0
    
    # Use both greedy and independent-set initializations
    init_methods = [
        ("greedy", 42),
        ("greedy", 55),
        ("greedy", 68),
        ("indep", 42),
        ("indep", 55),
        ("indep", 68),
        ("greedy", 81),
        ("indep", 81),
    ]
    
    for method, seed in init_methods:
        if time.monotonic() > deadline - 20:
            break
        if method == "greedy":
            code, counts = greedy_init(seed, time.monotonic() + time_per_init)
        else:
            code, counts = independent_set_init(seed, time.monotonic() + time_per_init)
        
        if not code:
            continue
        
        code, counts, code_set = remove_redundant(code, counts, time.monotonic() + 0.5)
        
        if len(code) < overall_best_size:
            overall_best_size = len(code)
            overall_best_code = code[:]
    
    # Phase 2: ILS with k-swaps on the best solution found
    if overall_best_code is not None and time.monotonic() < deadline - 10:
        code = overall_best_code[:]
        code_set = set(code)
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            counts[ball_matrix[c] == 1] += 1
        
        for k in [1, 2]:
            if time.monotonic() > deadline - 5:
                break
            time_budget = min(15.0, deadline - time.monotonic() - 5)
            code, counts, size = local_search_k_swap(code, counts, code_set, k, time.monotonic() + time_budget, rng)
            if size < overall_best_size:
                overall_best_size = size
                overall_best_code = code[:]
        
        # Phase 3: SA polish on the best
        if time.monotonic() < deadline - 5:
            code = overall_best_code[:]
            code_set = set(code)
            counts = np.zeros(total, dtype=np.int32)
            for c in code:
                counts[ball_matrix[c] == 1] += 1
            
            time_budget = min(20.0, deadline - time.monotonic() - 3)
            code, counts, size = sa_polish(code, counts, time.monotonic() + time_budget)
            if size < overall_best_size:
                overall_best_size = size
                overall_best_code = code[:]
    
    # If we haven't found anything, do one more greedy
    if overall_best_code is None:
        code, counts = greedy_init(42, time.monotonic() + 10)
        overall_best_code = code
        overall_best_size = len(code)
    
    # Convert to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result
