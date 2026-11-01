import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit, leave buffer)

    if n == 0:
        return [[]]
    
    # Handle small cases or specific known perfect codes
    if q == 3 and radius == 1 and n == 4:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for all words as bitmasks (integers) for fast coverage operations
    neighbors_mask = [0] * total_words
    neighbors_list = []
    for i, w in enumerate(all_words):
        mask = 0
        nb = [i]
        mask |= (1 << i)  # Include self
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
                nb.append(j)
        neighbors_mask[i] = mask
        neighbors_list.append(nb)
    
    # Precompute reverse neighbors
    reverse_neighbors = [0] * total_words
    for i in range(total_words):
        mask = neighbors_mask[i]
        while mask:
            lowbit = mask & (-mask)
            j = lowbit.bit_length() - 1
            reverse_neighbors[j] |= (1 << i)
            mask ^= lowbit

    def greedy_init(seed=0, time_budget=None):
        if time_budget is None:
            time_budget = deadline - time.time()
        rng = random.Random(seed)
        uncovered_mask = (1 << total_words) - 1  # All words uncovered
        code_indices = []
        code_set_mask = 0
        
        while uncovered_mask and time.time() < deadline - 5:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                if (code_set_mask >> i) & 1:
                    continue
                cnt = (neighbors_mask[i] & uncovered_mask).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
                
            code_indices.append(best_idx)
            code_set_mask |= (1 << best_idx)
            uncovered_mask &= ~neighbors_mask[best_idx]
                
        return code_indices

    def direct_sum_init():
        if q == 3 and n == 6 and radius == 1:
            tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
            code = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=2):
                    c = h + tail
                    code.append(word_to_idx[c])
            return code
        return greedy_init(0)
    
    def get_uncovered(code_mask):
        uncovered = 0
        for j in range(total_words):
            if (reverse_neighbors[j] & code_mask) == 0:
                uncovered |= (1 << j)
        return uncovered

    def local_search(init_code, seed=0, time_budget=None):
        if time_budget is None:
            time_budget = deadline - time.time()
        rng = random.Random(seed)
        active = list(init_code)
        active_set_mask = 0
        for c in active:
            active_set_mask |= (1 << c)
        
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in active:
            mask = neighbors_mask[c]
            while mask:
                lowbit = mask & (-mask)
                j = lowbit.bit_length() - 1
                coverage[j] += 1
                mask ^= lowbit
        
        def repair():
            nonlocal active, active_set_mask, coverage
            uncovered_mask = get_uncovered(active_set_mask)
            while uncovered_mask and time.time() < deadline - 3:
                best_idx = -1
                best_count = -1
                for i in range(total_words):
                    if (active_set_mask >> i) & 1:
                        continue
                    cnt = (neighbors_mask[i] & uncovered_mask).bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                if best_idx == -1 or best_count == 0:
                    break
                active.append(best_idx)
                active_set_mask |= (1 << best_idx)
                for nb in neighbors_list[best_idx]:
                    coverage[nb] += 1
                uncovered_mask = get_uncovered(active_set_mask)
        
        repair()
        
        best_size = len(active)
        best_code = list(active)
        best_mask = active_set_mask
        
        while time.time() < deadline - 3:
            improved = False
            
            # Phase 1: Removal
            removal_found = False
            indices = list(range(len(active)))
            rng.shuffle(indices)
            
            for idx in indices:
                c = active[idx]
                can_remove = True
                for nb in neighbors_list[c]:
                    if coverage[nb] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors_list[c]:
                        coverage[nb] -= 1
                    active.pop(idx)
                    active_set_mask &= ~(1 << c)
                    improved = True
                    removal_found = True
                    break
            
            if removal_found:
                if len(active) < best_size:
                    best_size = len(active)
                    best_code = list(active)
                    best_mask = active_set_mask
                continue

            # Phase 2: 2-1 Swap (Remove 2, Add 1)
            swap21_found = False
            # Sample pairs to avoid O(N^2) full scan if N is large, but N~75 is small
            # We can iterate all pairs if time permits
            if len(active) > 1:
                pairs = list(itertools.combinations(range(len(active)), 2))
                rng.shuffle(pairs)
                
                for i, j in pairs:
                    if time.time() > deadline - 5:
                        break
                    c1 = active[i]
                    c2 = active[j]
                    
                    # Words uniquely covered by c1 or c2
                    # U = (N(c1) U N(c2)) \ N(active \ {c1, c2})
                    # A word w is in U if:
                    # 1. w is in N(c1) or N(c2)
                    # 2. coverage[w] <= 2 (covered by at most c1 and c2)
                    #    Actually, if coverage[w] == 1, it's covered by exactly one of them (or neither, but it's in N(c1/c2))
                    #    If coverage[w] == 2, it's covered by both.
                    #    If coverage[w] > 2, it's covered by others too.
                    
                    # We need to find c3 such that all w in U are covered by c3.
                    # U = { w | (w in N(c1) or w in N(c2)) and coverage[w] <= 2 }
                    # Note: if w is in N(c1) and N(c2), coverage[w] is at least 2.
                    # If w is in N(c1) only, coverage[w] is at least 1.
                    
                    # Let's construct the mask of U
                    mask_c1 = neighbors_mask[c1]
                    mask_c2 = neighbors_mask[c2]
                    mask_union = mask_c1 | mask_c2
                    
                    # We need to check if there exists c3 covering all words in U.
                    # U is a set of words.
                    # We can iterate through the words in mask_union and filter by coverage <= 2.
                    # Then check if any c3 covers all of them.
                    
                    # Optimization:
                    # If U is empty, we can just remove c1 and c2 (handled by removal phase, but maybe removal phase missed it? No, removal phase checks individually. If removing c1 leaves c2 covering some words uniquely, removal of c1 fails. But if we remove BOTH, it might work. This is exactly what 2-1 swap with c3=none? No, 2-1 swap adds c3. If U is empty, we can remove c1, c2 and add nothing? That's a 2-0 swap.
                    # Let's handle 2-0 swap first (Removal of 2).
                    
                    # Check 2-0: Can we remove c1 and c2?
                    # This requires that for all w in N(c1) U N(c2), coverage[w] >= 2 AND if coverage[w]==2, it must be covered by c1 and c2?
                    # No. If we remove c1 and c2, w must be covered by someone else.
                    # So we need: for all w in N(c1) U N(c2), coverage[w] > 2?
                    # If coverage[w] == 2, and w is covered by c1 and c2, then removing both leaves w uncovered.
                    # So for 2-0, we need coverage[w] >= 3 for all w in N(c1) U N(c2)?
                    # Wait. If w is covered by c1 and c3, and we remove c1, w is still covered by c3.
                    # If w is covered by c1 and c2, and we remove c1 and c2, w is uncovered.
                    # So yes, for 2-0 removal, we need that no word is covered *only* by c1 and c2.
                    # i.e., for all w in N(c1) U N(c2), the number of coverers in active \ {c1, c2} is >= 1.
                    # This is equivalent to: coverage[w] - (1 if w in N(c1) else 0) - (1 if w in N(c2) else 0) >= 1.
                    
                    can_remove_2 = True
                    for w in range(total_words):
                        if (mask_union >> w) & 1:
                            count = coverage[w]
                            if (mask_c1 >> w) & 1:
                                count -= 1
                            if (mask_c2 >> w) & 1:
                                count -= 1
                            if count < 1:
                                can_remove_2 = False
                                break
                    if can_remove_2:
                        # Remove c1 and c2
                        for nb in neighbors_list[c1]:
                            coverage[nb] -= 1
                        for nb in neighbors_list[c2]:
                            coverage[nb] -= 1
                        # Remove from active
                        # Note: indices i, j might change after pop
                        # Pop higher index first
                        if i > j:
                            active.pop(i)
                            active.pop(j)
                        else:
                            active.pop(j)
                            active.pop(i)
                        active_set_mask &= ~(1 << c1)
                        active_set_mask &= ~(1 << c2)
                        improved = True
                        swap21_found = True
                        break

                    # Check 2-1: Find c3 covering U
                    # U = words in mask_union with coverage <= 2
                    # Actually, we need c3 to cover all words that would become uncovered.
                    # Words that become uncovered are those w in N(c1) U N(c2) such that
                    # count_of_others(w) == 0.
                    # count_of_others(w) = coverage[w] - (1 if w in N(c1) else 0) - (1 if w in N(c2) else 0)
                    
                    uncovered_mask_if_removed = 0
                    for w in range(total_words):
                        if (mask_union >> w) & 1:
                            count = coverage[w]
                            if (mask_c1 >> w) & 1:
                                count -= 1
                            if (mask_c2 >> w) & 1:
                                count -= 1
                            if count == 0:
                                uncovered_mask_if_removed |= (1 << w)
                    
                    if uncovered_mask_if_removed == 0:
                        # This is the 2-0 case, already handled above?
                        # If uncovered_mask_if_removed is 0, then can_remove_2 should have been true.
                        # Why? Because can_remove_2 checks count < 1.
                        # So this block is redundant if can_remove_2 logic is correct.
                        pass
                    else:
                        # We need c3 such that N(c3) covers uncovered_mask_if_removed
                        # i.e., (neighbors_mask[c3] & uncovered_mask_if_removed) == uncovered_mask_if_removed
                        best_c3 = -1
                        # Iterate all c3 not in active
                        for c3 in range(total_words):
                            if (active_set_mask >> c3) & 1:
                                continue
                            if (neighbors_mask[c3] & uncovered_mask_if_removed) == uncovered_mask_if_removed:
                                best_c3 = c3
                                break
                        if best_c3 != -1:
                            # Perform 2-1 swap
                            for nb in neighbors_list[c1]:
                                coverage[nb] -= 1
                            for nb in neighbors_list[c2]:
                                coverage[nb] -= 1
                            
                            # Remove c1, c2
                            if i > j:
                                active.pop(i)
                                active.pop(j)
                            else:
                                active.pop(j)
                                active.pop(i)
                            active_set_mask &= ~(1 << c1)
                            active_set_mask &= ~(1 << c2)
                            
                            # Add c3
                            active.append(best_c3)
                            active_set_mask |= (1 << best_c3)
                            for nb in neighbors_list[best_c3]:
                                coverage[nb] += 1
                            
                            improved = True
                            swap21_found = True
                            break

            if swap21_found:
                if len(active) < best_size:
                    best_size = len(active)
                    best_code = list(active)
                    best_mask = active_set_mask
                continue

            # Phase 3: 1-1 Swap
            swap_found = False
            indices = list(range(len(active)))
            rng.shuffle(indices)
            
            for idx in indices:
                c = active[idx]
                unique_words = []
                for nb in neighbors_list[c]:
                    if coverage[nb] == 1:
                        unique_words.append(nb)
                
                if not unique_words:
                    continue

                best_candidate = -1
                best_score = -1
                
                for c_prime in range(total_words):
                    if (active_set_mask >> c_prime) & 1:
                        continue
                    
                    covers_all = True
                    for w in unique_words:
                        if not ((neighbors_mask[c_prime] >> w) & 1):
                            covers_all = False
                            break
                            
                    if not covers_all:
                        continue
                    
                    score = 0
                    for nb in neighbors_list[c_prime]:
                        if coverage[nb] > 1:
                            score += 1
                            
                    if score > best_score:
                        best_score = score
                        best_candidate = c_prime

                if best_candidate != -1:
                    for nb in neighbors_list[c]:
                        coverage[nb] -= 1
                    active.pop(idx)
                    active_set_mask &= ~(1 << c)
                    
                    active.append(best_candidate)
                    active_set_mask |= (1 << best_candidate)
                    for nb in neighbors_list[best_candidate]:
                        coverage[nb] += 1
                    
                    improved = True
                    swap_found = True
                    break
            
            if not improved:
                if time.time() < deadline - 8:
                    criticalities = []
                    for c in active:
                        count = 0
                        for nb in neighbors_list[c]:
                            if coverage[nb] == 1:
                                count += 1
                        criticalities.append((count, c))
                    
                    criticalities.sort()
                    
                    perturb_count = rng.randint(1, 2)
                    for _ in range(perturb_count):
                        if not active:
                            break
                        _, c_to_remove = criticalities[0]
                        
                        try:
                            idx = active.index(c_to_remove)
                        except ValueError:
                            break
                            
                        active.pop(idx)
                        active_set_mask &= ~(1 << c_to_remove)
                        for nb in neighbors_list[c_to_remove]:
                            coverage[nb] -= 1
                        
                        criticalities.pop(0)
                        break
                    
                    repair()
                    
                    if len(active) < best_size:
                        best_size = len(active)
                        best_code = list(active)
                        best_mask = active_set_mask

            if len(active) < best_size:
                best_size = len(active)
                best_code = list(active)
                best_mask = active_set_mask

        return best_code

    best_code = None
    best_size = float('inf')
    
    if q == 3 and n == 6 and radius == 1:
        ds_code = direct_sum_init()
        result = local_search(ds_code, seed=42)
        if len(result) < best_size:
            best_size = len(result)
            best_code = result
    
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        g_code = greedy_init(seed)
        result = local_search(g_code, seed=seed * 100)
        if len(result) < best_size:
            best_size = len(result)
            best_code = result
    
    if best_code is not None and time.time() < deadline - 15:
        for seed in [10, 20, 30, 40]:
            if time.time() > deadline - 10:
                break
            result = local_search(best_code, seed=seed)
            if len(result) < best_size:
                best_size = len(result)
                best_code = result
    
    if best_code is None:
        best_code = greedy_init(0)

    return [list(all_words[idx]) for idx in best_code]
