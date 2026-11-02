import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    num_neighbors = 1 + n * (q - 1)  # 17
    
    # Precompute neighbors
    neighbors = np.zeros((total, num_neighbors), dtype=np.int32)
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1
    
    # Precompute neighbor sets as Python sets for fast intersection
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total)]
    
    best_size = total
    best_set = None
    
    while time.time() < deadline - 3.0:
        # 1. Randomized Greedy Construction with vectorized scoring
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                break
            
            # Vectorized: count how many uncovered words each candidate covers
            counts = uncovered_mask[neighbors].sum(axis=1)
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            # Among max-count candidates, prefer those with higher total coverage
            # (breaks symmetry, often finds better solutions)
            candidates = np.where(counts == max_count)[0]
            
            if len(candidates) > 1:
                # Secondary criterion: total coverage (including already-covered words)
                total_cov = cover_count[neighbors[candidates]].sum(axis=1)
                max_total = np.max(total_cov)
                # Prefer candidates that are NOT already selected (avoid duplicates)
                selected_arr = np.array(selected) if selected else np.array([], dtype=np.int32)
                is_new = np.array([c not in selected_arr for c in candidates])
                
                # Filter to new candidates with max total coverage
                valid = candidates[(total_cov == max_total) & is_new]
                if len(valid) > 0:
                    candidates = valid
                
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            ball = neighbors[best_idx]
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
        
        if uncovered_count > 0:
            continue
        
        # 2. Local Search
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            
            # Pass 1: Removal - check if any codeword is redundant
            indices_to_check = list(range(len(selected)))
            np.random.shuffle(indices_to_check)
            
            for i in indices_to_check:
                if time.time() > deadline - 5.0:
                    break
                cw = selected[i]
                if np.min(cover_count[neighbors[cw]]) > 1:
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    break
            
            # Pass 2: Single swap
            if not changed and len(selected) > 486 and time.time() < deadline - 10.0:
                orig_selected = selected.copy()
                orig_cover_count = cover_count.copy()
                
                idx_to_remove = np.random.randint(len(selected))
                cw_removed = selected.pop(idx_to_remove)
                cover_count[neighbors[cw_removed]] -= 1
                
                U = np.where(cover_count == 0)[0]
                
                if len(U) > 0 and len(U) <= 5:
                    # Find intersection of neighbor sets for all uncovered words
                    candidate_set = neighbor_sets[U[0]]
                    for u in U[1:]:
                        candidate_set = candidate_set & neighbor_sets[u]
                    
                    if candidate_set:
                        selected_mask = set(selected)
                        valid_candidates = [c for c in candidate_set if c not in selected_mask]
                        
                        if valid_candidates:
                            # Score candidates by how many additional words they cover
                            scored = []
                            for c in valid_candidates[:50]:
                                extra = np.sum(cover_count[neighbors[c]] == 0)
                                scored.append((extra, c))
                            scored.sort(reverse=True)
                            
                            # Pick from top candidates with some randomness
                            top_k = min(5, len(scored))
                            cw2 = scored[np.random.randint(top_k)][1]
                            selected.append(cw2)
                            cover_count[neighbors[cw2]] += 1
                            changed = True
                        else:
                            selected = orig_selected
                            cover_count = orig_cover_count
                    else:
                        selected = orig_selected
                        cover_count = orig_cover_count
                else:
                    selected = orig_selected
                    cover_count = orig_cover_count
            
            # Pass 3: 2-exchange for escaping local minima
            if not changed and len(selected) > 486 and time.time() < deadline - 15.0:
                if random.random() < 0.4:
                    orig_selected = selected.copy()
                    orig_cover_count = cover_count.copy()
                    
                    i1, i2 = np.random.choice(len(selected), size=2, replace=False)
                    # Ensure i1 < i2 after potential swap
                    if i1 > i2:
                        i1, i2 = i2, i1
                    cw1 = selected.pop(i1)
                    cw2 = selected.pop(i2 - 1)  # Adjust index after first pop
                    
                    cover_count[neighbors[cw1]] -= 1
                    cover_count[neighbors[cw2]] -= 1
                    
                    U = np.where(cover_count == 0)[0]
                    
                    if len(U) <= 6:
                        found_pair = False
                        
                        if len(U) == 0:
                            # Both were redundant
                            changed = True
                        else:
                            # For each pair of candidates, check if they cover all of U
                            # Get candidate pool from first uncovered word
                            pool = neighbor_sets[U[0]]
                            pool_list = list(pool)
                            
                            # Limit pool size for speed
                            if len(pool_list) > 40:
                                random.shuffle(pool_list)
                                pool_list = pool_list[:40]
                            
                            for a in pool_list:
                                # Check which words a covers
                                a_covers = set(neighbors[a].tolist())
                                remaining = [u for u in U if u not in a_covers]
                                
                                if len(remaining) == 0:
                                    # a covers all, pick any valid b
                                    selected_mask = set(selected)
                                    b_cands = [b for b in pool_list if b != a and b not in selected_mask]
                                    if b_cands:
                                        b = b_cands[0]
                                        selected.append(a)
                                        selected.append(b)
                                        cover_count[neighbors[a]] += 1
                                        cover_count[neighbors[b]] += 1
                                        found_pair = True
                                        break
                                elif len(remaining) == 1:
                                    # b must cover remaining[0]
                                    u_rem = remaining[0]
                                    b_cands = [b for b in neighbor_sets[u_rem] 
                                                if b != a and b not in set(selected)]
                                    if b_cands:
                                        b = b_cands[0]
                                        # Verify full coverage
                                        b_covers = set(neighbors[b].tolist())
                                        all_ok = all(u in a_covers or u in b_covers for u in U)
                                        if all_ok:
                                            selected.append(a)
                                            selected.append(b)
                                            cover_count[neighbors[a]] += 1
                                            cover_count[neighbors[b]] += 1
                                            found_pair = True
                                            break
                                else:
                                    continue
                        
                        if not found_pair:
                            selected = orig_selected
                            cover_count = orig_cover_count
                    else:
                        selected = orig_selected
                        cover_count = orig_cover_count
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            # Polish: aggressive removal pass
            if best_size <= 500 and time.time() < deadline - 10.0:
                polish_cover = np.zeros(total, dtype=np.int32)
                for cw in best_set:
                    polish_cover[neighbors[cw]] += 1
                
                polish_selected = best_set.copy()
                polish_changed = True
                while polish_changed and time.time() < deadline - 5.0:
                    polish_changed = False
                    for i in range(len(polish_selected)):
                        if time.time() > deadline - 5.0:
                            break
                        cw = polish_selected[i]
                        if np.min(polish_cover[neighbors[cw]]) > 1:
                            polish_selected.pop(i)
                            polish_cover[neighbors[cw]] -= 1
                            polish_changed = True
                            break
                
                if len(polish_selected) < best_size:
                    best_size = len(polish_selected)
                    best_set = polish_selected
        
        if best_size <= 486:
            break

    if best_set is None:
        return [list(range(n)) for _ in range(total)]
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
