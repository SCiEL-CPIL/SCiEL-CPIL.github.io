import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n_vertices = 12
    m = 40
    N_masks = 1 << n_vertices  # 4096
    
    # Precompute popcount
    popcount = [0] * N_masks
    for mask in range(1, N_masks):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Precompute comparability using bitsets
    # comp_bits[i] is a bitset of masks j where i is a proper subset of j
    comp_bits = [0] * N_masks
    # For each mask j, iterate all proper subsets sub
    for j in range(1, N_masks):
        sub = (j - 1) & j
        while sub:
            comp_bits[sub] |= (1 << j)
            sub = (sub - 1) & j
        # Also add empty set as proper subset of j
        comp_bits[0] |= (1 << j)
    
    # Helper: count comparable pairs in a family of masks
    def count_comparable(family):
        n = len(family)
        cnt = 0
        for i in range(n):
            mi = family[i]
            # Count how many j > i where mi is proper subset of mj or vice versa
            for j in range(i + 1, n):
                mj = family[j]
                if (mi & mj) == mi and mi != mj:
                    cnt += 1
                elif (mi & mj) == mj and mi != mj:
                    cnt += 1
        return cnt
    
    # Initial construction: build a chain + add sets that maximize comparability
    # Start with empty set
    selected = [0]
    selected_set = {0}
    
    # Add sets greedily: at each step, pick the mask that maximizes comparable pairs
    # with already-selected sets
    remaining = [mask for mask in range(1, N_masks)]
    
    # Precompute for each mask, its comparable partners (as a set for quick lookup)
    # comp_partners[i] = set of masks j where i is comparable with j (proper subset relation)
    comp_partners = [set() for _ in range(N_masks)]
    for i in range(N_masks):
        for j in range(i + 1, N_masks):
            if (i & j) == i and i != j:
                comp_partners[i].add(j)
                comp_partners[j].add(i)
            elif (i & j) == j and i != j:
                comp_partners[i].add(j)
                comp_partners[j].add(i)
    
    # Greedy: pick masks that have the most comparability with selected
    while len(selected) < m and time.time() < deadline - 30.0:
        best_mask = -1
        best_score = -1
        for mask in remaining:
            if mask in selected_set:
                continue
            # Count how many selected masks are comparable with mask
            score = 0
            for smask in selected:
                if smask in comp_partners[mask]:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask == -1:
            # Pick first available
            best_mask = remaining[0]
        selected.append(best_mask)
        selected_set.add(best_mask)
        remaining.remove(best_mask)
    
    # Ensure we have exactly m masks
    while len(selected) < m:
        for mask in range(N_masks):
            if mask not in selected_set:
                selected.append(mask)
                selected_set.add(mask)
                break
    
    current_score = count_comparable(selected)
    best_selected = list(selected)
    best_score = current_score
    
    # Local search: try swapping one selected mask with one unselected mask
    # Use bitset operations for speed
    selected_set = set(selected)
    unselected = [mask for mask in range(N_masks) if mask not in selected_set]
    
    # Precompute comparability as bitsets for fast counting
    # comp_bitset[i] = bitset of all masks comparable with i
    comp_bitset = [0] * N_masks
    for i in range(N_masks):
        for j in comp_partners[i]:
            comp_bitset[i] |= (1 << j)
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Try all possible swaps: remove selected[i], add unselected[j]
        for i in range(m):
            if time.time() > deadline - 10.0:
                break
            mi = selected[i]
            for mj in unselected:
                if time.time() > deadline - 10.0:
                    break
                # Compute delta: (comparability of mj with others) - (comparability of mi with others)
                # Others = selected except mi
                # Count comparability of mi with others
                old_comp = 0
                for k in range(m):
                    if k == i:
                        continue
                    mk = selected[k]
                    if mk in comp_partners[mi]:
                        old_comp += 1
                
                # Count comparability of mj with others
                new_comp = 0
                for k in range(m):
                    if k == i:
                        continue
                    mk = selected[k]
                    if mk in comp_partners[mj]:
                        new_comp += 1
                
                delta = new_comp - old_comp
                if delta > 0:
                    # Accept swap
                    selected[i] = mj
                    selected_set.discard(mi)
                    selected_set.add(mj)
                    unselected.remove(mi)
                    unselected.append(mj)
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                    break  # Restart inner loop
    
    # Second pass: try 2-swap moves for better escape
    if time.time() < deadline - 5.0:
        for _ in range(50):  # Limit number of 2-swap attempts
            if time.time() > deadline - 5.0:
                break
            # Pick two random selected and two random unselected
            idx1, idx2 = random.sample(range(m), 2)
            ui1, ui2 = random.sample(range(len(unselected)), 2)
            mi1, mi2 = selected[idx1], selected[idx2]
            mj1, mj2 = unselected[ui1], unselected[ui2]
            
            # Remove mi1, mi2; add mj1, mj2
            others = [selected[k] for k in range(m) if k != idx1 and k != idx2]
            
            old_comp = 0
            for mk in others:
                if mk in comp_partners[mi1]:
                    old_comp += 1
                if mk in comp_partners[mi2]:
                    old_comp += 1
            if mi1 in comp_partners[mi2]:
                old_comp += 1
            
            new_comp = 0
            for mk in others:
                if mk in comp_partners[mj1]:
                    new_comp += 1
                if mk in comp_partners[mj2]:
                    new_comp += 1
            if mj1 in comp_partners[mj2]:
                new_comp += 1
            
            delta = new_comp - old_comp
            if delta > 0:
                selected[idx1] = mj1
                selected[idx2] = mj2
                selected_set.discard(mi1)
                selected_set.discard(mi2)
                selected_set.add(mj1)
                selected_set.add(mj2)
                unselected[ui1] = mi1
                unselected[ui2] = mi2
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(selected)
    
    final_selected = best_selected
    
    # Convert to lists of integers
    result = []
    for mask in final_selected:
        subset = [i for i in range(n_vertices) if (mask >> i) & 1]
        result.append(subset)
    
    # Verify: exactly 40 distinct subsets
    if len(result) != 40:
        result = [[i % 12] for i in range(40)]
    
    if len(set(tuple(s) for s in result)) != 40:
        result = [[i % 12] for i in range(40)]
    
    for s in result:
        for x in s:
            if not isinstance(x, int) or isinstance(x, bool):
                pass
    
    return result
