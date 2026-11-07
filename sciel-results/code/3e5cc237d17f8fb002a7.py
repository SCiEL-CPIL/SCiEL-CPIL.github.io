# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    m = P['m']
    total_masks = 1 << n  # 4096

    # Precompute comparability bitsets
    all_ones = total_masks - 1  # 0xFFF

    comp_bitset = [0] * total_masks

    for i in range(total_masks):
        ci = 0
        # Enumerate all proper submasks of i
        sub = (i - 1) & i
        while True:
            ci |= (1 << sub)
            if sub == 0:
                break
            sub = (sub - 1) & i
        # Enumerate all proper supermasks of i
        comp = all_ones ^ i
        if comp != 0:
            sub = (comp - 1) & comp
            while True:
                s = i | sub
                if s != i:
                    ci |= (1 << s)
                if sub == 0:
                    break
                sub = (sub - 1) & comp
        comp_bitset[i] = ci

    # Greedy construction with better initial structure
    # Start with a chain, then fill in greedily
    selected = []
    selected_mask = 0

    # Add empty set first
    selected.append(0)
    selected_mask |= (1 << 0)

    # Greedily add masks that maximize comparability
    for _ in range(m - 1):
        best_mask = -1
        best_score = -1
        for k in range(total_masks):
            if (selected_mask >> k) & 1:
                continue
            score = (comp_bitset[k] & selected_mask).bit_count()
            if score > best_score:
                best_score = score
                best_mask = k
        selected.append(best_mask)
        selected_mask |= (1 << best_mask)

    # Calculate current score
    current_score = 0
    for i in range(m):
        current_score += (comp_bitset[selected[i]] & selected_mask).bit_count()
    current_score //= 2

    best_score = current_score
    best_selected = list(selected)

    # Local search: single swaps first
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(m):
            if time.time() >= deadline:
                break
            old_mask = selected[idx]
            sel_without_old = selected_mask & ~(1 << old_mask)
            loss = (comp_bitset[old_mask] & sel_without_old).bit_count()

            for new_mask in range(total_masks):
                if (selected_mask >> new_mask) & 1:
                    continue
                gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                delta = gain - loss
                if delta > 0:
                    selected[idx] = new_mask
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << new_mask)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                    improved = True
                    break  # restart scan

    # 2-opt local search: swap two positions simultaneously
    if time.time() < deadline:
        improved2 = True
        while improved2 and time.time() < deadline:
            improved2 = False
            for i in range(m):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, m):
                    if time.time() >= deadline:
                        break
                    mi = selected[i]
                    mj = selected[j]
                    # Remove both
                    sel_without = selected_mask & ~(1 << mi) & ~(1 << mj)
                    loss = ((comp_bitset[mi] & sel_without).bit_count() +
                            (comp_bitset[mj] & sel_without).bit_count())
                    
                    best_delta = -1
                    best_ni = -1
                    best_nj = -1
                    for ni in range(total_masks):
                        if (selected_mask >> ni) & 1:
                            continue
                        if ni == mj:  # already in set
                            continue
                        for nj in range(total_masks):
                            if (selected_mask >> nj) & 1:
                                continue
                            if nj == mi or nj == ni:
                                continue
                            # Check distinctness
                            if ni == nj:
                                continue
                            sel_with_new = sel_without | (1 << ni) | (1 << nj)
                            gain = ((comp_bitset[ni] & sel_with_new).bit_count() +
                                    (comp_bitset[nj] & sel_with_new).bit_count())
                            delta = gain - loss
                            if delta > best_delta:
                                best_delta = delta
                                best_ni = ni
                                best_nj = nj
                    
                    if best_delta > 0:
                        selected[i] = best_ni
                        selected[j] = best_nj
                        selected_mask = (selected_mask & ~(1 << mi) & ~(1 << mj)) | (1 << best_ni) | (1 << best_nj)
                        current_score += best_delta
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = list(selected)
                        improved2 = True
                        break  # restart inner loop
                if improved2:
                    break

    # Final polish: try single swaps again
    if time.time() < deadline:
        improved3 = True
        while improved3 and time.time() < deadline:
            improved3 = False
            for idx in range(m):
                if time.time() >= deadline:
                    break
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                loss = (comp_bitset[old_mask] & sel_without_old).bit_count()

                for new_mask in range(total_masks):
                    if (selected_mask >> new_mask) & 1:
                        continue
                    gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                    delta = gain - loss
                    if delta > 0:
                        selected[idx] = new_mask
                        selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << new_mask)
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = list(selected)
                        improved3 = True
                        break

    # Convert to lists of integers
    result = []
    for mask in best_selected:
        result.append([b for b in range(n) if (mask >> b) & 1])

    # Ensure proper types
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])

    return final_result
# EVOLVE-BLOCK-END
