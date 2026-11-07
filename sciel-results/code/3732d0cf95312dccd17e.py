import time
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 75  # Leave 15s margin for safety

    # 1. Build the full comparability matrix (4096x4096)
    masks = np.arange(total_masks, dtype=np.int64)
    i_idx = masks[:, None]
    j_idx = masks[None, :]
    mask_12 = (1 << n) - 1
    comp_subset = (i_idx & (mask_12 ^ j_idx)) == 0
    comp_superset = (j_idx & (mask_12 ^ i_idx)) == 0
    comp_matrix = (comp_subset | comp_superset).astype(np.uint8)
    np.fill_diagonal(comp_matrix, 0)

    # 2. Greedy construction from a chain
    family_masks = [0]
    cur = 0
    for i in range(n):
        cur |= (1 << i)
        family_masks.append(cur)
    fam_set = set(family_masks)
    fam_bitset = 0
    for f in family_masks:
        fam_bitset |= (1 << f)

    def popcount(x):
        return bin(x).count('1')

    # Precompute scores for all masks
    scores = np.zeros(total_masks, dtype=np.int32)
    for s in range(total_masks):
        scores[s] = popcount(fam_bitset & comp_matrix[s].view(np.uint8).reshape(-1))

    # Greedy loop
    while len(family_masks) < m:
        best_score = -1
        best_mask = -1
        for s in range(total_masks):
            if s in fam_set:
                continue
            if scores[s] > best_score:
                best_score = scores[s]
                best_mask = s
        if best_mask == -1:
            break
        family_masks.append(best_mask)
        fam_set.add(best_mask)
        fam_bitset |= (1 << best_mask)
        # Update scores incrementally
        row = comp_matrix[best_mask]
        for idx in range(total_masks):
            if idx not in fam_set and row[idx]:
                scores[idx] += 1

    # 3. Local search with 1-opt and 2-opt swaps
    def get_total_score(fam_list):
        cnt = 0
        L = len(fam_list)
        for i in range(L):
            ai = fam_list[i]
            for j in range(i + 1, L):
                bj = fam_list[j]
                if comp_matrix[ai, bj]:
                    cnt += 1
        return cnt

    best_family = family_masks[:]
    best_score = get_total_score(best_family)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        fam_set_local = set(best_family)

        # 1-opt: try swapping each element
        for i in range(len(best_family)):
            if time.time() > deadline:
                break
            old_mask = best_family[i]
            old_deg = int(comp_matrix[old_mask, best_family].sum()) - 1  # exclude self

            best_delta = 0
            best_repl = -1
            for s in range(total_masks):
                if s in fam_set_local:
                    continue
                new_deg = int(comp_matrix[s, best_family].sum())
                if comp_matrix[s, old_mask]:
                    new_deg -= 1
                delta = new_deg - old_deg
                if delta > best_delta:
                    best_delta = delta
                    best_repl = s

            if best_repl != -1 and best_delta > 0:
                best_family[i] = best_repl
                fam_set_local.remove(old_mask)
                fam_set_local.add(best_repl)
                best_score += best_delta
                improved = True

        # 2-opt: try removing 2 elements and adding 2 better ones
        if time.time() < deadline:
            for i in range(len(best_family)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_family)):
                    if time.time() > deadline:
                        break
                    m1 = best_family[i]
                    m2 = best_family[j]
                    # Degrees of m1 and m2 in the family (excluding each other and themselves)
                    deg1 = int(comp_matrix[m1, best_family].sum()) - 1
                    deg2 = int(comp_matrix[m2, best_family].sum()) - 1
                    if comp_matrix[m1, m2]:
                        deg1 -= 1
                        deg2 -= 1
                    removed_deg = deg1 + deg2

                    # Find best pair (s1, s2) not in family to replace
                    best_delta = 0
                    best_pair = (-1, -1)
                    # Only check a subset for speed: top candidates by degree
                    # Precompute degrees of all external masks
                    ext_masks = [s for s in range(total_masks) if s not in fam_set_local]
                    if len(ext_masks) > 200:
                        # Sort by degree and take top 200
                        ext_degrees = [(int(comp_matrix[s, best_family].sum()), s) for s in ext_masks]
                        ext_degrees.sort(reverse=True)
                        ext_masks = [s for _, s in ext_degrees[:200]]

                    for idx1 in range(len(ext_masks)):
                        s1 = ext_masks[idx1]
                        for idx2 in range(idx1 + 1, len(ext_masks)):
                            s2 = ext_masks[idx2]
                            new_deg1 = int(comp_matrix[s1, best_family].sum())
                            new_deg2 = int(comp_matrix[s2, best_family].sum())
                            # Subtract edges to m1 and m2
                            if comp_matrix[s1, m1]:
                                new_deg1 -= 1
                            if comp_matrix[s1, m2]:
                                new_deg1 -= 1
                            if comp_matrix[s2, m1]:
                                new_deg2 -= 1
                            if comp_matrix[s2, m2]:
                                new_deg2 -= 1
                            # Add edge between s1 and s2
                            if comp_matrix[s1, s2]:
                                new_deg1 += 1
                                new_deg2 += 1
                            delta = (new_deg1 + new_deg2) - removed_deg
                            if delta > best_delta:
                                best_delta = delta
                                best_pair = (s1, s2)

                    if best_pair[0] != -1 and best_delta > 0:
                        best_family[i] = best_pair[0]
                        best_family[j] = best_pair[1]
                        fam_set_local.remove(m1)
                        fam_set_local.remove(m2)
                        fam_set_local.add(best_pair[0])
                        fam_set_local.add(best_pair[1])
                        best_score += best_delta
                        improved = True

    # Final polish: multiple passes of 1-opt with random restarts on the family
    for restart in range(5):
        if time.time() > deadline:
            break
        # Perturb: randomly swap 2-3 elements
        perturbed = best_family[:]
        for _ in range(random.randint(2, 3)):
            i, j = random.sample(range(len(perturbed)), 2)
            if i != j:
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

        # Local search on perturbed
        fam_set_p = set(perturbed)
        for i in range(len(perturbed)):
            if time.time() > deadline:
                break
            old_mask = perturbed[i]
            old_deg = int(comp_matrix[old_mask, perturbed].sum()) - 1
            best_delta = 0
            best_repl = -1
            for s in range(total_masks):
                if s in fam_set_p:
                    continue
                new_deg = int(comp_matrix[s, perturbed].sum())
                if comp_matrix[s, old_mask]:
                    new_deg -= 1
                delta = new_deg - old_deg
                if delta > best_delta:
                    best_delta = delta
                    best_repl = s
            if best_repl != -1 and best_delta > 0:
                perturbed[i] = best_repl
                fam_set_p.remove(old_mask)
                fam_set_p.add(best_repl)

        pert_score = get_total_score(perturbed)
        if pert_score > best_score:
            best_family = perturbed
            best_score = pert_score

    # Convert to list of lists
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)

    return result
