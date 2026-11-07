import time

def construct():
    n = 12
    m = 40
    total_masks = 1 << n

    def count_comparables(family_masks):
        """Count comparable pairs in a list of masks."""
        cnt = 0
        L = len(family_masks)
        for i in range(L):
            ai = family_masks[i]
            for j in range(i + 1, L):
                bj = family_masks[j]
                if (ai & ~bj) == 0 or (bj & ~ai) == 0:
                    cnt += 1
        return cnt

    # 1. Initialize with the longest chain (length 13)
    family = []
    cur = 0
    family.append(0)
    for i in range(n):
        cur |= (1 << i)
        family.append(cur)

    fam_set = set(family)
    all_masks = list(range(total_masks))

    # 2. Greedily add sets that maximize comparable pairs with current family
    while len(family) < m:
        best_gain = -1
        best_mask = None

        for s in all_masks:
            if s in fam_set:
                continue

            gain = 0
            for f in family:
                if (s & ~f) == 0 or (f & ~s) == 0:
                    gain += 1

            if gain > best_gain:
                best_gain = gain
                best_mask = s

        if best_mask is None:
            break

        family.append(best_mask)
        fam_set.add(best_mask)

    # 3. Local Search with 1-swap and 2-swap moves
    best_family = family[:]
    best_fam_set = set(best_family)
    best_score = count_comparables(best_family)

    deadline = time.time() + 55

    def try_1swap(family, fam_set, score):
        """Try all 1-swap moves. Return (improved, new_family, new_fam_set, new_score) or (False, None, None, None)."""
        L = len(family)
        for i in range(L):
            old_mask = family[i]

            # Compute comparability of old_mask with rest of family
            old_comp = 0
            for j in range(L):
                if j == i:
                    continue
                fj = family[j]
                if (old_mask & ~fj) == 0 or (fj & ~old_mask) == 0:
                    old_comp += 1

            best_delta = 0
            best_repl = None

            for s in all_masks:
                if s in fam_set:
                    continue

                new_comp = 0
                for j in range(L):
                    if j == i:
                        continue
                    fj = family[j]
                    if (s & ~fj) == 0 or (fj & ~s) == 0:
                        new_comp += 1

                delta = new_comp - old_comp
                if delta > best_delta:
                    best_delta = delta
                    best_repl = s

            if best_repl is not None and best_delta > 0:
                new_family = family[:]
                new_family[i] = best_repl
                new_fam_set = fam_set - {old_mask}
                new_fam_set.add(best_repl)
                new_score = score + best_delta
                return (True, new_family, new_fam_set, new_score)

        return (False, None, None, None)

    def try_2swap(family, fam_set, score):
        """Try greedy 2-swap moves (replace two elements with two new ones)."""
        L = len(family)
        best_delta_total = 0
        best_swap = None  # (i, j, repl_i, repl_j)

        for i in range(L):
            if time.time() > deadline:
                break
            for j in range(i + 1, L):
                if time.time() > deadline:
                    break
                mi = family[i]
                mj = family[j]

                # Compute comparability of mi and mj with rest (excluding each other)
                mi_comp_rest = 0
                mj_comp_rest = 0
                mi_mj_comp = 0
                for k in range(L):
                    if k == i or k == j:
                        continue
                    fk = family[k]
                    if (mi & ~fk) == 0 or (fk & ~mi) == 0:
                        mi_comp_rest += 1
                    if (mj & ~fk) == 0 or (fk & ~mj) == 0:
                        mj_comp_rest += 1
                if (mi & ~mj) == 0 or (mj & ~mi) == 0:
                    mi_mj_comp = 1

                old_total = mi_comp_rest + mj_comp_rest + mi_mj_comp

                # Greedy: pick best si first, then best sj given si
                best_si = None
                best_si_comp = -1
                for si in all_masks:
                    if si in fam_set:
                        continue
                    comp = 0
                    for k in range(L):
                        if k == i or k == j:
                            continue
                        fk = family[k]
                        if (si & ~fk) == 0 or (fk & ~si) == 0:
                            comp += 1
                    if comp > best_si_comp:
                        best_si_comp = comp
                        best_si = si

                if best_si is None:
                    continue

                best_sj = None
                best_sj_comp = -1
                for sj in all_masks:
                    if sj in fam_set or sj == best_si:
                        continue
                    comp = 0
                    for k in range(L):
                        if k == i or k == j:
                            continue
                        fk = family[k]
                        if (sj & ~fk) == 0 or (fk & ~sj) == 0:
                            comp += 1
                    if (best_si & ~sj) == 0 or (sj & ~best_si) == 0:
                        comp += 1
                    if comp > best_sj_comp:
                        best_sj_comp = comp
                        best_sj = sj

                if best_sj is None:
                    continue

                new_total = best_si_comp + best_sj_comp
                delta_total = new_total - old_total

                if delta_total > best_delta_total:
                    best_delta_total = delta_total
                    best_swap = (i, j, best_si, best_sj)

        if best_swap is not None and best_delta_total > 0:
            i, j, si, sj = best_swap
            new_family = family[:]
            new_family[i] = si
            new_family[j] = sj
            new_fam_set = fam_set - {family[i], family[j]}
            new_fam_set.add(si)
            new_fam_set.add(sj)
            new_score = score + best_delta_total
            return (True, new_family, new_fam_set, new_score)

        return (False, None, None, None)

    # Main local search loop
    while time.time() < deadline:
        improved = False

        # Try 1-swap moves
        res = try_1swap(best_family, best_fam_set, best_score)
        if res[0]:
            best_family = res[1]
            best_fam_set = res[2]
            best_score = res[3]
            improved = True
            continue

        # Try 2-swap moves (greedy within the pair)
        res = try_2swap(best_family, best_fam_set, best_score)
        if res[0]:
            best_family = res[1]
            best_fam_set = res[2]
            best_score = res[3]
            improved = True
            continue

        # No improvement found; break
        break

    # Convert to list of lists
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)

    return result
