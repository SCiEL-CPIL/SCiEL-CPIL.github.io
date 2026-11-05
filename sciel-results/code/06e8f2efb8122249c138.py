import time
import math
from math import gcd, lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    N = 256
    candidates = list(range(2, 257))
    n = len(candidates)
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    idx_to_val = candidates

    # Precompute conflict masks: conflict_mask[i] has bit j set if lcm(candidates[i], candidates[j]) <= 256
    conflict_mask = [0] * n
    for i in range(n):
        vi = idx_to_val[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = idx_to_val[j]
            if lcm(vi, vj) <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def get_score_frac(mask):
        score = Fraction(0)
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            score += Fraction(1, idx_to_val[idx])
            m ^= lsb
        return score

    def get_selected(mask):
        return [idx_to_val[i] for i in range(n) if (mask >> i) & 1]

    def is_valid_mask(mask):
        # Verify no two selected elements conflict
        sel = get_selected(mask)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                if lcm(sel[i], sel[j]) <= 256:
                    return False
        return True

    def greedy_initial():
        # Greedy: add smallest possible (highest weight) first
        mask = 0
        for v in range(2, 257):
            idx = val_to_idx[v]
            if (conflict_mask[idx] & mask) == 0:
                mask |= (1 << idx)
        return mask

    def local_search(initial_mask, time_limit=10.0):
        current_mask = initial_mask
        current_score = get_score_frac(current_mask)
        improved = True
        start_ls_time = time.time()

        while improved:
            improved = False

            if time.time() > deadline or time.time() - start_ls_time > time_limit:
                break

            # Add phase: try to add any v not in mask, prioritize smaller v
            for v in range(2, 257):
                idx = val_to_idx[v]
                if not (current_mask & (1 << idx)):
                    if (conflict_mask[idx] & current_mask) == 0:
                        current_mask |= (1 << idx)
                        current_score += Fraction(1, v)
                        improved = True

            if improved:
                continue

            # Build lists
            in_mask = [i for i in range(n) if (current_mask >> i) & 1]
            not_in_mask = [i for i in range(n) if not ((current_mask >> i) & 1)]

            # 1-for-1 swap: remove u, add v where 1/v > 1/u
            in_mask_desc = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
            not_in_mask_asc = sorted(not_in_mask, key=lambda i: idx_to_val[i])

            found_swap = False
            for u_idx in in_mask_desc:
                if time.time() > deadline:
                    break
                u_val = idx_to_val[u_idx]
                mask_without_u = current_mask & ~(1 << u_idx)

                for v_idx in not_in_mask_asc:
                    v_val = idx_to_val[v_idx]
                    if v_val >= u_val:
                        break
                    if (conflict_mask[v_idx] & mask_without_u) == 0:
                        current_mask = (current_mask & ~(1 << u_idx)) | (1 << v_idx)
                        current_score += Fraction(1, v_val) - Fraction(1, u_val)
                        improved = True
                        found_swap = True
                        break

                if found_swap:
                    break

            if found_swap:
                continue

            # 2-for-1 swap: remove u1, u2, add v where 1/v > 1/u1 + 1/u2
            in_mask_desc2 = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
            not_in_mask_asc2 = sorted(not_in_mask, key=lambda i: idx_to_val[i])
            found_2swap = False
            for i1 in range(len(in_mask_desc2)):
                if time.time() > deadline:
                    break
                u1_idx = in_mask_desc2[i1]
                u1_val = idx_to_val[u1_idx]
                for i2 in range(i1 + 1, len(in_mask_desc2)):
                    u2_idx = in_mask_desc2[i2]
                    u2_val = idx_to_val[u2_idx]
                    gain_needed = Fraction(1, u1_val) + Fraction(1, u2_val)
                    mask_without_u1_u2 = current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)

                    for v_idx in not_in_mask_asc2:
                        v_val = idx_to_val[v_idx]
                        if Fraction(1, v_val) <= gain_needed:
                            break
                        if (conflict_mask[v_idx] & mask_without_u1_u2) == 0:
                            current_mask = (current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)) | (1 << v_idx)
                            current_score += Fraction(1, v_val) - Fraction(1, u1_val) - Fraction(1, u2_val)
                            improved = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break

            if found_2swap:
                continue

            # 1-for-2 swap: remove u, add v1, v2 where 1/v1 + 1/v2 > 1/u
            in_mask_desc3 = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
            not_in_mask_asc3 = sorted(not_in_mask, key=lambda i: idx_to_val[i])
            found_12swap = False
            for u_idx in in_mask_desc3:
                if time.time() > deadline:
                    break
                u_val = idx_to_val[u_idx]
                gain_needed = Fraction(1, u_val)
                mask_without_u = current_mask & ~(1 << u_idx)

                for v1_idx in not_in_mask_asc3:
                    v1_val = idx_to_val[v1_idx]
                    if (conflict_mask[v1_idx] & mask_without_u) != 0:
                        continue
                    if Fraction(1, v1_val) >= gain_needed:
                        continue
                    mask_with_v1 = mask_without_u | (1 << v1_idx)
                    for v2_idx in not_in_mask_asc3:
                        if v2_idx == v1_idx:
                            continue
                        v2_val = idx_to_val[v2_idx]
                        if Fraction(1, v1_val) + Fraction(1, v2_val) <= gain_needed:
                            break
                        if (conflict_mask[v2_idx] & mask_with_v1) != 0:
                            continue
                        current_mask = (current_mask & ~(1 << u_idx)) | (1 << v1_idx) | (1 << v2_idx)
                        current_score += Fraction(1, v1_val) + Fraction(1, v2_val) - Fraction(1, u_val)
                        improved = True
                        found_12swap = True
                        break
                    if found_12swap:
                        break
                if found_12swap:
                    break

        return current_mask, current_score

    # Phase 1: Greedy initial solution
    mask = greedy_initial()
    mask, score = local_search(mask, time_limit=10.0)

    # Phase 2: Try multiple restarts with different greedy orders
    import random
    rng = random.Random(42)

    best_mask = mask
    best_score = score

    # Try reversed greedy (largest first)
    mask_rev = 0
    for v in range(256, 1, -1):
        idx = val_to_idx[v]
        if (conflict_mask[idx] & mask_rev) == 0:
            mask_rev |= (1 << idx)
    mask_rev, score_rev = local_search(mask_rev, time_limit=8.0)
    if score_rev > best_score:
        best_mask = mask_rev
        best_score = score_rev

    # Try random restarts if time permits
    restarts_done = 2
    while time.time() < deadline - 15.0 and restarts_done < 8:
        # Randomized greedy
        rand_mask = 0
        order = list(range(2, 257))
        rng.shuffle(order)
        for v in order:
            idx = val_to_idx[v]
            if (conflict_mask[idx] & rand_mask) == 0:
                rand_mask |= (1 << idx)
        rand_mask, rand_score = local_search(rand_mask, time_limit=5.0)
        if rand_score > best_score:
            best_mask = rand_mask
            best_score = rand_score
        restarts_done += 1

    # Phase 3: Final polish - exhaustive local search with all swap types
    if time.time() < deadline - 5.0:
        best_mask, best_score = local_search(best_mask, time_limit=15.0)

    # Verify and return
    selected = get_selected(best_mask)
    if not is_valid_mask(best_mask):
        # Fallback to simple greedy
        selected = []
        for v in range(2, 257):
            if all(lcm(v, y) > 256 for y in selected):
                selected.append(v)

    if not selected:
        selected = [2]

    return selected
