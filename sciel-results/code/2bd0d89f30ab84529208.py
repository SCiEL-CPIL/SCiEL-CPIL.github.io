"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    max_target = 128

    # Generate all 2,3,5-smooth numbers up to max_target
    smooth = []
    for n in range(1, max_target + 1):
        x = n
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(n)

    n_smooth = len(smooth)
    idx_of = {v: i for i, v in enumerate(smooth)}

    # Precompute conflict matrix: conflict[i][j] = True if i and j CANNOT both be in antichain
    # i and j conflict if smooth[i] divides smooth[j] or vice versa
    conflict = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0:
                conflict[i][j] = True
                conflict[j][i] = True

    # Precompute compatibility lists for faster iteration
    compat_list = [[] for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if not conflict[i][j]:
                compat_list[i].append(j)

    # Precompute for each pair (i,j) with i<j, whether they're compatible
    # And precompute for triples using bitmask approach

    best_count = [max_target + 1] * (max_target + 1)
    best_terms = [None] * (max_target + 1)

    # Size 1
    for i, v in enumerate(smooth):
        if v <= max_target:
            best_count[v] = 1
            best_terms[v] = [v]

    # Size 2: all compatible pairs
    for i in range(n_smooth):
        for j in compat_list[i]:
            s = smooth[i] + smooth[j]
            if s <= max_target:
                if 2 < best_count[s]:
                    best_count[s] = 2
                    best_terms[s] = [smooth[i], smooth[j]]

    # Size 3: for each compatible pair (i,j), try adding k > j
    for i in range(n_smooth):
        for j in compat_list[i]:
            for k in range(j + 1, n_smooth):
                if not conflict[i][k] and not conflict[j][k]:
                    s3 = smooth[i] + smooth[j] + smooth[k]
                    if s3 <= max_target:
                        if 3 < best_count[s3]:
                            best_count[s3] = 3
                            best_terms[s3] = [smooth[i], smooth[j], smooth[k]]

    # Check time
    if time.time() > deadline - 5.0:
        return _fill_remaining(smooth, best_count, best_terms, max_target, n_smooth, conflict, deadline)

    # Size 4: for each compatible triple (i,j,k), try adding l > k
    for i in range(n_smooth):
        for j in compat_list[i]:
            for k in range(j + 1, n_smooth):
                if not conflict[i][k] and not conflict[j][k]:
                    base3 = smooth[i] + smooth[j] + smooth[k]
                    for l in range(k + 1, n_smooth):
                        if not conflict[i][l] and not conflict[j][l] and not conflict[k][l]:
                            s4 = base3 + smooth[l]
                            if s4 <= max_target:
                                if 4 < best_count[s4]:
                                    best_count[s4] = 4
                                    best_terms[s4] = [smooth[i], smooth[j], smooth[k], smooth[l]]

    # Check time
    if time.time() > deadline - 5.0:
        return _fill_remaining(smooth, best_count, best_terms, max_target, n_smooth, conflict, deadline)

    # Size 5: for each compatible quadruple, try adding m > l
    for i in range(n_smooth):
        for j in compat_list[i]:
            for k in range(j + 1, n_smooth):
                if not conflict[i][k] and not conflict[j][k]:
                    for l in range(k + 1, n_smooth):
                        if not conflict[i][l] and not conflict[j][l] and not conflict[k][l]:
                            base4 = smooth[i] + smooth[j] + smooth[k] + smooth[l]
                            for m in range(l + 1, n_smooth):
                                if not conflict[i][m] and not conflict[j][m] and not conflict[k][m] and not conflict[l][m]:
                                    s5 = base4 + smooth[m]
                                    if s5 <= max_target:
                                        if 5 < best_count[s5]:
                                            best_count[s5] = 5
                                            best_terms[s5] = [smooth[i], smooth[j], smooth[k], smooth[l], smooth[m]]

    # Check time
    if time.time() > deadline - 5.0:
        return _fill_remaining(smooth, best_count, best_terms, max_target, n_smooth, conflict, deadline)

    # For any remaining gaps, use targeted backtracking search
    gaps = [n for n in range(1, max_target + 1) if best_terms[n] is None]
    for target in gaps:
        if time.time() > deadline - 3.0:
            break
        found = _backtrack_search(smooth, target, n_smooth, conflict, min_terms=1, max_terms=10)
        if found:
            best_count[target] = len(found)
            best_terms[target] = found

    # Final fallback
    still_missing = [n for n in range(1, max_target + 1) if best_terms[n] is None]
    for target in still_missing:
        if time.time() > deadline - 2.0:
            break
        found = _backtrack_search(smooth, target, n_smooth, conflict, min_terms=1, max_terms=15)
        if found:
            best_count[target] = len(found)
            best_terms[target] = found
        else:
            # Last resort: try any representation
            for cnt in range(1, 15):
                cands = [x for x in smooth if x <= target]
                for terms in combinations(cands, cnt):
                    if sum(terms) == target:
                        # Verify antichain
                        ok = True
                        for a, b in combinations(terms, 2):
                            if b % a == 0 or a % b == 0:
                                ok = False
                                break
                        if ok:
                            best_count[target] = cnt
                            best_terms[target] = list(terms)
                            break
                if best_terms[target] is not None:
                    break
            if best_terms[target] is None:
                best_count[target] = 1
                best_terms[target] = [target] if target in idx_of else [1]

    return [best_terms[n] for n in range(1, max_target + 1)]


def _backtrack_search(smooth, target, n_smooth, compat, min_terms, max_terms):
    """Backtracking search for antichain representation of target."""
    best = None
    smooth_sorted = sorted(smooth)  # already sorted
    # Index mapping
    idx_of = {v: i for i, v in enumerate(smooth)}

    def backtrack(remaining, min_idx, current_indices):
        nonlocal best
        cur_len = len(current_indices)
        if cur_len > max_terms:
            return
        if remaining == 0:
            if cur_len >= min_terms:
                if best is None or cur_len < len(best):
                    best = list(current_indices)
            return
        if remaining < 0:
            return

        slots_left = max_terms - cur_len
        if slots_left <= 0:
            return
        # Need at least 'slots_left' terms each >= 1, so remaining >= slots_left
        if remaining < slots_left:
            return
        # Also, the largest possible single term is remaining, so we need at least one
        # term <= remaining

        for i in range(min_idx, n_smooth):
            v = smooth[i]
            if v > remaining:
                break
            # Check compatibility with all in current
            compatible = True
            for j in current_indices:
                if not compat[i][j]:
                    compatible = False
                    break
            if not compatible:
                continue
            current_indices.append(i)
            backtrack(remaining - v, i + 1, current_indices)
            current_indices.pop()
            if best is not None and len(best) == min_terms:
                return

    backtrack(target, 0, [])
    if best:
        return [smooth[i] for i in best]
    return None


def _fill_remaining(smooth, best_count, best_terms, max_target, n_smooth, compat, deadline):
    """Fill remaining targets using backtracking search."""
    for n in range(1, max_target + 1):
        if best_terms[n] is None:
            if time.time() > deadline - 1.0:
                break
            found = _backtrack_search(smooth, n, n_smooth, compat, min_terms=1, max_terms=10)
            if found:
                best_count[n] = len(found)
                best_terms[n] = found
            else:
                # Last resort
                for cnt in range(1, 12):
                    cands = [x for x in smooth if x <= n]
                    for terms in combinations(cands, cnt):
                        if sum(terms) == n:
                            ok = True
                            for a, b in combinations(terms, 2):
                                if b % a == 0 or a % b == 0:
                                    ok = False
                                    break
                            if ok:
                                best_count[n] = cnt
                                best_terms[n] = list(terms)
                                break
                    if best_terms[n] is not None:
                        break
                if best_terms[n] is None:
                    best_count[n] = 1
                    best_terms[n] = [n] if n in smooth else [1]
    return [best_terms[n] for n in range(1, max_target + 1)]


# EVOLVE-BLOCK-END
