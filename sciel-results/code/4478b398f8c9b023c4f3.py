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

    # Precompute divisibility bitmask: div_mask[i] has bit j set if smooth[i] divides smooth[j]
    # For antichain check: a set of indices is an antichain iff for all i != j in set,
    # bit j is NOT in div_mask[i]
    div_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                div_mask[i] |= (1 << j)

    # For a bitmask of selected indices, check antichain:
    # For each i in selected, (div_mask[i] & selected) should be 0
    def is_antichain_mask(mask):
        # Extract indices from mask
        selected = []
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            selected.append(idx)
            m ^= lsb
        for i in selected:
            if div_mask[i] & mask:
                return False
        return True

    # Better: precompute compatibility for pairs
    # compat[i][j] = True if i and j can both be in an antichain
    # i and j are compatible iff smooth[i] does not divide smooth[j] and vice versa
    compat = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                if smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0:
                    compat[i][j] = False

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
    pair_list = []  # (sum, i, j)
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if compat[i][j]:
                s = smooth[i] + smooth[j]
                if s <= max_target:
                    pair_list.append((s, i, j))
                    if 2 < best_count[s]:
                        best_count[s] = 2
                        best_terms[s] = [smooth[i], smooth[j]]

    # Size 3: for each compatible pair (i,j), try adding k > j
    # Precompute for efficiency: for each pair (i,j), which k are compatible with both
    triple_list = []
    for s, i, j in pair_list:
        for k in range(j + 1, n_smooth):
            if compat[i][k] and compat[j][k]:
                s3 = s + smooth[k]
                if s3 <= max_target:
                    triple_list.append((s3, i, j, k))
                    if 3 < best_count[s3]:
                        best_count[s3] = 3
                        best_terms[s3] = [smooth[i], smooth[j], smooth[k]]

    # Check time
    if time.time() > deadline - 3.0:
        return _fill_remaining(smooth, best_count, best_terms, max_target, n_smooth, compat, deadline)

    # Size 4: for each compatible triple (i,j,k), try adding l > k
    quad_list = []
    for s3, i, j, k in triple_list:
        for l in range(k + 1, n_smooth):
            if compat[i][l] and compat[j][l] and compat[k][l]:
                s4 = s3 + smooth[l]
                if s4 <= max_target:
                    quad_list.append((s4, i, j, k, l))
                    if 4 < best_count[s4]:
                        best_count[s4] = 4
                        best_terms[s4] = [smooth[i], smooth[j], smooth[k], smooth[l]]

    # Check time
    if time.time() > deadline - 3.0:
        return _fill_remaining(smooth, best_count, best_terms, max_target, n_smooth, compat, deadline)

    # For any remaining gaps, use targeted backtracking search
    gaps = [n for n in range(1, max_target + 1) if best_terms[n] is None]
    for target in gaps:
        if time.time() > deadline - 2.0:
            break
        found = _backtrack_search(smooth, target, n_smooth, compat, min_terms=4, max_terms=8)
        if found:
            best_count[target] = len(found)
            best_terms[target] = found

    # Final fallback
    still_missing = [n for n in range(1, max_target + 1) if best_terms[n] is None]
    for target in still_missing:
        if time.time() > deadline - 1.0:
            break
        found = _backtrack_search(smooth, target, n_smooth, compat, min_terms=4, max_terms=12)
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
