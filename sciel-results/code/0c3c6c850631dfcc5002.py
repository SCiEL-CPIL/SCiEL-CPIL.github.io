"""Football pool covering code K_3(7,1). New design: greedy init + local search."""
import time
import itertools
import numpy as np

_DEADLINE = 150.0
_MARGIN = 10.0


def construct(q, n, radius):
    # Only the task's parameters are used, but keep the signature generic.
    start = time.time()
    total = q ** n  # 2187 for q=3,n=7
    # Precompute the radius-1 ball (as a list of integer indices) for every word.
    # A word is an integer in base-q. Its radius-1 ball: itself plus all words
    # differing in exactly one coordinate.
    base = q
    # digit extraction helper
    def digits_of(x):
        d = [0] * n
        for i in range(n):
            d[i] = x % base
            x //= base
        return d

    # Precompute for each coordinate i and value v the "offset" to change coord i to v.
    # power[i] = base**i
    power = [1] * n
    for i in range(1, n):
        power[i] = power[i - 1] * base

    # ball[i] for each integer word i: list of covered word indices.
    # Build once.
    balls = []
    for i in range(total):
        d = digits_of(i)
        b = [i]
        for pos in range(n):
            cur = d[pos]
            for v in range(base):
                if v == cur:
                    continue
                b.append(i + (v - cur) * power[pos])
        balls.append(b)

    # Convert to numpy arrays for fast coverage ops.
    balls_arr = [np.array(b, dtype=np.int64) for b in balls]

    # Greedy initialization: pick codewords to cover the whole space.
    uncovered = np.ones(total, dtype=bool)
    selected = []
    while uncovered.any():
        # For each candidate, count how many currently-uncovered words it covers.
        best_idx = -1
        best_cnt = -1
        # Iterate candidates; use vectorized count.
        for c in range(total):
            if not uncovered.any():
                break
            cnt = np.count_nonzero(uncovered[balls_arr[c]])
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = c
        if best_idx < 0:
            break
        selected.append(best_idx)
        uncovered[balls_arr[best_idx]] = False
        if time.time() > start + _DEADLINE - _MARGIN:
            break

    # Now local search to minimize |selected| while keeping full coverage.
    # Coverage count per word: how many selected codewords cover it.
    def build_cov(selected_list):
        cov = np.zeros(total, dtype=np.int64)
        for c in selected_list:
            cov[balls_arr[c]] += 1
        return cov

    cov = build_cov(selected)
    best_list = list(selected)
    best_size = len(best_list)

    def is_full(cov_arr):
        return cov_arr.min() >= 1

    # Helper: try removing index k from a list (returns new list or None)
    def try_remove(lst, k):
        # Check if removing lst[k] keeps full coverage.
        c = lst[k]
        # words only covered by c:
        only = (cov == 1) & (np.isin(np.arange(total), balls_arr[c]))
        # cheaper: recompute using cov and ball
        # A word w is uncovered after removal iff cov[w]==1 and w in ball(c)
        # So check: any w in ball(c) with cov[w]==1?
        if not (cov[balls_arr[c]] == 1).any():
            newlst = lst[:k] + lst[k + 1:]
            return newlst
        return None

    # Local search loop
    improved = True
    while improved and (time.time() < start + _DEADLINE - _MARGIN):
        improved = False
        cov = build_cov(selected)
        # Pass 1: removals
        for k in range(len(selected) - 1, -1, -1):
            c = selected[k]
            if not (cov[balls_arr[c]] == 1).any():
                # safe to remove
                cov[balls_arr[c]] -= 1
                selected = selected[:k] + selected[k + 1:]
                improved = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_list = list(selected)
        # Pass 2: swaps (replace one codeword with another to allow a removal later)
        if time.time() < start + _DEADLINE - _MARGIN:
            cov = build_cov(selected)
            for i in range(len(selected)):
                if time.time() > start + _DEADLINE - _MARGIN:
                    break
                ci = selected[i]
                # Try swapping ci with some other word j not in selected,
                # if it allows the total set to still cover and possibly shrink.
                # Cheap heuristic: only attempt if removing ci alone fails.
                if (cov[balls_arr[ci]] == 1).any():
                    # find words uniquely covered by ci
                    uniq = np.where((cov == 1))[0]
                    uniq_in_ci = [w for w in uniq if w in set(balls_arr[ci])]
                    if not uniq_in_ci:
                        continue
                    # try candidate j that covers all uniq_in_ci and is not selected
                    sel_set = set(selected)
                    for j in range(total):
                        if j in sel_set:
                            continue
                        # check j covers all unique words
                        if all(w in set(balls_arr[j]) for w in uniq_in_ci):
                            # swap: replace ci with j, then try removals
                            newlst = list(selected)
                            newlst[i] = j
                            ncov = build_cov(newlst)
                            # try to remove any now-redundant
                            changed = True
                            while changed:
                                changed = False
                                for k in range(len(newlst) - 1, -1, -1):
                                    if not (ncov[balls_arr[newlst[k]]] == 1).any():
                                        ncov[balls_arr[newlst[k]]] -= 1
                                        newlst = newlst[:k] + newlst[k + 1:]
                                        changed = True
                            if len(newlst) < len(selected):
                                selected = newlst
                                improved = True
                                if len(selected) < best_size:
                                    best_size = len(selected)
                                    best_list = list(selected)
                                break
                            # else revert
                            break

    # Ensure validity; if not fully covered, fall back to greedy completion.
    cov = build_cov(selected)
    if cov.min() < 1:
        # complete with greedy
        uncovered = cov < 1
        while uncovered.any():
            best_idx = -1
            best_cnt = -1
            for c in range(total):
                cnt = np.count_nonzero(uncovered[balls_arr[c]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = c
            if best_idx < 0:
                break
            selected.append(best_idx)
            uncovered[balls_arr[best_idx]] = False
            if time.time() > start + _DEADLINE - _MARGIN:
                break

    # Deduplicate and convert to digit lists.
    seen = set()
    final = []
    for c in selected:
        if c in seen:
            continue
        seen.add(c)
        final.append(digits_of(c))

    # Safety: if empty or invalid, use the tetracode construction as fallback.
    if not final:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return final
