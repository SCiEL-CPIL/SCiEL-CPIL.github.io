"""Term-minimal smooth-antichain representation with optimized antichain for 2-term coverage."""
import time

def _gen_smooth(limit):
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(vals)

# Precompute smooth numbers up to 128
_SMOOTH = _gen_smooth(128)
_SMOOTH_SET = set(_SMOOTH)
_N = len(_SMOOTH)
# index map for O(1) lookup
_IDX = {v: i for i, v in enumerate(_SMOOTH)}

# Precompute incompatibility matrix (a|b or b|a)
_INCOMP = [[False] * _N for _ in range(_N)]
for i in range(_N):
    a = _SMOOTH[i]
    for j in range(i + 1, _N):
        b = _SMOOTH[j]
        if a % b == 0 or b % a == 0:
            _INCOMP[i][j] = True
            _INCOMP[j][i] = True

def _incompatible_idx(i, j):
    return _INCOMP[i][j]

def _incompatible_val(a, b):
    if a == b:
        return True
    return (a % b == 0) or (b % a == 0)

# Non-smooth targets 2..128
_TARGETS = [n for n in range(2, 129) if n not in _SMOOTH_SET]

def _build_initial_antichain():
    """Greedy by descending value."""
    antichain = []
    for v in sorted(_SMOOTH, reverse=True):
        ok = True
        for u in antichain:
            if _incompatible_val(u, v):
                ok = False
                break
        if ok:
            antichain.append(v)
    return antichain

def _two_term_coverage(antichain_set, antichain_list):
    """Count non-smooth targets representable as 2-term sum from antichain."""
    count = 0
    for n in _TARGETS:
        found = False
        for a in antichain_list:
            if a >= n:
                break
            b = n - a
            if b in antichain_set and b != a and not _incompatible_val(a, b):
                found = True
                break
        if found:
            count += 1
    return count

def _build_optimized_antichain(deadline):
    """Local search to maximize 2-term coverage."""
    antichain = _build_initial_antichain()
    antichain_set = set(antichain)
    best_count = _two_term_coverage(antichain_set, antichain)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < deadline:
        improved = False
        # Try adding each smooth number not in antichain
        for v in _SMOOTH:
            if v in antichain_set:
                continue
            ok = True
            for u in antichain:
                if _incompatible_val(u, v):
                    ok = False
                    break
            if not ok:
                continue
            antichain.append(v)
            antichain_set.add(v)
            new_count = _two_term_coverage(antichain_set, antichain)
            if new_count > best_count:
                best_count = new_count
                improved = True
            else:
                antichain.pop()
                antichain_set.discard(v)
        # Try removing each element
        for v in list(antichain):
            antichain.remove(v)
            antichain_set.discard(v)
            new_count = _two_term_coverage(antichain_set, antichain)
            if new_count > best_count:
                best_count = new_count
                improved = True
            else:
                antichain.append(v)
                antichain_set.add(v)
        # Try swaps: remove u, add v
        for u in list(antichain):
            for v in _SMOOTH:
                if v in antichain_set or v == u:
                    continue
                # check v compatible with all except u
                ok = True
                for w in antichain:
                    if w == u:
                        continue
                    if _incompatible_val(w, v):
                        ok = False
                        break
                if not ok:
                    continue
                antichain.remove(u)
                antichain_set.discard(u)
                antichain.append(v)
                antichain_set.add(v)
                new_count = _two_term_coverage(antichain_set, antichain)
                if new_count > best_count:
                    best_count = new_count
                    improved = True
                else:
                    antichain.remove(v)
                    antichain_set.discard(v)
                    antichain.append(u)
                    antichain_set.add(u)
    antichain.sort()
    return antichain, best_count

def _fast_two_term(n, antichain, antichain_set):
    for a in antichain:
        if a >= n:
            break
        b = n - a
        if b <= 0 or b == a:
            continue
        if b not in antichain_set:
            continue
        if _incompatible_val(a, b):
            continue
        return [a, b]
    return None

def _fast_three_term(n, antichain, antichain_set):
    ac = antichain
    aset = antichain_set
    L = len(ac)
    for i in range(L):
        a = ac[i]
        if a >= n:
            break
        rem1 = n - a
        for j in range(i + 1, L):
            b = ac[j]
            if b >= rem1:
                break
            if _incompatible_val(a, b):
                continue
            c = rem1 - b
            if c <= b or c not in aset:
                continue
            if _incompatible_val(a, c) or _incompatible_val(b, c):
                continue
            return [a, b, c]
    return None

def _fallback_fill(n):
    """Find any valid div-free smooth representation (minimal k)."""
    if n in _SMOOTH_SET:
        return [n]
    # 2 terms from full smooth set
    for a in _SMOOTH:
        if a >= n:
            break
        b = n - a
        if b <= 0 or b == a:
            continue
        if b not in _SMOOTH_SET:
            continue
        if _incompatible_val(a, b):
            continue
        return [a, b]
    # 3 terms
    for i in range(_N):
        a = _SMOOTH[i]
        if a >= n:
            break
        for j in range(i + 1, _N):
            b = _SMOOTH[j]
            if a + b >= n:
                break
            c = n - a - b
            if c <= b or c not in _SMOOTH_SET:
                continue
            if _incompatible_val(a, b) or _incompatible_val(a, c) or _incompatible_val(b, c):
                continue
            return [a, b, c]
    # Should not reach here for 1..128, but safe fallback
    return [n]

def construct():
    start = time.time()
    DEADLINE = 80.0

    # Build optimized antichain with a time budget
    ac_budget = min(20.0, DEADLINE - 10.0)
    _ANTICHAIN, _ = _build_optimized_antichain(ac_budget)
    _ANTICHAIN_SET = set(_ANTICHAIN)

    answers = [None] * 129
    for n in range(1, 129):
        if time.time() - start > DEADLINE:
            break
        # k=1
        if n in _SMOOTH_SET:
            answers[n] = [n]
            continue
        # k=2
        found = _fast_two_term(n, _ANTICHAIN, _ANTICHAIN_SET)
        if found is not None:
            answers[n] = found
            continue
        # k=3
        found = _fast_three_term(n, _ANTICHAIN, _ANTICHAIN_SET)
        if found is not None:
            answers[n] = found
            continue
        # fallback
        answers[n] = _fallback_fill(n)

    for n in range(1, 129):
        if answers[n] is None:
            answers[n] = _fallback_fill(n)

    return answers[1:129]
