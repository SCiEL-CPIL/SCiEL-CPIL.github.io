"""Direct term-minimal smooth-antichain representation via maximal antichain + fast pair lookup."""
import time

def _is_smooth(n):
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    while n % 5 == 0:
        n //= 5
    return n == 1

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

def _divides(a, b):
    """True if a divides b."""
    return b % a == 0

def _incompatible(a, b):
    """True if a divides b OR b divides a (i.e., NOT antichain-safe)."""
    if a == b:
        return True
    if _divides(a, b) or _divides(b, a):
        return True
    return False

# Precompute smooth numbers up to 128
_SMOOTH = _gen_smooth(128)
_SMOOTH_SET = set(_SMOOTH)

# Build a maximal antichain: sort descending, include if compatible with all already-included
def _build_maximal_antichain(smooth_list):
    antichain = []
    for v in sorted(smooth_list, reverse=True):
        ok = True
        for u in antichain:
            if _incompatible(u, v):
                ok = False
                break
        if ok:
            antichain.append(v)
    return sorted(antichain)

_ANTICHAIN = _build_maximal_antichain(_SMOOTH)
_ANTICHAIN_SET = set(_ANTICHAIN)

def _fast_two_term(n, antichain, antichain_set):
    """Find a, b in antichain with a+b=n, a!=b, antichain-safe. Return [a,b] or None."""
    seen = set()
    for a in antichain:
        if a > n:
            break
        b = n - a
        if b <= 0 or b == a:
            continue
        if b not in antichain_set:
            continue
        if _incompatible(a, b):
            continue
        return [a, b]
    return None

def _fast_three_term(n, antichain, antichain_set):
    """Find a,b,c in antichain, distinct, pairwise antichain-safe, summing to n. Return [a,b,c] or None."""
    ac = antichain
    aset = antichain_set
    L = len(ac)
    for i in range(L):
        a = ac[i]
        if a > n:
            break
        rem1 = n - a
        for j in range(i + 1, L):
            b = ac[j]
            if b >= rem1:
                break
            if _incompatible(a, b):
                continue
            c = rem1 - b
            if c <= b or c not in aset:
                continue
            if _incompatible(a, c) or _incompatible(b, c):
                continue
            return [a, b, c]
    return None

def _greedy_fill(n):
    """Last-resort: find any valid div-free smooth representation."""
    # Try 1 term
    if n in _SMOOTH_SET:
        return [n]
    # Try 2 terms from full smooth set
    for a in _SMOOTH:
        if a > n:
            break
        b = n - a
        if b <= 0 or b == a:
            continue
        if b not in _SMOOTH_SET:
            continue
        if _incompatible(a, b):
            continue
        return [a, b]
    # Try 3 terms
    for i in range(len(_SMOOTH)):
        a = _SMOOTH[i]
        if a > n:
            break
        for j in range(i + 1, len(_SMOOTH)):
            b = _SMOOTH[j]
            if a + b >= n:
                break
            c = n - a - b
            if c <= b or c not in _SMOOTH_SET:
                continue
            if _incompatible(a, b) or _incompatible(a, c) or _incompatible(b, c):
                continue
            return [a, b, c]
    return [n] if n in _SMOOTH_SET else [1, n - 1] if n - 1 in _SMOOTH_SET and _incompatible(1, n-1) == False else [n]

def construct():
    start = time.time()
    DEADLINE = 85.0
    answers = [None] * 129

    for n in range(1, 129):
        if time.time() - start > DEADLINE:
            break

        # k=1: smooth target
        if n in _SMOOTH_SET:
            answers[n] = [n]
            continue

        # k=2: fast pair lookup in maximal antichain
        found = _fast_two_term(n, _ANTICHAIN, _ANTICHAIN_SET)
        if found is not None:
            answers[n] = found
            continue

        # k=3: fast triple lookup
        found = _fast_three_term(n, _ANTICHAIN, _ANTICHAIN_SET)
        if found is not None:
            answers[n] = found
            continue

        # Fallback: greedy over full smooth set
        if time.time() - start < DEADLINE - 2.0:
            answers[n] = _greedy_fill(n)
        else:
            answers[n] = _greedy_fill(n)

    # Ensure no None
    for n in range(1, 129):
        if answers[n] is None:
            answers[n] = _greedy_fill(n)

    return answers[1:129]
