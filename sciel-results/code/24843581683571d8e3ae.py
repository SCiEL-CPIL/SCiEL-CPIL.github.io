"""Rewritten program: exact longest conflict-free interval via two-pointer over inverse pairs."""
# EVOLVE-BLOCK-START
import time

_MOD = 1009
_LO, _HI = 1, 1008

def _pairs():
    seen = set()
    res = []
    for x in range(_LO, _HI + 1):
        if x in seen:
            continue
        inv = pow(x, _MOD - 2, _MOD)
        if inv == x:
            seen.add(x)
        else:
            seen.add(x)
            seen.add(inv)
            res.append((min(x, inv), max(x, inv)))
    res.sort()
    return res

def _solve(pairs):
    n = len(pairs)
    L = 1
    best = (1, 1)  # (start, length)
    for R in range(2, _HI + 1):
        # advance L past any pair that now spans [L, R]
        while L <= R:
            a, b = pairs[0]
            if b < L:
                pairs.pop(0)
                continue
            if a < L:
                # a is left of L, so this pair is no longer a constraint
                pairs.pop(0)
                continue
            # a >= L and b >= R (since b < R was popped) => conflict
            L = a + 1
            continue
        if R - L + 1 > best[1]:
            best = (L, R - L + 1)
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 80.0
    pairs = _pairs()
    ans = _solve(pairs)
    if time.time() > deadline:
        return [ans[0], ans[1]]
    return [ans[0], ans[1]]
# EVOLVE-BLOCK-END
