"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_N = 256
_DEADLINE = 80.0  # safety margin vs 90s kill
_BIG = 10**9


def _max_ap_abs(s):
    """Return the max absolute signed sum over ANY arithmetic progression in [1,N].

    s: list/array of 256 values in {-1,+1}, 0-indexed (position i is index i).
    For step d, progressions are contiguous sub-arrays within each residue class.
    The max absolute sum over such sub-arrays equals max(prefix)-min(prefix) per class.
    """
    n = len(s)
    worst = 0
    # Steps 1..n//2: each has at least 2 elements in some class -> use prefix trick.
    # Step d: there are d residue classes, class r has positions r, r+d, ...
    # For d > n//2, every class has at most 1 element -> handled by brute force below.
    half = n // 2
    for d in range(1, half + 1):
        # iterate over residue classes
        for r in range(d):
            # prefix sums over this class
            # collect in a tight loop
            # use local vars for speed
            cur = 0
            mn = 0
            mx = 0
            # walk the class
            # positions: r, r+d, r+2d, ...
            # unrolled-ish loop
            i = r
            # first element
            cur = s[i]
            mn = cur if cur < 0 else 0
            mx = cur if cur > 0 else 0
            # Actually we want max-min over prefix sums INCLUDING 0.
            # prefix[0]=0, prefix[k]=sum of first k elements.
            # max-min over {0, p1, p2, ...}
            mn = 0
            mx = 0
            cur = 0
            i = r
            while i < n:
                cur += s[i]
                if cur > mx:
                    mx = cur
                elif cur < mn:
                    mn = cur
                i += d
            span = mx - mn
            if span > worst:
                worst = span
    # Steps d > n//2: each class has at most 2 elements (only when d<=n and r+d<n).
    # For d in (n//2, n): class r has elements r and r+d if r+d < n.
    # So the only progressions of length>=2 with such step are pairs (r, r+d) with r+d<n,
    # i.e. r < n-d. Also singletons (|1|). Pairs have sum s[r]+s[r+d] in {-2,0,2}.
    # So worst over these is 2 if any pair sums to +/-2, else 1 (singletons).
    # We can just check: does there exist r,d with d>n//2 and r+d<n and s[r]==s[r+d]?
    # That's equivalent to: is there any pair (a,b) with b-a>n//2 and a<b<n and s[a]==s[b]?
    # Max b-a is n-1. So we need same sign at distance > n//2.
    # Simplest: for each r in 0..n-1, for d in n//2+1..n-1-r: check.
    # But that's O(n^2/2) ~ 32k, fine.
    # Even simpler: for each pair distance dd in n//2+1..n-1, check any r with s[r]==s[r+dd].
    # We already know singletons give 1. So worst is at least 1.
    # Determine if 2 achievable:
    # We can short-circuit: if worst>=2 already return worst.
    if worst < 2:
        # check for a same-sign pair at distance > n//2
        found = False
        for dd in range(half + 1, n):
            # r from 0 to n-dd-1
            # check s[r]==s[r+dd] for any r
            for r in range(n - dd):
                if s[r] == s[r + dd]:
                    found = True
                    break
            if found:
                break
        if found:
            worst = 2
        else:
            worst = 1  # singletons
    return worst


def _try_flip(s, idx, deadline, rng):
    """Attempt to flip s[idx]; return (improved, new_worst) using quick check."""
    old = s[idx]
    s[idx] = -old
    # quick full eval
    w = _max_ap_abs(s)
    s[idx] = old
    return w


def _local_search(s, deadline, rng, max_iter=None):
    """Hill-climb / anneal: flip single signs to reduce worst AP sum."""
    n = len(s)
    cur_w = _max_ap_abs(s)
    best_w = cur_w
    best = s[:]
    iters = 0
    # temperature schedule
    T = 1.0
    while time.time() < deadline:
        iters += 1
        idx = rng.randrange(n)
        old = s[idx]
        s[idx] = -old
        new_w = _max_ap_abs(s)
        if new_w <= cur_w or rng.random() < min(1.0, T):
            cur_w = new_w
            if new_w < best_w:
                best_w = new_w
                best = s[:]
                # cool a bit when improving
                T = max(0.05, T * 0.97)
        else:
            s[idx] = old
        # slow cooling
        T = max(0.02, T * 0.99995)
        if max_iter is not None and iters >= max_iter:
            break
    return best, best_w


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)
    n = _N

    best = None
    best_w = _BIG

    # Multi-restart simulated annealing
    restart = 0
    while time.time() < deadline - 3.0:  # keep margin for final eval
        # random init, but bias: start from a low-discrepancy-ish pattern
        if restart == 0:
            # Thue-Morse-like or simple alternating to start near optimal
            # Use a deterministic good-ish seed: alternating
            s = [1 if (i & 1) == 0 else -1 for i in range(n)]
        else:
            s = [rng.choice((-1, 1)) for _ in range(n)]
        s = s[:]
        b, w = _local_search(s, min(deadline - 2.0, time.time() + 12.0), rng)
        if w < best_w:
            best_w = w
            best = b[:]
            if best_w <= 2:
                # can't do better than 2 (singletons give 1, but 2 likely minimal feasible)
                # Actually minimum possible is at least 1. Try to push to 1? 1 means every AP sum in {-1,0,1}.
                # Hard to beat 2; but keep searching a little more.
                pass
        restart += 1

    # Final safety: ensure we return a valid list of 256 signs
    if best is None:
        best = [1 if (i & 1) == 0 else -1 for i in range(n)]

    # Verify length and values
    assert len(best) == n
    # Return as Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
