"""Rewritten program: optimized multi-start local search for 3-AP-free subset of [1,256]
maximizing sum of reciprocals. Enhanced with full-swap search, double swaps, and better polish."""

import time
import random
from math import log

_N = 256
_KILL = 85.0
_MARGIN = 3.0

# Precompute reciprocals
_RECIP = [0.0] + [1.0 / i for i in range(1, _N + 1)]


def _get_blocked(S):
    """Compute the blocked set: elements that would form a 3-AP with two elements of S."""
    blocked = set()
    S_list = sorted(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            # AP: a, b, c => c = 2b - a
            c = 2 * b - a
            if 1 <= c <= _N:
                blocked.add(c)
            # AP: c, a, b => c = 2a - b
            c = 2 * a - b
            if 1 <= c <= _N:
                blocked.add(c)
            # AP: a, c, b => c = (a+b)//2
            s = a + b
            if s % 2 == 0:
                c = s // 2
                if 1 <= c <= _N:
                    blocked.add(c)
    return blocked


def _add_to_blocked(blocked, S, x):
    """Update blocked set when adding x to S. Mutates blocked in place."""
    for a in S:
        if a == x:
            continue
        # x, a, c => c = 2a - x
        c = 2 * a - x
        if 1 <= c <= _N:
            blocked.add(c)
        # a, x, c => c = 2x - a
        c = 2 * x - a
        if 1 <= c <= _N:
            blocked.add(c)
        # a, c, x => c = (a+x)//2
        s = a + x
        if s % 2 == 0:
            c = s // 2
            if 1 <= c <= _N:
                blocked.add(c)
    blocked.discard(x)


def _remove_from_blocked(blocked, S, x):
    """Remove contributions of x from blocked set. Mutates blocked in place.
    Note: This is an approximation - we recompute fully when needed."""
    pass  # We'll just recompute blocked when removing


def _local_search(S, deadline, full_swap=False):
    """Optimize S via greedy additions and improving swaps.
    If full_swap=True, consider all valid b for each a (not just smallest)."""
    S = set(S)
    blocked = _get_blocked(S)
    weight = sum(_RECIP[x] for x in S)

    while time.time() < deadline:
        improved = False

        # Greedy additions: add smallest valid elements
        added_any = False
        for x in range(1, _N + 1):
            if x not in S and x not in blocked:
                S.add(x)
                weight += _RECIP[x]
                _add_to_blocked(blocked, S, x)
                added_any = True

        if added_any:
            continue

        # Swaps: replace a in S with b not in S to increase weight
        best_gain = 0.0
        best_a = None
        best_b = None

        S_list = sorted(S, reverse=True)  # large a first (low reciprocal)
        for a in S_list:
            if time.time() > deadline:
                break

            S_temp = S - {a}
            blocked_temp = _get_blocked(S_temp)

            # Find best b not in S_temp, not in blocked_temp, b != a
            if full_swap:
                # Consider all valid b, pick the one with max gain
                for b in range(1, _N + 1):
                    if b == a or b in S_temp or b in blocked_temp:
                        continue
                    gain = _RECIP[b] - _RECIP[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_a = a
                        best_b = b
            else:
                # Smallest valid b gives largest gain for this a
                for b in range(1, _N + 1):
                    if b == a or b in S_temp or b in blocked_temp:
                        continue
                    gain = _RECIP[b] - _RECIP[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_a = a
                        best_b = b
                    break  # smallest b gives largest gain for this a

        if best_a is not None and best_gain > 1e-12:
            S.remove(best_a)
            S.add(best_b)
            weight += best_gain
            blocked = _get_blocked(S)
            improved = True

        # Try double swap: remove two elements, add two others
        if not improved and time.time() < deadline:
            best_dgain = 0.0
            best_remove = None
            best_add = None

            S_list2 = sorted(S, reverse=True)
            n = len(S_list2)
            # Only try pairs from the larger half to limit computation
            start_idx = max(0, n // 2)
            for i in range(start_idx, n):
                if time.time() > deadline:
                    break
                a1 = S_list2[i]
                for j in range(i + 1, n):
                    a2 = S_list2[j]
                    S_temp = S - {a1, a2}
                    blocked_temp = _get_blocked(S_temp)

                    # Find two best elements to add
                    candidates = []
                    for b in range(1, _N + 1):
                        if b in S_temp or b in blocked_temp:
                            continue
                        candidates.append(b)
                        if len(candidates) >= 5:  # Only check top few
                            break

                    if len(candidates) >= 2:
                        # Try all pairs from candidates
                        for k in range(len(candidates)):
                            for l in range(k + 1, len(candidates)):
                                b1, b2 = candidates[k], candidates[l]
                                gain = _RECIP[b1] + _RECIP[b2] - _RECIP[a1] - _RECIP[a2]
                                if gain > best_dgain:
                                    best_dgain = gain
                                    best_remove = (a1, a2)
                                    best_add = (b1, b2)

            if best_remove is not None and best_dgain > 1e-12:
                S.discard(best_remove[0])
                S.discard(best_remove[1])
                S.add(best_add[0])
                S.add(best_add[1])
                weight += best_dgain
                blocked = _get_blocked(S)
                improved = True

        if not improved:
            break

    return S


def _greedy_init(order):
    """Build a 3-AP-free set by greedily adding elements in given order."""
    S = set()
    blocked = set()
    for x in order:
        if x not in blocked:
            S.add(x)
            _add_to_blocked(blocked, S, x)
    return S


def _validate(S):
    """Check S is 3-AP-free."""
    Sset = set(S)
    for a in Sset:
        for b in Sset:
            if a == b:
                continue
            c = 2 * b - a
            if c in Sset and c != a and c != b:
                return False
    return True


def _polish(S, deadline):
    """Aggressive polish: try all single swaps (full search) and double swaps repeatedly."""
    S = set(S)
    weight = sum(_RECIP[x] for x in S)

    for polish_iter in range(50):
        if time.time() > deadline - 0.5:
            break
        improved = False

        # Single swap: full search over all a, all b
        best_gain = 0.0
        best_a = None
        best_b = None

        S_list = sorted(S, reverse=True)
        for a in S_list:
            if time.time() > deadline - 0.5:
                break
            S_temp = S - {a}
            blocked_temp = _get_blocked(S_temp)

            for b in range(1, _N + 1):
                if b == a or b in S_temp or b in blocked_temp:
                    continue
                gain = _RECIP[b] - _RECIP[a]
                if gain > best_gain:
                    best_gain = gain
                    best_a = a
                    best_b = b

        if best_a is not None and best_gain > 1e-12:
            S.remove(best_a)
            S.add(best_b)
            weight += best_gain
            improved = True

        # Double swap
        if not improved and time.time() < deadline - 0.5:
            best_dgain = 0.0
            best_remove = None
            best_add = None

            S_list2 = sorted(S, reverse=True)
            n = len(S_list2)
            start_idx = max(0, n // 3)
            for i in range(start_idx, n):
                if time.time() > deadline - 0.5:
                    break
                a1 = S_list2[i]
                for j in range(i + 1, n):
                    a2 = S_list2[j]
                    S_temp = S - {a1, a2}
                    blocked_temp = _get_blocked(S_temp)

                    candidates = []
                    for b in range(1, _N + 1):
                        if b in S_temp or b in blocked_temp:
                            continue
                        candidates.append(b)
                        if len(candidates) >= 8:
                            break

                    if len(candidates) >= 2:
                        for k in range(len(candidates)):
                            for l in range(k + 1, len(candidates)):
                                b1, b2 = candidates[k], candidates[l]
                                gain = _RECIP[b1] + _RECIP[b2] - _RECIP[a1] - _RECIP[a2]
                                if gain > best_dgain:
                                    best_dgain = gain
                                    best_remove = (a1, a2)
                                    best_add = (b1, b2)

            if best_remove is not None and best_dgain > 1e-12:
                S.discard(best_remove[0])
                S.discard(best_remove[1])
                S.add(best_add[0])
                S.add(best_add[1])
                weight += best_dgain
                improved = True

        if not improved:
            break

    return S


def construct():
    start = time.time()
    deadline = start + _KILL - _MARGIN
    rng = random.Random(42)

    best_S = None
    best_w = -1.0

    def _evaluate(S):
        nonlocal best_S, best_w
        w = sum(_RECIP[x] for x in S)
        if w > best_w:
            best_w = w
            best_S = set(S)
        return w

    # Generate diverse initial solutions
    inits = []

    # 1. Greedy ascending
    inits.append(_greedy_init(range(1, _N + 1)))

    # 2. Greedy by priority: prefer small numbers but with some randomness
    for seed in range(8):
        rng2 = random.Random(1000 + seed)
        order = list(range(1, _N + 1))
        weighted_order = []
        for i, x in enumerate(order):
            weighted_order.append((x ** 0.5 + rng2.random() * 5, x))
        weighted_order.sort(key=lambda t: t[0])
        inits.append(_greedy_init([x for _, x in weighted_order]))

    # 3. Cantor set (base-3 digits 0,1 only)
    S_cantor = set()
    for i in range(_N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid and 1 <= i <= _N:
            S_cantor.add(i)
    inits.append(S_cantor)

    # 4. Random greedy with different seeds
    for seed in range(10):
        rng3 = random.Random(5000 + seed)
        order = list(range(1, _N + 1))
        rng3.shuffle(order)
        alpha = 0.3 + 0.1 * seed
        weighted = sorted(order, key=lambda x: x ** alpha + rng3.random() * 2)
        inits.append(_greedy_init(weighted))

    # 5. Additional diverse inits with different bias parameters
    for seed in range(5):
        rng4 = random.Random(9000 + seed)
        order = list(range(1, _N + 1))
        alpha = 0.5 + 0.1 * seed
        noise_scale = 1.0 + 0.5 * seed
        weighted = sorted(order, key=lambda x: x ** alpha + rng4.random() * noise_scale)
        inits.append(_greedy_init(weighted))

    # 6. Greedy descending (start with large, then fill in small)
    inits.append(_greedy_init(range(_N, 0, -1)))

    # 7. Mixed: alternate small and large
    order_mixed = []
    small = list(range(1, _N // 2 + 1))
    large = list(range(_N // 2 + 1, _N + 1))
    for i in range(_N // 2):
        order_mixed.append(small[i])
        order_mixed.append(large[i])
    inits.append(_greedy_init(order_mixed))

    # Run local search on each initial solution
    for idx, S_init in enumerate(inits):
        if time.time() > deadline - 10:
            break
        # Use full_swap for the first few to get better local optima
        use_full = idx < 5
        S_opt = _local_search(S_init, deadline, full_swap=use_full)
        _evaluate(S_opt)

        # Perturbation-based restarts from best
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            S_pert = set(best_S)
            n_drop = min(4, len(S_pert) // 4)
            elems = list(S_pert)
            rng.shuffle(elems)
            for x in elems[:n_drop]:
                S_pert.discard(x)
            S_opt2 = _local_search(S_pert, deadline, full_swap=True)
            _evaluate(S_opt2)

    # Final aggressive polish on best_S
    if best_S and time.time() < deadline - 1:
        S_final = _polish(set(best_S), deadline)
        w_final = sum(_RECIP[x] for x in S_final)
        if w_final > best_w:
            best_S = S_final
            best_w = w_final

    if not best_S:
        best_S = {1}

    res = sorted(best_S)

    # Defensive verification
    Sset = set(res)
    ok = True
    for a in Sset:
        for b in Sset:
            if a == b:
                continue
            c = 2 * b - a
            if c in Sset and c != a and c != b:
                ok = False
                break
        if not ok:
            break

    if not ok:
        # Fallback to simple greedy
        S = _greedy_init(range(1, _N + 1))
        res = sorted(S)

    return res
