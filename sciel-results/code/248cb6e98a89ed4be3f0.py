"""Rewritten program: optimized multi-start local search for 3-AP-free subset of [1,256]
maximizing sum of reciprocals."""

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
    S_list = list(S)
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
    """Update blocked set when removing x from S. Mutates blocked in place.
    Rebuilds blocked from scratch excluding x's contributions."""
    # Simplest correct approach: rebuild
    new_blocked = set()
    S_without = S - {x}
    for a in S_without:
        for b in S_without:
            if a >= b:
                continue
            # a, b, c => c = 2b - a
            c = 2 * b - a
            if 1 <= c <= _N and c not in S_without:
                new_blocked.add(c)
            # c, a, b => c = 2a - b
            c = 2 * a - b
            if 1 <= c <= _N and c not in S_without:
                new_blocked.add(c)
            # a, c, b => c = (a+b)//2
            s = a + b
            if s % 2 == 0:
                c = s // 2
                if 1 <= c <= _N and c not in S_without:
                    new_blocked.add(c)
    blocked.clear()
    blocked.update(new_blocked)


def _local_search(S, deadline):
    """Optimize S via greedy additions and improving swaps."""
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

        # Swaps: replace a in S with b < a not in S to increase weight
        # Try all a in descending order of value (largest a first = smallest reciprocal)
        best_gain = 0.0
        best_a = None
        best_b = None

        S_sorted = sorted(S, reverse=True)
        for a in S_sorted:
            if time.time() > deadline:
                break

            # Temporarily remove a to compute what's blocked without a
            S_temp = S - {a}
            blocked_temp = _get_blocked(S_temp)

            # Find best b < a not in S_temp, not in blocked_temp
            for b in range(1, a):
                if b in S_temp or b in blocked_temp:
                    continue
                gain = _RECIP[b] - _RECIP[a]
                if gain > best_gain:
                    best_gain = gain
                    best_a = a
                    best_b = b
                break  # smallest b gives largest gain for this a

        # Also try: remove a large element and add TWO smaller ones
        if best_gain <= 1e-12:
            for a in S_sorted:
                if time.time() > deadline:
                    break
                S_temp = S - {a}
                blocked_temp = _get_blocked(S_temp)
                # Find two smallest available b1 < b2 < a
                avail = []
                for b in range(1, a):
                    if b not in S_temp and b not in blocked_temp:
                        avail.append(b)
                    if len(avail) >= 2:
                        break
                if len(avail) >= 2:
                    gain2 = _RECIP[avail[0]] + _RECIP[avail[1]] - _RECIP[a]
                    if gain2 > 1e-12:
                        # Try this 1-for-2 move
                        S.remove(a)
                        S.add(avail[0])
                        S.add(avail[1])
                        weight += gain2
                        blocked = _get_blocked(S)
                        improved = True
                        break

        if best_a is not None and best_gain > 1e-12:
            S.remove(best_a)
            S.add(best_b)
            weight += best_gain
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
        # Shuffle but keep small numbers biased toward front
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
        # Bias: sort by x**alpha with random perturbation
        alpha = 0.3 + 0.1 * seed
        weighted = sorted(order, key=lambda x: x ** alpha + rng3.random() * 2)
        inits.append(_greedy_init(weighted))

    # Run local search on each initial solution
    for S_init in inits:
        if time.time() > deadline - 5:
            break
        S_opt = _local_search(S_init, deadline)
        _evaluate(S_opt)

        # Perturbation-based restarts from best
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            S_pert = set(best_S)
            n_drop = min(3, len(S_pert) // 3)
            elems = list(S_pert)
            rng.shuffle(elems)
            for x in elems[:n_drop]:
                S_pert.discard(x)
            S_opt2 = _local_search(S_pert, deadline)
            _evaluate(S_opt2)

    # Final high-precision polish: iterative single swaps and 1-for-2 moves
    if best_S and time.time() < deadline - 1:
        S_final = set(best_S)
        weight_final = sum(_RECIP[x] for x in S_final)

        # Repeatedly try improving single swaps until no improvement
        for _round in range(20):
            if time.time() > deadline - 1:
                break
            S_sorted = sorted(S_final, reverse=True)
            any_improve = False
            for a in S_sorted:
                if time.time() > deadline - 1:
                    break
                S_temp = S_final - {a}
                blocked_temp = _get_blocked(S_temp)
                for b in range(1, a):
                    if b in S_temp or b in blocked_temp:
                        continue
                    gain = _RECIP[b] - _RECIP[a]
                    if gain > 1e-12:
                        S_final.remove(a)
                        S_final.add(b)
                        weight_final += gain
                        any_improve = True
                        break
                    break
            if not any_improve:
                break

        # Try 1-for-2 moves in final polish
        if time.time() < deadline - 1:
            S_sorted = sorted(S_final, reverse=True)
            for a in S_sorted:
                if time.time() > deadline - 1:
                    break
                S_temp = S_final - {a}
                blocked_temp = _get_blocked(S_temp)
                avail = []
                for b in range(1, a):
                    if b not in S_temp and b not in blocked_temp:
                        avail.append(b)
                    if len(avail) >= 2:
                        break
                if len(avail) >= 2:
                    gain2 = _RECIP[avail[0]] + _RECIP[avail[1]] - _RECIP[a]
                    if gain2 > 1e-12:
                        S_final.remove(a)
                        S_final.add(avail[0])
                        S_final.add(avail[1])
                        weight_final += gain2

        # Try 2-for-1 moves: remove two large elements, add one small
        if time.time() < deadline - 1:
            S_sorted = sorted(S_final, reverse=True)
            n_s = len(S_sorted)
            for i in range(n_s):
                if time.time() > deadline - 1:
                    break
                a1 = S_sorted[i]
                for j in range(i + 1, n_s):
                    a2 = S_sorted[j]
                    S_temp = S_final - {a1, a2}
                    blocked_temp = _get_blocked(S_temp)
                    for b in range(1, min(a1, a2)):
                        if b in S_temp or b in blocked_temp:
                            continue
                        gain = _RECIP[b] - _RECIP[a1] - _RECIP[a2]
                        if gain > 1e-12:
                            S_final.remove(a1)
                            S_final.remove(a2)
                            S_final.add(b)
                            weight_final += gain
                            break
                        break

        if weight_final > best_w:
            best_S = S_final
            best_w = weight_final

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
