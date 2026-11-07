import time
import random
from math import isqrt

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # evaluator kills at 90s; keep a 5s safety margin


# ---------- prime-exponent structure (inspiration 3) ----------
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_PN = len(_PRIMES)
_EXP = {}          # n -> tuple of exponents over _PRIMES
_EXP_LIST = {}     # n -> list of exponents (mutable)


def _build_exp_table():
    for n in range(2, 41):
        m = n
        vec = [0] * _PN
        for i, p in enumerate(_PRIMES):
            while m % p == 0:
                vec[i] += 1
                m //= p
        # m should be 1 for n <= 40
        _EXP[n] = tuple(vec)
        _EXP_LIST[n] = vec


_build_exp_table()


# ---------- core validity machinery (current program) ----------
def _subset_products_set(nums):
    prods = {1}
    for x in nums:
        add = set()
        for p in prods:
            add.add(p * x)
        prods = prods | add
    return prods


def _can_add(prods, x):
    """True iff adding x keeps all subset products distinct."""
    for q in prods:
        if q % x == 0 and (q // x) in prods:
            return False
    return True


def _try_add(selected, prods, x):
    if not _can_add(prods, x):
        return False, selected, prods
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    return True, selected + [x], new_prods


# ---------- structural conflict (exponent-space, inspiration 3) ----------
def _exp_conflict(a, b):
    """Two numbers conflict if their exponent vectors are rationally linearly
    dependent (a subset of powers of the same prime), which guarantees a
    subset-product collision. Used only as a fast pre-filter."""
    va = _EXP_LIST[a]
    vb = _EXP_LIST[b]
    # find first non-zero component of each
    ia = -1
    for i in range(_PN):
        if va[i]:
            ia = i
            break
    ib = -1
    for i in range(_PN):
        if vb[i]:
            ib = i
            break
    if ia < 0 or ib < 0:
        return False
    if ia != ib:
        return False
    # same prime: check proportionality of the two exponent vectors
    ratio_num = va[ia]
    ratio_den = vb[ia]
    for i in range(_PN):
        if va[i] * ratio_den != vb[i] * ratio_num:
            return False
    return True


# ---------- construction ----------
def _construct_initial_set():
    """Pure-powers heuristic (known valid 16-set)."""
    return [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]


def _greedy_construct(order):
    """Greedy build following a given candidate order."""
    selected = []
    prods = {1}
    for x in order:
        if len(selected) >= 18:
            break
        if _can_add(prods, x):
            selected.append(x)
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            prods = new_prods
    return selected, prods


def _construct_random_initial_set(rng):
    selected = []
    prods = {1}
    candidates = list(range(2, 41))
    rng.shuffle(candidates)
    for x in candidates:
        if len(selected) >= 18:
            break
        if _can_add(prods, x):
            selected.append(x)
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            prods = new_prods
    return selected, prods


# ---------- local search (current program, extended) ----------
def _local_search(selected, prods, deadline, max_rounds=20):
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
        improved = False
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]

        # 1. Try adding any candidate (prefer structurally-independent ones first)
        candidates.sort(key=lambda x: sum(1 for s in selected if _exp_conflict(s, x)))
        for x in candidates:
            if time.time() > deadline:
                break
            ok, new_sel, new_prods = _try_add(selected, prods, x)
            if ok:
                selected = new_sel
                prods = new_prods
                improved = True
                used = set(selected)
                candidates = [c for c in range(2, 41) if c not in used]
                if len(selected) >= 18:
                    break

        if not improved:
            # 2. Single swaps: remove one, add one different element
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                remaining = selected[:i] + selected[i + 1:]
                rem_prods = _subset_products_set(remaining)
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if x == removed:
                        continue
                    ok1, sel1, prods1 = _try_add(remaining, rem_prods, x)
                    if ok1:
                        selected = sel1
                        prods = prods1
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            # 3. Remove one, add two (net +1)
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                remaining = selected[:i] + selected[i + 1:]
                rem_prods = _subset_products_set(remaining)
                added_any = False
                for j in range(len(candidates)):
                    if time.time() > deadline:
                        break
                    x1 = candidates[j]
                    ok1, sel1, prods1 = _try_add(remaining, rem_prods, x1)
                    if not ok1:
                        continue
                    for k in range(j + 1, len(candidates)):
                        if time.time() > deadline:
                            break
                        x2 = candidates[k]
                        ok2, sel2, prods2 = _try_add(sel1, prods1, x2)
                        if ok2 and len(sel2) > len(selected):
                            selected = sel2
                            prods = prods2
                            improved = True
                            added_any = True
                            break
                    if added_any:
                        break
                if improved:
                    break

        if not improved:
            break

    return selected, prods


def _greedy_fill(selected, prods, deadline):
    used = set(selected)
    for x in range(2, 41):
        if time.time() > deadline:
            break
        if x in used:
            continue
        if len(selected) >= 18:
            break
        ok, selected, prods = _try_add(selected, prods, x)
        if ok:
            used.add(x)
    return selected, prods


def construct():
    if time.time() > _DEADLINE:
        return [2]

    best_selected = None
    best_len = 0

    # Phase 1: strong heuristic set + local search
    selected = _construct_initial_set()
    prods = _subset_products_set(selected)
    if len(selected) < 18 and time.time() < _DEADLINE:
        selected, prods = _local_search(selected, prods, _DEADLINE)
    if time.time() < _DEADLINE and len(selected) < 18:
        selected, prods = _greedy_fill(selected, prods, _DEADLINE)
    if len(selected) > best_len:
        best_len = len(selected)
        best_selected = list(selected)

    # Phase 2: random restarts (recovered budget -> more restarts)
    rng = random.Random(42)
    remaining = _DEADLINE - time.time()
    num_restarts = max(1, int(remaining / 4.0))
    for restart in range(num_restarts):
        if time.time() > _DEADLINE - 2.0:
            break
        rsel, rprods = _construct_random_initial_set(rng)
        if len(rsel) >= 18:
            best_len = 18
            best_selected = list(rsel)
            break
        if time.time() < _DEADLINE - 1.0:
            rsel, rprods = _local_search(rsel, rprods, _DEADLINE - 1.0, max_rounds=10)
        if time.time() < _DEADLINE - 0.5 and len(rsel) < 18:
            rsel, rprods = _greedy_fill(rsel, rprods, _DEADLINE - 0.5)
        if len(rsel) > best_len:
            best_len = len(rsel)
            best_selected = list(rsel)
            if best_len >= 18:
                break

    # Ensure validity and type correctness
    selected = best_selected
    final_prods = _subset_products_set(selected)
    if len(final_prods) != 2 ** len(selected):
        selected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    seen = set()
    out = []
    for x in selected:
        if 2 <= x <= 40 and x not in seen:
            seen.add(x)
            out.append(int(x))

    if not out:
        out = [2]

    return out
