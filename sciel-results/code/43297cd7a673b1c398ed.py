"""Closest strict underpacking of 1 using unit fractions 1/2..1/48."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

_DENOMS = list(range(2, 49))
_N = len(_DENOMS)
_ONE = Fraction(1, 1)
_TIME_LIMIT = 60.0
_DEADLINE = time.time() + _TIME_LIMIT


def _valid(idx):
    s = Fraction()
    for i in idx:
        s += Fraction(1, _DENOMS[i])
    return s < _ONE


def _greedy():
    """Greedy: add smallest denominator that keeps sum < 1."""
    chosen, total = [], Fraction()
    for i in range(_N):
        cand = total + Fraction(1, _DENOMS[i])
        if cand < _ONE:
            chosen.append(i)
            total = cand
    return chosen, total


def _initial():
    """Stronger start: greedy by ascending denominator, then try removing
    single large denominators to tighten the deficit."""
    chosen, total = _greedy()
    best = (chosen, total, _ONE - total)
    # Try removing one element to see if deficit improves (rarely, but cheap)
    for k in range(len(chosen)):
        c2 = chosen[:k] + chosen[k + 1:]
        t2 = total - Fraction(1, _DENOMS[chosen[k]])
        if t2 < _ONE:
            d = _ONE - t2
            if d < best[2]:
                best = (c2, t2, d)
    return best


def _local_search(chosen, total, deficit):
    """1-swap local search: replace one chosen denominator with an unchosen
    one if it lowers the deficit (stays strictly below 1)."""
    chosen = list(chosen)
    total = Fraction(total)
    deficit = Fraction(deficit)
    best_idx = list(chosen)
    best_total = total
    best_def = deficit
    improved = True
    while improved:
        improved = False
        chosen_set = set(chosen)
        for pos in range(len(chosen)):
            cur_i = chosen[pos]
            cur_val = Fraction(1, _DENOMS[cur_i])
            for j in range(_N):
                if j in chosen_set:
                    continue
                # new deficit = old deficit - (1/d_j - 1/d_cur)
                delta = Fraction(1, _DENOMS[j]) - cur_val
                new_def = deficit - delta
                if new_def <= 0:
                    continue  # would reach or exceed 1
                if new_def < best_def:
                    new_chosen = chosen[:pos] + [j] + chosen[pos + 1:]
                    new_total = total - cur_val + Fraction(1, _DENOMS[j])
                    best_idx = new_chosen
                    best_total = new_total
                    best_def = new_def
                    improved = True
                    # restart from the improved solution
                    chosen = best_idx
                    total = best_total
                    deficit = best_def
                    chosen_set = set(chosen)
                    break
            if improved:
                break
        if time.time() > _DEADLINE:
            break
    return best_idx, best_total, best_def


def _pair_improve(chosen, total, deficit):
    """Try 2-exchange: remove two chosen, add two unchosen, if deficit drops."""
    chosen = list(chosen)
    best_idx = list(chosen)
    best_total = Fraction(total)
    best_def = Fraction(deficit)
    n = len(chosen)
    chosen_set = set(chosen)
    unchosen = [j for j in range(_N) if j not in chosen_set]
    for a in range(n):
        for b in range(a + 1, n):
            ra = Fraction(1, _DENOMS[chosen[a]])
            rb = Fraction(1, _DENOMS[chosen[b]])
            for x in unchosen:
                for y in unchosen:
                    if y <= x:
                        continue
                    rx = Fraction(1, _DENOMS[x])
                    ry = Fraction(1, _DENOMS[y])
                    new_def = best_def - (rx + ry - ra - rb)
                    if new_def <= 0:
                        continue
                    if new_def < best_def:
                        new_chosen = [c for k, c in enumerate(chosen) if k != a and k != b]
                        new_chosen.extend([x, y])
                        new_total = best_total - ra - rb + rx + ry
                        best_idx = new_chosen
                        best_total = new_total
                        best_def = new_def
                        chosen_set = set(chosen)
    return best_idx, best_total, best_def


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    best_idx, best_total, best_def = _initial()
    # Local search (1-swap)
    best_idx, best_total, best_def = _local_search(best_idx, best_total, best_def)
    if time.time() < _DEADLINE - 5:
        # 2-exchange refinement
        best_idx, best_total, best_def = _pair_improve(best_idx, best_total, best_def)
        # one more 1-swap pass after 2-exchange
        best_idx, best_total, best_def = _local_search(best_idx, best_total, best_def)
    # Safety: ensure validity and nonempty
    if not best_idx:
        return [0]
    if not _valid(best_idx):
        # fall back to a safe greedy
        idx, _ = _greedy()
        return idx
    return best_idx
# EVOLVE-BLOCK-END
