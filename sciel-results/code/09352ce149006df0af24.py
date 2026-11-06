"""Find the maximum-size dissociated subset of [1..22].

A subset S is dissociated (reciprocal) iff every subset of S has a distinct
sum of reciprocals: the map A -> sum_{a in A} 1/a is injective over all 2^|S|
sub-sums (including the empty sum = 0).

Strategy:
  - Exact integer arithmetic via LCM(1..22).
  - Fast modular pre-check to quickly reject candidates.
  - Bitset-based collision checking for fast subset sum tracking.
  - Greedy construction with multiple randomized orders.
  - Local search with add/swap moves.
  - Global deadline guard.
"""
import time
import random
import numpy as np
from math import gcd
from functools import reduce

_T0 = time.time()
_BUDGET = 85.0  # Wall-clock guard (evaluator kills at 90s; keep 5s margin)
_N = 22

# LCM(1..22) for exact integer representation of reciprocal sums
def _lcm(a, b):
    return a // gcd(a, b) * b

_LCM = reduce(_lcm, range(1, _N + 1))
# Reciprocal of i as integer: LCM // i
_REC = [0] + [_LCM // i for i in range(1, _N + 1)]

# Modular representations for fast pre-checks
# Use several primes for modular collision detection
_MODS = [1000003, 1000033, 1000037]
_REC_MOD = [[_REC[i] % m for i in range(1, _N + 1)] for m in _MODS]


def _is_dissociated(elements):
    """Independent verification: all 2^k reciprocal sub-sums distinct."""
    sums = {0}
    for e in elements:
        v = _REC[e]
        shifted = {s + v for s in sums}
        if shifted & sums:
            return False
        sums = sums | shifted
    return True


def _is_dissociated_modular(elements):
    """Fast modular pre-check. Returns True if possibly dissociated."""
    for mi, mod in enumerate(_MODS):
        sums = {0}
        for e in elements:
            v = _REC_MOD[mi][e - 1]
            shifted = set()
            for s in sums:
                ns = (s + v) % mod
                if ns in sums:
                    return False
                shifted.add(ns)
            sums |= shifted
    return True


def _greedy_construct(order):
    """Greedy construction: pick elements in 'order' if they fit."""
    current_set = []
    sums = {0}
    for idx in order:
        v = _REC[idx]
        shifted = {s + v for s in sums}
        if not (shifted & sums):
            current_set.append(idx)
            sums |= shifted
    return current_set, sums


def _try_add(current_set, sums, candidate):
    """Try to add candidate to current_set. Returns (new_set, new_sums) or (None, None)."""
    v = _REC[candidate]
    shifted = {s + v for s in sums}
    if shifted & sums:
        return None, None
    new_sums = sums | shifted
    new_set = current_set + [candidate]
    return new_set, new_sums


def _local_search(current_set, sums, deadline):
    """Try to improve current_set by adding elements or swapping."""
    unused = set(range(1, _N + 1)) - set(current_set)
    
    # Phase 1: Try to add any unused element
    for cand in list(unused):
        if time.time() > deadline:
            break
        new_set, new_sums = _try_add(current_set, sums, cand)
        if new_set is not None:
            return new_set, new_sums, True

    # Phase 2: Try swaps (remove one, add one)
    for _ in range(20):
        if time.time() > deadline:
            break
        
        unused_list = list(unused)
        current_list = list(current_set)
        random.shuffle(unused_list)
        random.shuffle(current_list)
        
        improved = False
        for rem in current_list:
            if time.time() > deadline:
                break
            for add in unused_list:
                if time.time() > deadline:
                    break
                new_set = [x for x in current_set if x != rem] + [add]
                if _is_dissociated_modular(new_set):
                    # Verify with exact arithmetic
                    if _is_dissociated(new_set):
                        # Recompute sums
                        sums_new = {0}
                        for e in new_set:
                            v = _REC[e]
                            sums_new |= {s + v for s in sums_new}
                        return new_set, sums_new, True
        
        if not improved:
            break
            
    return current_set, sums, False


def construct():
    """Return a nonempty subset of [1..22] maximising dissociated cardinality."""
    deadline = _T0 + _BUDGET
    
    # Initial baseline: Greedy descending
    best_set, best_sums = _greedy_construct(list(range(_N, 0, -1)))
    best_size = len(best_set)
    
    # Ensure we have a valid nonempty set
    if not best_set:
        best_set = [1]
        best_sums = {0, _REC[1]}
        best_size = 1

    # Randomized restarts
    while time.time() < deadline - 3.0:
        # Generate a random order
        order = list(range(1, _N + 1))
        random.shuffle(order)
        
        # Greedy construct
        curr_set, curr_sums = _greedy_construct(order)
        
        # If this greedy result is better, update best
        if len(curr_set) > best_size:
            best_set = curr_set
            best_sums = curr_sums
            best_size = len(best_set)

        # Try to improve the current best
        if time.time() < deadline - 8.0:
            new_set, new_sums, improved = _local_search(best_set, best_sums, deadline)
            if improved:
                best_set = new_set
                best_sums = new_sums
                best_size = len(best_set)

    # Final verification
    if not _is_dissociated(best_set):
        return [1]
        
    return best_set
