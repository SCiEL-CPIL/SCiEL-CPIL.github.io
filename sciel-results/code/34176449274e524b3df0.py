"""Maximize cardinality of a reciprocal-subsum-free subset of [1,24].

Design: greedy construction with a maintained set of achievable
reciprocal subset-sums (scaled to integers via LCM), multiple insertion
orders, and a local-improvement pass (single add / swap) to escape
greedy traps. Exact arithmetic throughout.
"""
import math
import time
import random
from itertools import permutations

# LCM(1..24) so that 1/k maps to the integer LCM // k
_LCM_24 = 5354228880


def _weights():
    # w[k] = LCM // k for k in 1..24
    return [0] + [_LCM_24 // k for k in range(1, 25)]


def _build_sums(chosen, W):
    """Return the set of all subset-sums of {W[k] for k in chosen}."""
    sums = {0}
    for k in chosen:
        w = W[k]
        sums = sums | {s + w for s in sums}
    return sums


def _greedy(order, W):
    """Greedy: walk through `order` (a permutation of 1..24); add k iff
    W[k] is not a subset-sum of the currently chosen weights.
    Returns (chosen_sorted, sums)."""
    sums = {0}
    chosen = []
    for k in order:
        w = W[k]
        if w not in sums:
            chosen.append(k)
            sums = sums | {s + w for s in sums}
    return sorted(chosen), sums


def _local_improve(chosen, W, deadline):
    """Try single-add and single-swap moves to grow the set.
    Returns (improved_chosen, improved_sums)."""
    chosen = list(chosen)
    sums = _build_sums(chosen, W)
    chosen_set = set(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # --- single add ---
        best_add = None
        for k in range(1, 25):
            if k in chosen_set:
                continue
            if W[k] not in sums:
                # adding k is valid
                best_add = k
                break
        if best_add is not None:
            k = best_add
            chosen.append(k)
            chosen_set.add(k)
            w = W[k]
            sums = sums | {s + w for s in sums}
            improved = True
            continue
        # --- single swap: remove r, add k ---
        # For each pair (r in chosen, k not in chosen), check validity.
        # After removing r, the sums change. We need W[k] not in
        # subset-sums of (chosen - {r}).
        # To be efficient, precompute for each r the sums without r.
        # But that's O(n * 2^n) which is fine for n <= 24.
        # Simpler: for each candidate (r, k), rebuild sums without r and check.
        # Limit work by only trying promising k (small W[k], i.e. large k).
        candidates = [k for k in range(24, 0, -1) if k not in chosen_set]
        found_swap = False
        for r in list(chosen):
            if found_swap:
                break
            for k in candidates:
                if time.time() >= deadline:
                    return chosen, sums
                # sums without r: we need subset-sums of {W[j] for j in chosen, j != r}
                # Recompute is O(2^(n-1)) which for n~15 is ~16384, fine.
                others = [j for j in chosen if j != r]
                sums_no_r = {0}
                for j in others:
                    wj = W[j]
                    sums_no_r = sums_no_r | {s + wj for s in sums_no_r}
                if W[k] not in sums_no_r:
                    # valid swap
                    chosen.remove(r)
                    chosen_set.discard(r)
                    chosen.append(k)
                    chosen_set.add(k)
                    sums = sums_no_r | {s + W[k] for s in sums_no_r}
                    improved = True
                    found_swap = True
                    break
        if not found_swap and best_add is None:
            break
    return sorted(chosen), sums


def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    W = _weights()
    rng = random.Random(42)

    best = None
    best_size = -1

    # Phase 1: try several deterministic and random insertion orders
    orders = []
    # Descending (largest first) — tends to pick large numbers
    orders.append(list(range(24, 0, -1)))
    # Ascending
    orders.append(list(range(1, 25)))
    # Random orders
    for _ in range(200):
        if time.time() >= deadline - 10:
            break
        o = list(range(1, 25))
        rng.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() >= deadline - 5:
            break
        chosen, sums = _greedy(order, W)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = list(chosen)

    # Phase 2: local improvement on the best greedy result
    if best is not None and time.time() < deadline - 2:
        improved, _ = _local_improve(best, W, deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)

    # Phase 3: try local improvement on a few other greedy results
    if time.time() < deadline - 2:
        # Re-run a few random orders and improve
        for _ in range(50):
            if time.time() >= deadline - 2:
                break
            o = list(range(1, 25))
            rng.shuffle(o)
            chosen, sums = _greedy(o, W)
            if len(chosen) >= best_size:
                improved, _ = _local_improve(chosen, W, deadline)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best = improved

    if best is None or not best:
        best = list(range(13, 25))  # fallback: the seed

    return sorted(best)


if __name__ == "__main__":
    print(construct())
