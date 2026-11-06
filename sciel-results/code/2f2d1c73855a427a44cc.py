"""Stronger solver: DP over achievable-sum bitsets for fast feasibility,
greedy + branch-and-bound + local search.
For each x, only y > x can subsume x (since 1/y < 1/x).
"""
import time
import math

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit

    def lcm(a, b):
        return a * b // math.gcd(a, b)

    L = 1
    for i in range(1, 25):
        L = lcm(L, i)

    val = {i: L // i for i in range(1, 25)}
    total_val = sum(val.values())
    CAP = (1 << (total_val + 1)) - 1

    # ---------- Fast feasibility ----------
    def is_subsum_free(S):
        S_list = sorted(S)
        for i, x in enumerate(S_list):
            target = val[x]
            cands = [val[y] for y in S_list[i + 1:] if y > x]
            if not cands:
                continue
            achievable = 1
            for c in cands:
                if c > target:
                    continue
                achievable |= (achievable << c)
                if (achievable >> target) & 1:
                    return False
        return True

    # ---------- Greedy ----------
    def greedy_add(order):
        S = set()
        for x in order:
            if time.time() >= deadline:
                break
            S2 = S | {x}
            if is_subsum_free(S2):
                S = S2
        return S

    best_set = greedy_add(range(24, 0, -1))
    best_size = len(best_set)

    # ---------- Branch and bound (bitset-driven) ----------
    # Process x from 24 down to 1. bitset = achievable sums from selected elements > x.
    # include x iff bit val[x] not set; new bitset = bitset | (bitset << val[x]).
    # Prune: size + x <= best_size.

    def bnb(x, bitset, S, size):
        nonlocal best_set, best_size
        if time.time() >= deadline:
            return
        if size + x <= best_size:
            return
        if x < 1:
            return
        target = val[x]
        # include x if not subsumed
        if not ((bitset >> target) & 1):
            new_bits = (bitset | (bitset << target)) & CAP
            S2 = S | {x}
            if size + 1 > best_size:
                best_set = S2
                best_size = size + 1
            bnb(x - 1, new_bits, S2, size + 1)
        # exclude x
        bnb(x - 1, bitset, S, size)

    # Run B&B with a time budget; it will stop early via deadline guard.
    bnb(24, 1, set(), 0)

    # ---------- Local search ----------
    if time.time() < deadline:
        all_elements = set(range(1, 25))

        # add
        for x in sorted(all_elements - best_set, reverse=True):
            if time.time() >= deadline:
                break
            cand = best_set | {x}
            if is_subsum_free(cand):
                best_set = cand

        # 1-swap
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for y in sorted(best_set):
                if time.time() >= deadline:
                    break
                temp = best_set - {y}
                for x in sorted(all_elements - temp, reverse=True):
                    if time.time() >= deadline:
                        break
                    cand = temp | {x}
                    if len(cand) > len(best_set) and is_subsum_free(cand):
                        best_set = cand
                        improved = True
                        break
                if improved:
                    break

    # ---------- Final verification & fallback ----------
    if not best_set or not is_subsum_free(best_set):
        best_set = greedy_add(range(24, 0, -1))
        if not best_set or not is_subsum_free(best_set):
            best_set = {24}

    return list(best_set)
