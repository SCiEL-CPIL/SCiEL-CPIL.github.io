"""Combined program: randomized greedy with bitset residue feasibility + local improvement pass.
Key insight: coprime moduli are ALWAYS incompatible (gcd=1 means residues always congruent mod 1),
so we need pairwise non-coprime moduli with careful residue assignment."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

MAX_MOD = 128

def _build_gcd_table(max_mod):
    table = [[0]*(max_mod+1) for _ in range(max_mod+1)]
    for m1 in range(2, max_mod+1):
        for m2 in range(2, max_mod+1):
            table[m1][m2] = gcd(m1, m2)
    return table

def _count_distinct_prime_factors(n):
    count = 0
    d = 2
    while d*d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def _find_max_classes(max_mod=MAX_MOD, deadline=None):
    if deadline is None:
        deadline = time.time() + 80.0

    all_moduli = list(range(2, max_mod+1))
    gcd_table = _build_gcd_table(max_mod)
    dpf = {m: _count_distinct_prime_factors(m) for m in all_moduli}

    def find_valid_residue(selected, m):
        """Find residue r in [0,m-1] with (r - r_old) % gcd(m, m_old) != 0 for all selected."""
        if not selected:
            return 0
        forbidden = bytearray(m)
        for r_old, m_old in selected:
            g = gcd_table[m][m_old]
            if g == 1:
                return -1  # impossible: coprime moduli always incompatible
            target = r_old % g
            for r in range(target, m, g):
                forbidden[r] = 1
        for r in range(m):
            if not forbidden[r]:
                return r
        return -1

    best_set = []
    best_size = 0

    def try_ordering(ordering):
        nonlocal best_set, best_size
        current = []
        for m in ordering:
            r = find_valid_residue(current, m)
            if r != -1:
                current.append((r, m))
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)

    # Structured orderings: prioritize moduli with many prime factors (more non-coprime partners)
    structured = [
        sorted(all_moduli, key=lambda m: -dpf[m]),
        sorted(all_moduli, key=lambda m: dpf[m]),
        sorted(all_moduli),
        sorted(all_moduli, reverse=True),
    ]

    for ordering in structured:
        if time.time() > deadline:
            break
        try_ordering(ordering)

    # Randomized restarts
    for it in range(50000):
        if time.time() > deadline:
            break
        moduli = all_moduli[:]
        random.shuffle(moduli)
        try_ordering(moduli)

    # Local improvement pass
    if time.time() < deadline - 2:
        chosen_mods = set(m for _, m in best_set)
        remaining = [m for m in all_moduli if m not in chosen_mods]
        random.shuffle(remaining)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for m in remaining[:]:
                r = find_valid_residue(best_set, m)
                if r != -1:
                    best_set.append((r, m))
                    best_size += 1
                    remaining.remove(m)
                    improved = True

    if not best_set:
        return [[1, 2]]
    return [[r, m] for r, m in best_set]

def construct():
    result = _find_max_classes(max_mod=MAX_MOD)
    return result
# EVOLVE-BLOCK-END
