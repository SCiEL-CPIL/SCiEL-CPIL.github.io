import time
import math
import random

def _is_disjoint(r1, m1, r2, m2, gcd_table=None):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    if gcd_table is not None:
        g = gcd_table[m1][m2]
    else:
        g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    MAX_MOD = 128

    moduli_list = list(range(2, MAX_MOD + 1))

    # Precompute gcd table for moduli 0..128
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    factor_counts = {m: num_prime_factors(m) for m in moduli_list}
    div_counts = {m: num_divisors(m) for m in moduli_list}

    def try_add(chosen, m, gcd_table, residue_strategy="smallest_gcd", seed=0):
        """Try to find a valid residue for modulus m given current chosen set."""
        if not chosen:
            return 0
        # Build candidate residues ordered by "least conflict" with chosen
        # For each residue r in 0..m-1, compute max gcd(r - r2, m, m2) over chosen
        # Actually we want residues where (r - r2) % gcd(m, m2) != 0 for all chosen.
        # Heuristic: prefer residues where gcd(r - r2, m*m2) is small (less overlap)
        # Simpler: just try residues in an order that varies with seed
        if residue_strategy == "smallest_gcd":
            # Score each residue by sum of gcds with chosen (lower = better)
            scored = []
            for r in range(m):
                score = 0
                for (r2, m2) in chosen:
                    g = gcd_table[m][m2]
                    diff = abs(r - r2)
                    # We need (r - r2) % g != 0. If it is 0, it's invalid.
                    # For scoring, we want to avoid residues close to r2 mod g
                    # Use the actual gcd of (r-r2) and m*m2 as a proxy for "closeness"
                    # But simpler: count how many constraints are "tight"
                    if diff % g == 0:
                        score += 1000  # invalid, put at end
                    else:
                        # Distance to boundary: how far is (r-r2) mod g from 0?
                        rem = diff % g
                        score += min(rem, g - rem)  # smaller = more constrained
                scored.append((score, r))
            scored.sort()
            candidates = [r for _, r in scored]
        elif residue_strategy == "random":
            candidates = list(range(m))
            random.shuffle(candidates)
        else:  # sequential with offset
            start_r = (seed + m) % m
            candidates = [(start_r + i) % m for i in range(m)]

        for r in candidates:
            ok = True
            for (r2, m2) in chosen:
                g = gcd_table[m][m2]
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                return r
        return None

    def greedy_construct(order, gcd_table, residue_strategy="smallest_gcd", seed=0):
        """Greedy: add moduli in given order, picking best valid residue."""
        chosen = []
        for m in order:
            r = try_add(chosen, m, gcd_table, residue_strategy, seed)
            if r is not None:
                chosen.append((r, m))
        return chosen

    def backfill(chosen, gcd_table, residue_strategy="smallest_gcd", seed=0):
        """Try to add back any moduli that were skipped."""
        chosen = list(chosen)
        used_moduli = set(m for _, m in chosen)
        unused = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
        # Sort unused by some heuristic
        unused.sort(key=lambda m: -div_counts[m])
        improved = True
        while improved:
            improved = False
            for m in unused[:]:
                if m in used_moduli:
                    continue
                r = try_add(chosen, m, gcd_table, residue_strategy, seed)
                if r is not None:
                    chosen.append((r, m))
                    used_moduli.add(m)
                    unused.remove(m)
                    improved = True
        return chosen

    def local_search(chosen, gcd_table, deadline, residue_strategy="smallest_gcd"):
        """Try to improve by removing one and adding two, or swapping."""
        chosen = list(chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(chosen)
            # Try removing one element and adding two new ones
            for i in range(n):
                if time.time() > deadline:
                    break
                remaining = [c for j, c in enumerate(chosen) if j != i]
                removed = chosen[i]
                used_moduli = set(m for _, m in remaining)
                unused = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
                # Sort unused by heuristic
                unused.sort(key=lambda m: -div_counts[m])
                new_chosen = list(remaining)
                added = 0
                for m in unused:
                    r = try_add(new_chosen, m, gcd_table, residue_strategy)
                    if r is not None:
                        new_chosen.append((r, m))
                        added += 1
                    if added >= 2:
                        break
                if added >= 2:
                    chosen = new_chosen
                    improved = True
                    break
        return chosen

    best_count = 0
    best_result = None

    # Precompute orderings
    orderings = [
        sorted(moduli_list, reverse=True),
        sorted(moduli_list),
        sorted(moduli_list, key=lambda x: factor_counts[x], reverse=True),
        sorted(moduli_list, key=lambda x: factor_counts[x]),
        sorted(moduli_list, key=lambda x: -div_counts[x]),
        sorted(moduli_list, key=lambda x: div_counts[x]),
    ]

    num_restarts = 3000

    for it in range(num_restarts):
        if time.time() > deadline:
            break

        # Choose an ordering
        if it % 6 == 0:
            order = orderings[0][:]  # Descending
        elif it % 6 == 1:
            order = orderings[1][:]  # Ascending
        elif it % 6 == 2:
            order = orderings[2][:]  # High factors first
        elif it % 6 == 3:
            order = orderings[3][:]  # Low factors first
        elif it % 6 == 4:
            order = orderings[4][:]  # High divisors first
        else:
            order = moduli_list[:]
            random.shuffle(order)

        # Vary residue strategy
        if it % 3 == 0:
            strat = "smallest_gcd"
        elif it % 3 == 1:
            strat = "random"
        else:
            strat = "sequential"
        seed = it

        chosen = greedy_construct(order, gcd_table, strat, seed)
        chosen = backfill(chosen, gcd_table, strat, seed)
        chosen = local_search(chosen, gcd_table, deadline, strat)

        count = len(chosen)
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in chosen]

    # Fallback
    fallback = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    if best_result is None or len(best_result) < len(fallback):
        best_result = fallback

    return best_result
