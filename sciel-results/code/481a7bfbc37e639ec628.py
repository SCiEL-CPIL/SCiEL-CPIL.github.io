"""
Structural rewrite: use a deterministic algebraic construction over GF(31) for n=31,
then use a fast local-search to extend to n=34 if possible, with a deadline guard.

The key insight is that the seed's random local-search is too slow and unreliable.
Instead:
1. Use a structured coloring over GF(31) (a prime field) to get a valid n=31 coloring
   quickly (deterministic, O(n^3) construction).
2. Attempt to extend to n=32, 33, 34 by adding vertices and using targeted local search
   only on the newly added triples, which is much faster.
3. If extension fails, fall back to the best valid n found.

The GF(31) construction: color triple (a,b,c) red (0) if (a-b)*(b-c)*(c-a) is a quadratic
residue mod 31, else blue (1). This is a known type of construction for hypergraph Ramsey
problems that avoids monochromatic cliques of certain sizes.
"""
import time
import random
from itertools import combinations

_DEADLINE = time.time() + 130.0  # 20s safety margin under 150s kill


def _is_quadratic_residue(x, p):
    """Check if x is a nonzero quadratic residue mod p."""
    if x % p == 0:
        return True
    return pow(x, (p - 1) // 2, p) == 1


def _gf31_coloring(n=31):
    """
    Deterministic coloring of triples of {0, ..., n-1} using GF(31) structure.
    Color triple (a,b,c) red (0) if the product (a-b)*(b-c)*(c-a) mod 31 is a QR, else blue (1).
    """
    p = 31
    colors = []
    triples = list(combinations(range(n), 3))
    for a, b, c in triples:
        # Compute product of differences mod p
        d1 = (a - b) % p
        d2 = (b - c) % p
        d3 = (c - a) % p
        prod = (d1 * d2 * d3) % p
        if prod == 0:
            # Degenerate case (shouldn't happen for distinct a,b,c mod 31 when n<=31)
            colors.append(0)
        else:
            colors.append(0 if _is_quadratic_residue(prod, p) else 1)
    return triples, colors


def _count_violations_fast(colors, n):
    """
    Count red K4 violations and blue K5 violations.
    colors is a list of 0/1 for each triple in lex order.
    """
    # Build triple index
    tmap = {}
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                tmap[(a, b, c)] = idx
                idx += 1

    red4_count = 0
    for quad in combinations(range(n), 4):
        all_red = True
        for i in range(4):
            t = tuple(quad[j] for j in range(4) if j != i)
            if colors[tmap[t]] != 0:
                all_red = False
                break
        if all_red:
            red4_count += 1

    blue5_count = 0
    for quint in combinations(range(n), 5):
        all_blue = True
        for combo in combinations(range(5), 3):
            t = tuple(quint[j] for j in combo)
            if colors[tmap[t]] != 1:
                all_blue = False
                break
        if all_blue:
            blue5_count += 1

    return red4_count + blue5_count


def _try_extend(base_colors, base_n, target_n, time_budget, rng):
    """
    Try to extend a valid coloring of base_n vertices to target_n vertices.
    Only search over the new triples involving at least one vertex >= base_n.
    """
    if target_n <= base_n:
        return base_colors[:], target_n

    # Build full triple list for target_n
    triples = list(combinations(range(target_n), 3))
    tmap = {t: i for i, t in enumerate(triples)}
    m = len(triples)

    # Initialize colors: copy base, random for new
    colors = [0] * m
    for i in range(base_n * base_n * base_n):  # placeholder
        pass

    # Copy base colors
    base_triples = list(combinations(range(base_n), 3))
    for i, t in enumerate(base_triples):
        colors[tmap[t]] = base_colors[i]

    # New triples: those with at least one vertex >= base_n
    new_triple_indices = []
    for i, t in enumerate(triples):
        if any(v >= base_n for v in t):
            new_triple_indices.append(i)
            colors[i] = rng.randint(0, 1)

    # Build membership for violation counting
    # For each 4-set, the 4 triples it contains
    four_sets = list(combinations(range(target_n), 4))
    four_triples = []
    for fs in four_sets:
        ts = []
        for i in range(4):
            t = tuple(fs[j] for j in range(4) if j != i)
            ts.append(tmap[t])
        four_triples.append(ts)

    five_sets = list(combinations(range(target_n), 5))
    five_triples = []
    for fs in five_sets:
        ts = []
        for combo in combinations(range(5), 3):
            t = tuple(fs[j] for j in combo)
            ts.append(tmap[t])
        five_triples.append(ts)

    # Initialize violation counters
    red4 = [0] * len(four_sets)
    blue5 = [0] * len(five_sets)
    for fi, ts in enumerate(four_triples):
        c = 0
        for t in ts:
            if colors[t] == 0:
                c += 1
        red4[fi] = c
    for fi, ts in enumerate(five_triples):
        c = 0
        for t in ts:
            if colors[t] == 1:
                c += 1
        blue5[fi] = c

    violations = sum(1 for x in red4 if x == 4) + sum(1 for x in blue5 if x == 10)

    if violations == 0:
        return colors, target_n

    # Build reverse mapping: triple -> which 4-sets and 5-sets it's in
    triple_to_fours = [[] for _ in range(m)]
    for fi, ts in enumerate(four_triples):
        for t in ts:
            triple_to_fours[t].append(fi)
    triple_to_fives = [[] for _ in range(m)]
    for fi, ts in enumerate(five_triples):
        for t in ts:
            triple_to_fives[t].append(fi)

    # Local search: only flip new triples
    start = time.time()
    best_v = violations
    best_colors = colors[:]
    no_improve = 0

    while time.time() - start < time_budget:
        if not new_triple_indices:
            break
        t = new_triple_indices[rng.randrange(len(new_triple_indices))]
        old = colors[t]
        new = 1 - old

        delta = 0
        for fi in triple_to_fours[t]:
            old_cnt = red4[fi]
            if old == 0:
                new_cnt = old_cnt - 1
                if old_cnt == 4:
                    delta -= 1
                if new_cnt == 4:
                    delta += 1
            else:
                new_cnt = old_cnt + 1
                if old_cnt == 4:
                    delta -= 1
                if new_cnt == 4:
                    delta += 1
            red4[fi] = new_cnt
        for fi in triple_to_fives[t]:
            old_cnt = blue5[fi]
            if old == 1:
                new_cnt = old_cnt - 1
                if old_cnt == 10:
                    delta -= 1
                if new_cnt == 10:
                    delta += 1
            else:
                new_cnt = old_cnt + 1
                if old_cnt == 10:
                    delta -= 1
                if new_cnt == 10:
                    delta += 1
            blue5[fi] = new_cnt

        colors[t] = new
        violations += delta

        if violations < best_v:
            best_v = violations
            best_colors = colors[:]
            no_improve = 0
            if best_v == 0:
                return best_colors, target_n
        else:
            no_improve += 1
            if no_improve > 2000:
                # Revert
                colors[t] = old
                for fi in triple_to_fours[t]:
                    red4[fi] = red4[fi] + (1 if old == 0 else -1)
                for fi in triple_to_fives[t]:
                    blue5[fi] = blue5[fi] + (1 if old == 1 else -1)
                violations -= delta
                no_improve = 0

    return None, target_n


def construct():
    global _DEADLINE
    start = time.time()
    rng = random.Random(12345)

    # Step 1: Get a valid n=31 coloring via GF(31) construction
    triples31, colors31 = _gf31_coloring(31)

    # Verify validity
    v31 = _count_violations_fast(colors31, 31)
    if v31 != 0:
        # Fallback: try a simpler construction
        # Use parity-based coloring
        n = 12
        from itertools import combinations as _comb
        colors = [0 if (a + b + c) % 2 == 0 else 1 for a, b, c in _comb(range(n), 3)]
        return {'n': n, 'colors': colors}

    best_n = 31
    best_colors = colors31

    # Step 2: Try to extend to 32, 33, 34
    for target_n in [32, 33, 34]:
        remaining = _DEADLINE - time.time()
        if remaining < 10:
            break
        budget = min(remaining * 0.5, 30.0)
        result, _ = _try_extend(best_colors, best_n, target_n, budget, rng)
        if result is not None:
            # Verify
            v = _count_violations_fast(result, target_n)
            if v == 0:
                best_n = target_n
                best_colors = result
            else:
                break
        else:
            break

    # Final verification
    final_v = _count_violations_fast(best_colors, best_n)
    if final_v != 0:
        # Shouldn't happen, but fallback
        n = 12
        from itertools import combinations as _comb
        best_colors = [0 if (a + b + c) % 2 == 0 else 1 for a, b, c in _comb(range(n), 3)]
        best_n = n

    return {'n': best_n, 'colors': best_colors}
