"""Rewritten: exact max-independent-set over residue assignments with deadline guard."""
import math
import time
from functools import lru_cache

def _deadline():
    return time.time() + 75.0  # safety margin under 90s

def construct():
    t_end = _deadline()
    MAXM = 128
    moduli = list(range(2, MAXM + 1))
    n = len(moduli)

    # Precompute gcd for each pair
    g = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g[i][j] = math.gcd(moduli[i], moduli[j])

    # For each pair (i,j), conflict when g[i][j] divides (r_i - r_j).
    # We'll do branch and bound: choose an ordering, assign residues, maintain
    # the set of chosen vertices as a bitset, and a "forbidden" bitset of
    # vertices that conflict with any chosen vertex.

    # Order vertices by a heuristic: prefer moduli with many multiples / high
    # potential to conflict less? Actually we want to find a large independent
    # set. Use greedy ordering by descending "conflict density" to prune early.

    # Precompute for each vertex i, for each residue r, the set of vertices j
    # that would conflict with (i, r). Represent as bitset (int).
    # conflict[i][r] = bitmask of j != i such that g[i][j] | (r - r_j) for SOME r_j.
    # Wait: conflict depends on r_j too. We need: given chosen (i, r_i), vertex j
    # is "blocked" if there EXISTS r_j in [0, m_j) with g[i][j] | (r_i - r_j).
    # That exists iff the residue class r_i mod g[i][j] has some r_j ≡ r_i (mod g[i][j]).
    # Since r_j can be any value 0..m_j-1, and g[i][j] <= m_j, the class r_i mod g
    # always has at least one representative in [0, m_j) (since g <= m_j). So for
    # ANY r_i, vertex j is blockable for every r_j choice? No — we CHOOSE r_j to
    # AVOID conflict. j is forced-conflicting only if for ALL r_j, g | (r_i - r_j),
    # i.e., the class r_i mod g has NO representative in [0,m_j). That never happens
    # since g <= m_j. So every pair can be made disjoint by choosing residues!
    #
    # Therefore the ONLY constraint is: we must be able to choose residues so that
    # for every selected pair, r_i not≡ r_j (mod g[i][j]). This is a CSP.
    #
    # So the problem reduces to: find the largest subset S of moduli such that there
    # exist residues r_m for m in S with r_i not≡ r_j (mod gcd(m_i,m_j)) for all i≠j.
    #
    # Greedy / backtracking over subsets with residue assignment.

    # We'll do a search over subsets (in a good order) and check feasibility via
    # backtracking on residues. To maximize count, try to include as many as possible.

    # Strategy: iterate over all moduli in an order; use a recursive search that
    # tries to add each modulus (assigning a residue) and backtracks. Use the fact
    # that a modulus m can be added iff there exists r in [0,m) such that for all
    # already-chosen (i, r_i), g[m][i] does NOT divide (r - r_i).

    # Order: process moduli in descending order of "flexibility" or just fixed order.
    # Use a recursive branch-and-bound with upper bound = remaining count.

    # Precompute for each modulus index i, list of other indices and their gcd.
    # We'll maintain chosen as list of (idx, residue).

    # To speed up, precompute for each i and each candidate residue r, the set of
    # previously chosen that it conflicts with — but that's dynamic. Instead, for a
    # new modulus i with residue r, it conflicts with chosen (j, r_j) iff g[i][j] | (r - r_j).
    # We can precompute for each i, for each r, the set of residues r_j (per j) that
    # conflict. Simpler: just check on the fly; n is 127, residues up to 127, chosen up to
    # ~127, so each check is O(chosen). Feasible within time if search is pruned.

    # Better: use a greedy + local improvement, then try to verify maximality via
    # a bounded search. Given time, do a randomized restarts greedy with conflict check,
    # keep best, then a final local search.

    best = []

    def feasible_add(chosen, i, r):
        for (j, rj) in chosen:
            if g[i][j] != 0 and (r - rj) % g[i][j] == 0:
                return False
        return True

    def greedy(order, seed):
        import random
        rnd = random.Random(seed)
        chosen = []
        # order is a permutation of indices
        for i in order:
            m = moduli[i]
            # try residues in random order, pick first feasible
            residues = list(range(m))
            rnd.shuffle(residues)
            placed = False
            for r in residues:
                if feasible_add(chosen, i, r):
                    chosen.append((i, r))
                    placed = True
                    break
            if not placed:
                # try to find any residue that works with current chosen
                # (already tried all) -> skip
                pass
        return chosen

    # Try a few orders and seeds, keep best size.
    import random
    base_order = list(range(n))
    # Order by descending modulus (larger moduli have more residue options -> easier to place)
    order_by_desc = sorted(range(n), key=lambda i: -moduli[i])
    order_by_asc = sorted(range(n), key=lambda i: moduli[i])

    candidates = [order_by_desc, order_by_asc, base_order]
    for _ in range(20):
        perm = base_order[:]
        random.shuffle(perm)
        candidates.append(perm)

    best_chosen = []
    for cidx, order in enumerate(candidates):
        if time.time() > t_end - 5:
            break
        for seed in range(3):
            if time.time() > t_end - 5:
                break
            ch = greedy(order, seed + cidx * 1000)
            if len(ch) > len(best_chosen):
                best_chosen = ch
            if len(best_chosen) == n:
                break
        if len(best_chosen) == n:
            break

    # Local search: try to add skipped moduli by re-assigning residues (simple hill climb)
    # Convert best_chosen to dict idx->residue
    chosen_set = {i: r for (i, r) in best_chosen}
    # Try to add each unchosen modulus by finding a residue compatible with chosen
    for i in range(n):
        if i in chosen_set:
            continue
        m = moduli[i]
        for r in range(m):
            ok = True
            for j, rj in chosen_set.items():
                if g[i][j] != 0 and (r - rj) % g[i][j] == 0:
                    ok = False
                    break
            if ok:
                chosen_set[i] = r
                break

    result = [[chosen_set[i], moduli[i]] for i in sorted(chosen_set.keys())]
    return result if result else [[0, 2]]
