"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 80.0  # 80s safety margin before 90s kill
    N = 128
    nums = list(range(1, N + 1))

    # Precompute gcd matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute bad triples: triples where all three pairwise gcds are equal
    # A triple (a,b,c) is bad if gcd(a,b)==gcd(a,c)==gcd(b,c)
    # Build conflict graph: edge between a,b if they form part of a bad triple
    # Actually, the constraint is on triples, not pairs.
    # A set S is valid if for all distinct a,b,c in S, gcd(a,b)!=gcd(a,c) or gcd(a,b)!=gcd(b,c) or gcd(a,c)!=gcd(b,c)
    # i.e., not all three pairwise gcds are equal.

    # Precompute for each pair (a,b), the gcd value g = gcd(a,b)
    # A triple (a,b,c) is bad iff gcd(a,b)==gcd(a,c)==gcd(b,c)
    # For a fixed pair (a,b) with gcd g, the "forbidden" c values are those
    # where gcd(a,c)==g and gcd(b,c)==g.

    # Build for each number x, the set of pairs (y,z) in the current set that
    # would form a bad triple with x. But this is dynamic.

    # Better approach: build a 3-uniform hypergraph of bad triples,
    # then find maximum independent set (no hyperedge fully contained).

    # Precompute all bad triples
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                    bad_triples.add((a, b, c))

    # Now we want the largest subset of [1,128] containing no bad triple.
    # This is the maximum independent set in a 3-uniform hypergraph.

    # Strategy: Use a greedy + local search approach.
    # 1. Greedy: add numbers in some order, checking validity.
    # 2. Local search: try to add/remove/swap to improve.

    def is_valid_set(S):
        """Check if set S has no bad triple."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = S_list[i], S_list[j]
                g_ab = gcd_mat[a][b]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                        return False
        return True

    def count_bad_triples(S):
        """Count number of bad triples in S."""
        S_list = list(S)
        n = len(S_list)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = S_list[i], S_list[j]
                g_ab = gcd_mat[a][b]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                        count += 1
        return count

    def can_add(S, x):
        """Check if adding x to S creates a bad triple."""
        for a, b in combinations(S, 2):
            g_ab = gcd_mat[a][b]
            if gcd_mat[a][x] == g_ab and gcd_mat[b][x] == g_ab:
                return False
        return True

    # Greedy construction with different orderings
    best_set = set()

    # Try multiple greedy orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: -x.bit_count()),  # more prime factors first
        sorted(range(1, N + 1), key=lambda x: x.bit_count()),
    ]

    for ordering in orderings:
        S = set()
        for x in ordering:
            if can_add(S, x):
                S.add(x)
        if len(S) > len(best_set):
            best_set = S.copy()

    # Local search: try to swap elements to improve
    def local_search(S, max_iters=5000):
        S = S.copy()
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5:
                break
            S_list = list(S)
            for i in range(len(S_list)):
                a = S_list[i]
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    # Try swapping a with x
                    S_new = S - {a} | {x}
                    if count_bad_triples(S_new) < count_bad_triples(S):
                        S = S_new
                        improved = True
                        break
                if improved:
                    break
        return S

    # Also try: start with best_set, try to add elements
    def try_add_elements(S):
        S = S.copy()
        for x in range(1, N + 1):
            if x not in S:
                if can_add(S, x):
                    S.add(x)
        return S

    best_set = try_add_elements(best_set)

    # Local search with restarts
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        # Random perturbation of best_set
        S = best_set.copy()
        # Remove 1-3 random elements
        if S:
            n_remove = random.randint(1, min(3, len(S)))
            to_remove = random.sample(list(S), n_remove)
            S -= set(to_remove)
            # Add 1-3 random elements
            n_add = random.randint(1, min(3, N - len(S)))
            candidates = [x for x in range(1, N + 1) if x not in S]
            if candidates:
                to_add = random.sample(candidates, min(n_add, len(candidates)))
                S |= set(to_add)
        # Local search
        S = local_search(S, max_iters=2000)
        S = try_add_elements(S)
        if len(S) > len(best_set):
            best_set = S.copy()

    # Final polish: try all single swaps
    S_list = list(best_set)
    for i in range(len(S_list)):
        a = S_list[i]
        for x in range(1, N + 1):
            if x in best_set:
                continue
            S_new = best_set - {a} | {x}
            if count_bad_triples(S_new) == 0 and len(S_new) > len(best_set):
                best_set = S_new
                break

    # Ensure nonempty
    if not best_set:
        best_set = {1}

    return list(best_set)
# EVOLVE-BLOCK-END
