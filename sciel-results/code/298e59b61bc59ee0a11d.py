"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 128
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)

    # Precompute all "bad triples" (three distinct elements where all three pairwise gcds are equal)
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    # We'll build a hypergraph: each edge is a bad triple.

    # For efficiency, we'll track conflicts differently.
    # A set S is valid if no triple in S is a bad triple.
    # Equivalently, for any two elements a, b in S, no c in S can form a bad triple with them.
    # For a fixed pair (a, b), the "forbidden" c values are those where gcd(a,c) == gcd(b,c) == gcd(a,b).

    # Let's precompute for each pair (i, j), the set of k that would form a bad triple.
    # Actually, for local search and branch-and-bound, we need:
    # - Given a candidate set S and a new element x, check if adding x creates any bad triple.
    # - A bad triple involving x and two elements a, b in S exists iff gcd(x,a) == gcd(x,b) == gcd(a,b).

    # Precompute for each x and each a, the gcd(x,a).
    # For a given x, and a set S, x conflicts with S if there exist a, b in S such that
    # gcd(x,a) == gcd(x,b) == gcd(a,b).
    # This means: for the value g = gcd(x,a), we need another b in S where gcd(x,b) = g and gcd(a,b) = g.

    # Let's precompute: for each x and each possible gcd value g, which numbers a have gcd(x,a) = g?
    # Then for a given x and g, the forbidden a's are those in set G[x][g].
    # And among those a's, we need gcd(a,b) = g for some b also in G[x][g].

    # Alternative approach for checking:
    # For candidate x and current set S:
    #   For each a in S:
    #     g = gcd(x, a)
    #     For each b in S, b != a:
    #       if gcd(x,b) == g and gcd(a,b) == g:
    #         return False (bad triple found)
    #   return True
    # This is O(|S|^2) per insertion, which is fine for |S| up to ~60-70.

    def is_valid_with(x, S):
        """Check if x can be added to S without creating a bad triple."""
        for a in S:
            g = gcd_table[x][a]
            for b in S:
                if b == a:
                    continue
                if gcd_table[x][b] == g and gcd_table[a][b] == g:
                    return False
        return True

    # Greedy initialization: try multiple orderings
    def greedy(order):
        S = []
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_with(x, S):
                S.append(x)
        return S

    # Local search: try to swap elements to improve
    def local_search(S):
        S = list(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing one element and adding a better one
            for i, rem in enumerate(S):
                if time.time() > deadline:
                    break
                S_no_rem = S[:i] + S[i+1:]
                # Try all candidates not in S_no_rem
                best_add = None
                best_score = len(S_no_rem)  # we want to increase size
                for x in range(1, N + 1):
                    if x in S_no_rem:
                        continue
                    if is_valid_with(x, S_no_rem):
                        # This would give size len(S_no_rem) + 1 = len(S)
                        # We need strictly better, so only consider if we can do better
                        # Actually, swapping keeps size same. We want to find a way to increase size.
                        pass
                # Better: try to find a set S' of size > len(S)
                # Let's try: remove rem, then greedily add elements
                S_candidate = list(S_no_rem)
                for x in range(1, N + 1):
                    if x in S_candidate:
                        continue
                    if is_valid_with(x, S_candidate):
                        S_candidate.append(x)
                if len(S_candidate) > len(S):
                    S = S_candidate
                    improved = True
                    break
            # Also try simple additions
            for x in range(1, N + 1):
                if x in S:
                    continue
                if is_valid_with(x, S):
                    S.append(x)
                    improved = True
                    break
        return S

    # Branch and bound for maximum independent set in hypergraph
    # Represent the problem as: maximize |S| subject to S contains no bad triple.
    # Use recursive backtracking with pruning.

    # For B&B, we need to track the current set and use bounds.
    # A simple bound: the maximum possible size is len(remaining candidates) + len(current).
    # But we need a tighter bound.

    # Let's use a simpler approach: iterative improvement with multiple restarts.

    best = []
    best_size = 0

    # Try different greedy orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        [x for x in range(1, N + 1) if x % 2 == 0] + [x for x in range(1, N + 1) if x % 2 == 1],
        [x for x in range(1, N + 1) if x % 2 == 1] + [x for x in range(1, N + 1) if x % 2 == 0],
        list(range(N, 0, -2)) + list(range(N - 1, 0, -2)),  # evens descending + odds descending
    ]

    import random
    for _ in range(20):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orderings.append(random_order)

    for order in orderings:
        if time.time() > deadline:
            break
        S = greedy(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best = S

    # More local search iterations
    while time.time() < deadline - 5.0:
        S = list(best)
        S_new = local_search(S)
        if len(S_new) > best_size:
            best_size = len(S_new)
            best = S_new
        else:
            # Try a random perturbation
            if len(best) >= 2:
                S_perturbed = list(best)
                for _ in range(3):
                    if S_perturbed:
                        S_perturbed.pop(random.randint(0, len(S_perturbed) - 1))
                S_new = local_search(S_perturbed)
                if len(S_new) > best_size:
                    best_size = len(S_new)
                    best = S_new

    return best
# EVOLVE-BLOCK-END
