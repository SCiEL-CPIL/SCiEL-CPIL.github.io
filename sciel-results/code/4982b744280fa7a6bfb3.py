import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill

    N = 192
    K = 3
    TARGET = 48

    # ---- Precomputation: for each x in 1..N, find all triples (a,b,c) with a<b<c,
    #      a+b+c divisible by x.  Also, for each x, find all pairs (b,c) such that
    #      for some a in S, a | (x + b + c).  We handle the second type dynamically
    #      but precompute triple-sum divisibility. ----

    # Precompute all triples and their sums
    # Triple count: C(192,3) = 1,152,640 — manageable
    # For speed, use numpy to generate all triples
    idx = np.arange(1, N + 1)
    # Generate all C(N,3) triples via meshgrid on indices
    # This is memory-heavy; instead, do it in chunks or use a smarter approach.
    # Actually 1,152,640 triples * 3 ints = ~8.7M ints. Fine in numpy.
    # But generating via meshgrid of 3 192-arrays = 192^3 = 7M entries, filter a<b<c.
    # 7M * 3 * 8 bytes = ~168 MB. Acceptable.

    # Alternative: loop over a, then use numpy for b,c pairs.
    # For each a, b ranges a+1..N, c ranges b+1..N.
    # Total iterations sum over a of C(N-a, 2) = C(N,3) = 1.15M.
    # In pure Python this is ~1.15M iterations, each doing a numpy op on small arrays.
    # Let's just do it in a tight Python loop with precomputed pair sums.

    # Precompute pair sums: for each pair (i,j) with i<j, store sum
    # Then for each a, triple sum = a + pair_sum[b,c] for b,c > a
    # Actually, let's precompute: for each x in 1..N, the list of triples (a,b,c)
    # such that x | (a+b+c).

    # Efficient approach:
    # For each x, we need all (a,b,c) distinct, a<b<c, x | (a+b+c).
    # Equivalently: a+b+c ≡ 0 (mod x).
    # We can iterate over residues. But simplest: precompute all triple sums
    # and for each x, mark which triples have sum % x == 0.

    # Let's build: triple_sums array of shape (num_triples, 3) and sums array.
    # Then for each x, valid_triples[x] = indices where sums % x == 0.

    # Generate all triples
    triples = []
    sums = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                triples.append((a, b, c))
                sums.append(a + b + c)

    triples = np.array(triples, dtype=np.int32)  # (1152640, 3)
    sums = np.array(sums, dtype=np.int32)         # (1152640,)

    # For each x in 1..N, precompute the set of triple-indices where x | sum
    # Store as a list of numpy arrays (indices)
    triple_div = {}  # x -> np.array of triple indices where x divides sum
    for x in range(1, N + 1):
        mask = (sums % x) == 0
        triple_div[x] = np.where(mask)[0]

    # Also, for each x, we need: given a set S, check if any a in S divides x + b + c
    # where b, c in S, b != c, b != a, c != a.
    # This is harder to precompute fully. We'll handle it with a different strategy:
    # For the greedy, when considering adding x to S:
    #   Check 1: x divides sum of any 3 from S → use precomputed triple_div[x]
    #            intersected with triples fully in S.
    #   Check 2: any a in S divides x + b + c for b,c in S\{a}
    #            → for each a in S, check if there exist b,c in S\{a} with a | (x+b+c)
    #            → equivalently, b+c ≡ -x (mod a)
    #            → precompute for each a and each residue r mod a, the set of pairs (b,c)
    #              in [1..N] with b<c, b,c != a, and (b+c) % a == r.
    #            Then check if any such pair is fully in S.

    # Precompute pair residue classes
    # pair_div[a][r] = set of (b,c) pairs with b<c, b,c != a, (b+c)%a == r
    # This is large: for a=192, 191*190/2 ≈ 18k pairs, 192 residues.
    # Total over all a: sum_a C(N-1,2) ≈ 192 * 18k ≈ 3.5M pairs total.
    # Storing as sets of tuples is heavy. Instead, store as bitmasks or sorted arrays.
    # For efficiency, store for each (a, r) a frozenset of (b,c) pairs.
    # Actually, for the check we need: given S (a set), is there a pair (b,c) in pair_div[a][r]
    # such that both b and c are in S?
    # We can precompute for each (a, r) the list of pairs, and during check iterate.
    # But that's O(|pairs|) per (a,r) check, and we have |S| * 1 such checks per insertion.
    # |S| up to 48, pairs per (a,r) up to ~100. So ~4800 ops per check. Acceptable.

    # Let's precompute: for each a in 1..N, for each r in 0..a-1:
    #   pair_res[a][r] = list of (b, c) with 1<=b<c<=N, b!=a, c!=a, (b+c)%a==r
    pair_res = {}
    for a in range(1, N + 1):
        pair_res[a] = {}
        for r in range(a):
            pair_res[a][r] = []
        for b in range(1, N + 1):
            if b == a:
                continue
            for c in range(b + 1, N + 1):
                if c == a:
                    continue
                r = (b + c) % a
                pair_res[a][r].append((b, c))

    # Now, for the greedy check:
    def _can_add(x, S_set, S_list):
        """Check if x can be added to S (S_set is a set, S_list is a list)."""
        if len(S_list) < K:
            return True

        # Check 1: x divides sum of any 3 distinct elements from S
        # Use precomputed triple_div[x]: indices of triples where x | sum
        # We need to check if any such triple is fully contained in S.
        # Instead of checking all triple_div[x] indices (could be many),
        # iterate over triples of S and check divisibility.
        # For |S| <= 48, C(48,3) = 17296. That's fast.
        for a, b, c in combinations(S_list, 3):
            if (a + b + c) % x == 0:
                return False

        # Check 2: any a in S divides x + b + c for b, c in S \ {a}
        for a in S_list:
            r = (-x) % a  # need b + c ≡ -x (mod a)
            pairs = pair_res[a][r]
            for b, c in pairs:
                if b in S_set and c in S_set and b != a and c != a:
                    return False
        return True

    # ---- Greedy with randomized restarts ----
    best_set = []
    best_size = 0

    # Initial deterministic greedy (descending)
    S = []
    S_set = set()
    for x in range(N, 0, -1):
        if _can_add(x, S_set, S):
            S.append(x)
            S_set.add(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
    if best_size >= TARGET:
        return best_set

    # Randomized greedy restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        order = list(range(1, N + 1))
        random.shuffle(order)

        S = []
        S_set = set()
        for x in order:
            if _can_add(x, S_set, S):
                S.append(x)
                S_set.add(x)

        # Local search: try to add more elements
        remaining = [x for x in range(1, N + 1) if x not in S_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in remaining:
                if _can_add(x, S_set, S):
                    S.append(x)
                    S_set.add(x)
                    remaining.remove(x)
                    improved = True
                    break

        # Swap-based local search: remove one, try to add two
        if time.time() < deadline - 2:
            for _ in range(200):  # limited swaps
                if time.time() > deadline:
                    break
                if not S:
                    break
                # Pick a random element to remove
                idx = random.randrange(len(S))
                removed = S[idx]
                S.pop(idx)
                S_set.discard(removed)

                # Try to add elements from remaining + {removed}
                candidates = [x for x in range(1, N + 1) if x not in S_set]
                random.shuffle(candidates)
                added = 0
                for x in candidates:
                    if added >= 2:
                        break
                    if _can_add(x, S_set, S):
                        S.append(x)
                        S_set.add(x)
                        added += 1

                # If we added fewer than we removed, revert
                if added < 1:
                    S.append(removed)
                    S_set.add(removed)

        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            if best_size >= TARGET:
                break

    return best_set
