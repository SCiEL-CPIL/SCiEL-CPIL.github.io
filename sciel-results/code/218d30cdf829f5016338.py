"""Branch-and-bound maximum-cardinality search for equal-gcd triple avoidance."""
import random
import time
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 128
    UNIVERSE = list(range(1, N + 1))
    N1 = N + 1

    # Precompute gcd matrix
    gcd_mat = [[0] * N1 for _ in range(N1)]
    for i in range(1, N1):
        for j in range(i + 1, N1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute conflict: conflict[a][b] = set of c such that {a,b,c} is a bad triple
    # i.e., gcd(a,b)==gcd(a,c)==gcd(b,c)
    # For efficiency, store as frozensets
    conflict = [set() for _ in range(N1)]
    for a in range(1, N1):
        for b in range(a + 1, N1):
            g = gcd_mat[a][b]
            if g == 1:
                continue  # no c can have gcd 1 with both a and b unless c=1, but 1 gives gcd(1,a)=1,gcd(1,b)=1 -> triple {1,a,b} with all gcds 1! Wait.
                # Actually if gcd(a,b)=1, then we need gcd(a,c)=1 and gcd(b,c)=1.
                # c=1: gcd(a,1)=1, gcd(b,1)=1 -> yes, {1,a,b} is bad if gcd(a,b)=1.
                # So we can't skip.
                pass
            for c in range(b + 1, N1):
                if gcd_mat[a][c] == g and gcd_mat[b][c] == g:
                    conflict[a].add(c)
                    conflict[b].add(c)
                    conflict[c].add(a)
                    conflict[c].add(b)

    # Also need: for a pair (a,b), the set of c that conflict with BOTH a and b given a current set S.
    # Actually, the constraint is: no triple in S has all three pairwise gcds equal.
    # When adding x to S, we need to check: for every pair (a,b) in S, gcd(a,b) != gcd(a,x) != gcd(b,x) (all three equal).
    # Equivalently, for each a in S, let ga = gcd(a,x). We need that there is no b in S such that gcd(a,b)==ga and gcd(b,x)==ga.
    # Precompute for each x and each possible gcd value g, the set of a such that gcd(a,x)==g.
    # Then for a given S and x, for each a in S, g=gcd(a,x), we need to check if any b in S has gcd(a,b)==g AND gcd(b,x)==g.
    # This is still O(|S|) per a, so O(|S|^2) total.

    # Alternative: precompute for each pair (a,b), the set of x that would create a bad triple with a,b.
    # bad_with_ab = {x : gcd(a,b)==gcd(a,x)==gcd(b,x)}
    # Then when adding x to S, it's invalid iff there exists (a,b) in S such that x in bad_with_ab.
    # Precompute bad_with_ab for all pairs.

    # For n=128, there are 128*127/2 = 8128 pairs. For each pair, the bad set could be large.
    # Let's precompute: for each pair (a,b), the frozenset of c such that {a,b,c} is bad.
    bad_triple = {}  # key: (a,b) with a<b, value: frozenset of c
    for a in range(1, N1):
        for b in range(a + 1, N1):
            g = gcd_mat[a][b]
            s = set()
            for c in range(1, N1):
                if c == a or c == b:
                    continue
                if gcd_mat[a][c] == g and gcd_mat[b][c] == g:
                    s.add(c)
            bad_triple[(a, b)] = frozenset(s)

    # For branch and bound, we'll use a backtracking approach.
    # State: current set S (as a frozenset for hashing), remaining candidates R.
    # We want to maximize |S|.
    # Upper bound: |S| + |R'| where R' = {x in R : x is compatible with all elements in S}
    # But even better: |S| + max independent set in the conflict graph induced by R, where conflict between two elements x,y in R means they can't both be in S together with the current S.
    # Actually, the constraint is global: no three in S form a bad triple.
    # When we have current S, and we want to add more from R, the new elements must:
    # 1. Be individually compatible with S (i.e., for each x in R, adding x to S doesn't create a bad triple).
    # 2. Among themselves, no three form a bad triple, and no two of them plus one from S form a bad triple.
    # This is complex. Let's use a simpler bound: |S| + |compatible_R| where compatible_R = {x in R : x can be added to S}.
    # This is a valid upper bound (not tight, but simple).

    def can_add_to_S(S_set, x, S_list):
        """Check if x can be added to S without creating a bad triple."""
        for a in S_list:
            gax = gcd_mat[a][x]
            for b in S_list:
                if b <= a:
                    continue
                if gcd_mat[a][b] == gax and gcd_mat[b][x] == gax:
                    return False
        return True

    # For speed, let's precompute for each x, and for each a, the set of b such that {a,b,x} is bad.
    # bad_with_x[a] = set of b such that gcd(a,x)==gcd(a,b)==gcd(b,x)
    # Then can_add_to_S(S, x) is: for each a in S, if bad_with_x[a] intersects S, then invalid.

    bad_with = [None] * N1  # bad_with[x] is a list of sets, indexed by a
    for x in range(1, N1):
        bad_with[x] = [set() for _ in range(N1)]
        for a in range(1, N1):
            if a == x:
                continue
            gax = gcd_mat[a][x]
            for b in range(a + 1, N1):
                if b == x:
                    continue
                if gcd_mat[a][b] == gax and gcd_mat[b][x] == gax:
                    bad_with[x][a].add(b)
                    bad_with[x][b].add(a)

    def can_add_fast(S_set, x):
        """Fast check using precomputed bad_with."""
        for a in S_set:
            if bad_with[x][a] & S_set:
                return False
        return True

    # Now branch and bound.
    # We'll use a recursive approach with memoization on (frozenset(S), frozenset(R)) but that's too large.
    # Instead, we'll do iterative deepening or just a simple DFS with pruning.
    # Given n=128, the search space is huge. Let's try a different approach.

    # Alternative: Use the fact that the problem is about avoiding equal-gcd triples.
    # Key insight: if we pick all primes, no two primes have gcd > 1, so no bad triple.
    # Number of primes <= 128: 31.
    # We can also pick 1, and numbers that are pairwise coprime with each other and with the primes.
    # Let's try a constructive approach: pick a set of "representatives" for each gcd value.

    # Actually, let's try a simpler heuristic first: greedy by smallest prime factor, then local search with better swaps.
    # But the directive says to use a different strategy. Let's do branch and bound with a tight bound.

    # For branch and bound, let's use the following:
    # - Order elements by some heuristic (e.g., number of conflicts).
    # - At each step, decide to include or exclude the next element.
    # - Upper bound: current_size + remaining_count (optimistic).
    # - This is too loose. Let's use: current_size + size of maximum independent set in the "conflict graph" of remaining elements, where two remaining elements are connected if they can't both be added to the current set.
    # Computing MIS is NP-hard, but for small remaining sets, we can do it.

    # Given the time limit, let's try a hybrid:
    # 1. Use a fast greedy to get a good initial solution.
    # 2. Use local search with 2-swap and 3-swap moves.
    # 3. Use a time limit to stop.

    # But this is similar to the current program. Let's try something truly different.

    # New idea: Formulate as a constraint satisfaction problem and use a SAT-like solver.
    # Variables: x_i for i in 1..128, x_i = 1 if i is in S.
    # Constraints: for every triple (a,b,c) with a<b<c, not (x_a and x_b and x_c) if gcd(a,b)==gcd(a,c)==gcd(b,c).
    # Maximize sum x_i.
    # This is a maximum independent set problem on a 3-uniform hypergraph.
    # We can use a simple branch and bound on the hypergraph.

    # Precompute all bad triples.
    bad_triples = []
    for a in range(1, N1):
        for b in range(a + 1, N1):
            g = gcd_mat[a][b]
            for c in range(b + 1, N1):
                if gcd_mat[a][c] == g and gcd_mat[b][c] == g:
                    bad_triples.append((a, b, c))

    # For branch and bound, we'll use a recursive function.
    # State: index in the sorted list of elements, current set, current size.
    # We'll sort elements by a heuristic: number of bad triples they participate in (ascending, so we try "safe" elements first).

    triple_count = [0] * N1
    for a, b, c in bad_triples:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1

    order = sorted(range(1, N1), key=lambda x: (triple_count[x], x))

    # Precompute for each element x, the set of bad triples that include x.
    bad_triples_with = [set() for _ in range(N1)]
    for i, (a, b, c) in enumerate(bad_triples):
        bad_triples_with[a].add(i)
        bad_triples_with[b].add(i)
        bad_triples_with[c].add(i)

    best_size = 0
    best_set = set()

    def backtrack(idx, current_set, current_size):
        nonlocal best_size, best_set
        if time.time() > DEADLINE - 2:
            return
        # Upper bound: current_size + (len(order) - idx)
        if current_size + (len(order) - idx) <= best_size:
            return
        if idx == len(order):
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return

        x = order[idx]

        # Option 1: exclude x
        backtrack(idx + 1, current_set, current_size)

        # Option 2: include x, if it doesn't create a bad triple with current_set
        # Check: for each pair (a,b) in current_set, if (a,b,x) is a bad triple, then can't include.
        # Use precomputed: for each a in current_set, check if bad_with[x][a] intersects current_set.
        can_include = True
        for a in current_set:
            if bad_with[x][a] & current_set:
                can_include = False
                break

        if can_include:
            current_set.add(x)
            backtrack(idx + 1, current_set, current_size + 1)
            current_set.remove(x)

    # This is too slow for n=128. Let's use a different approach.

    # Final approach: Use a fast greedy + local search with a better heuristic.
    # But to be different, let's use a simulated annealing approach with a conflict-based energy function.

    # Actually, let's just do a fast greedy with a good ordering, then local search.
    # The key difference: use a conflict-degree-based ordering and 2-opt swaps.

    def greedy_by_conflict():
        """Greedy: add elements in order of increasing conflict degree."""
        selected = []
        S_set = set()
        remaining = list(range(1, N1))
        # Compute dynamic conflict degree
        while remaining and time.time() < DEADLINE - 5:
            # Pick the element with the fewest conflicts with the current set
            best_x = None
            best_conf = float('inf')
            for x in remaining:
                conf = 0
                for a in S_set:
                    if bad_with[x][a] & S_set:
                        conf += 1
                if conf < best_conf:
                    best_conf = conf
                    best_x = x
                if conf == 0:
                    break
            if best_x is None or best_conf > 0:
                break
            selected.append(best_x)
            S_set.add(best_x)
            remaining.remove(best_x)
        return selected

    def local_search_2opt(S, seed_val=0):
        """Local search with 2-opt swaps."""
        import random
        random.seed(seed_val)
        S = sorted(S)
        S_set = set(S)
        remaining = [x for x in range(1, N1) if x not in S_set]
        improved = True
        while improved and time.time() < DEADLINE - 3:
            improved = False
            # Try to add
            for x in remaining[:]:
                if can_add_fast(S_set, x):
                    S.append(x)
                    S.sort()
                    S_set.add(x)
                    remaining.remove(x)
                    improved = True
                    break
            # Try 2-opt swap
            if not improved and remaining:
                for i in range(len(S)):
                    for j in range(len(remaining)):
                        a = S[i]
                        b = remaining[j]
                        new_S = S[:i] + S[i+1:] + [b]
                        new_S.sort()
                        new_S_set = set(new_S)
                        # Check validity
                        valid = True
                        for x in new_S_set:
                            for y in new_S_set:
                                if y <= x:
                                    continue
                                if bad_with[x][y] & new_S_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            S = new_S
                            S_set = new_S_set
                            remaining.remove(b)
                            remaining.append(a)
                            improved = True
                            break
                    if improved:
                        break
        return S

    # Get initial solution
    S = greedy_by_conflict()
    if not S:
        S = [1]
    S = local_search_2opt(S, 0)

    # Try multiple random restarts
    best = S
    for seed in range(1, 100):
        if time.time() > DEADLINE - 5:
            break
        random.seed(seed)
        # Random greedy
        S_rand = []
        S_set = set()
        remaining = list(range(1, N1))
        random.shuffle(remaining)
        for x in remaining:
            if can_add_fast(S_set, x):
                S_rand.append(x)
                S_set.add(x)
        S_rand = local_search_2opt(S_rand, seed)
        if len(S_rand) > len(best):
            best = S_rand

    # Final check
    if not best:
        best = [1]

    return sorted(best)


# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
