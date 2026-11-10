"""Optimized solver for equal-gcd triple avoidance on [1,128]."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s kill

    N = 128
    nums = list(range(1, N + 1))

    # GCD table
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]

    # conflict_mask[a][b] = bitmask of x (x != a, x != b) such that
    # gcd(a,x) == gcd(b,x) == gcd(a,b)
    # Using Python ints as bitsets (bit i set means element i is in the mask)
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = g[a][b]
            mask = 0
            for x in range(1, N + 1):
                if x != a and x != b:
                    if g[a][x] == gab and g[b][x] == gab:
                        mask |= (1 << x)
            conflict[a][b] = mask
            conflict[b][a] = mask

    # Build conflict graph: edge (a,b) exists if there's a third element c
    # that would form a bad triple with a and b.
    # Actually, the independent set problem: we want a set S where no triple
    # has all three pairwise gcds equal.
    # 
    # Reformulation: For a set S, it's valid iff for every pair (a,b) in S,
    # there is no c in S (c != a, c != b) with gcd(a,c)=gcd(b,c)=gcd(a,b).
    #
    # This is NOT a simple graph independent set problem because the constraint
    # is on triples, not pairs. However, we can still use a greedy + local search
    # approach but with much better heuristics.
    #
    # Alternative: Think of it as a 3-uniform hypergraph independent set problem.
    # Vertices: 1..128. Hyperedge: {a,b,c} if gcd(a,b)=gcd(a,c)=gcd(b,c).
    # We want the largest set with no hyperedge fully contained.
    #
    # For maximum independent set in a hypergraph, we can use:
    # 1. Branch and bound with good upper bounds
    # 2. Or reduce to a graph problem via auxiliary variables
    #
    # Let me use a different approach: iterative improvement with smart local search.

    def can_add(x, S_set, S_list):
        """Check if x can be added to S without creating a bad triple."""
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (conflict[a][b] >> x) & 1:
                    return False
        return True

    def count_conflicts(x, S_list):
        """Count pairs in S_list that would conflict with x."""
        count = 0
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (conflict[a][b] >> x) & 1:
                    count += 1
        return count

    def find_all_violations(S_list):
        """Find all violating triples."""
        violations = []
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                mask = conflict[a][b]
                # Check which elements in S_list are in the mask
                for k in range(j + 1, n):
                    c = S_list[k]
                    if (mask >> c) & 1:
                        violations.append((a, b, c))
        return violations

    # Strategy: Use a sophisticated local search with multiple phases.
    # Phase 1: Build initial solution with multiple greedy strategies
    # Phase 2: Intensive local search with 1-move, 2-swap, and 3-exchange moves
    # Phase 3: Branch and bound for final optimization

    best_set = set()
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)

    # Precompute for each element x, a "conflict score" with other elements
    # to guide greedy ordering
    def conflict_degree(x, S_list):
        """Number of pairs in S_list that conflict with x."""
        return count_conflicts(x, S_list)

    # Multiple greedy strategies
    def greedy(order):
        S = set()
        S_list = []
        for x in order:
            if can_add(x, S, S_list):
                S.add(x)
                S_list.append(x)
        return S

    # Strategy 1: Greedy with various orderings
    all_nums = nums[:]

    # Precompute useful properties
    def prime_factors(n):
        factors = set()
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                factors.add(d)
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)
        return factors

    pf = {x: prime_factors(x) for x in all_nums}
    omega = {x: len(pf[x]) for x in all_nums}  # number of distinct prime factors

    # Ordering strategies
    import random
    random.seed(42)

    strategies = []

    # 1. Random orderings
    for _ in range(20):
        order = all_nums[:]
        random.shuffle(order)
        strategies.append(order)

    # 2. By omega (fewer prime factors first)
    strategies.append(sorted(all_nums, key=lambda x: (omega[x], x)))
    strategies.append(sorted(all_nums, key=lambda x: (-omega[x], x)))

    # 3. By value
    strategies.append(sorted(all_nums))
    strategies.append(sorted(all_nums, reverse=True))

    # 4. By gcd with small numbers
    for p in [2, 3, 5, 7]:
        strategies.append(sorted(all_nums, key=lambda x: (x % p, omega[x], x)))

    # 5. Primes first
    primes = [x for x in all_nums if omega[x] == 1 and x > 1]
    composites = [x for x in all_nums if x not in primes]
    strategies.append(primes + composites)
    strategies.append(composites + primes)

    # 6. Powers of 2 first
    powers2 = [x for x in all_nums if x & (x - 1) == 0]
    non_powers2 = [x for x in all_nums if x not in powers2]
    strategies.append(powers2 + non_powers2)

    for order in strategies:
        if time.time() > deadline - 30:
            break
        S = greedy(order)
        update_best(S)

    # Local search: try to add elements, swap, etc.
    def local_search(S, max_iters=500):
        S = set(S)
        S_list = list(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 10:
            improved = False
            iters += 1

            # Try adding any element not in S
            for x in all_nums:
                if x not in S:
                    if can_add(x, S, S_list):
                        S.add(x)
                        S_list.append(x)
                        improved = True

            # If we added something, restart the loop
            if improved:
                continue

            # Try 1-for-1 swaps: remove a, add b
            for a in list(S_list):
                S_trial = S - {a}
                S_trial_list = [x for x in S_list if x != a]
                for b in all_nums:
                    if b not in S_trial:
                        if can_add(b, S_trial, S_trial_list):
                            S = S_trial | {b}
                            S_list = list(S)
                            improved = True
                            break
                if improved:
                    break

            if improved:
                continue

            # Try 2-for-1: remove two, add one (net -1, but might enable more adds)
            # Actually, let's try: remove one, then greedily add as many as possible
            for a in list(S_list):
                S_trial = S - {a}
                S_trial_list = [x for x in S_list if x != a]
                # Greedily add
                for b in all_nums:
                    if b not in S_trial:
                        if can_add(b, S_trial, S_trial_list):
                            S_trial.add(b)
                            S_trial_list.append(b)
                if len(S_trial) > len(S):
                    S = S_trial
                    S_list = list(S)
                    improved = True
                    break

        return S

    # Apply local search to best
    if best_size > 0 and time.time() < deadline - 20:
        S_ls = local_search(best_set)
        update_best(S_ls)

    # Branch and bound approach for final optimization
    # Use a recursive search with pruning

    def branch_and_bound(S_current, candidates, depth=0):
        """
        S_current: current set (as a set)
        candidates: list of elements not yet considered
        Returns best set found
        """
        nonlocal best_set, best_size
        best = set(S_current)

        if time.time() > deadline - 5:
            return best

        if not candidates:
            if len(best) > len(best_set):
                best_set = set(best)
                best_size = len(best)
            return best

        # Upper bound: len(S_current) + len(candidates)
        # If this can't beat best, prune
        if len(S_current) + len(candidates) <= best_size:
            return best

        # Try adding each candidate
        for i, x in enumerate(candidates):
            if time.time() > deadline - 5:
                return best

            # Check if x can be added
            S_list = list(S_current)
            can = True
            for ai in range(len(S_list)):
                a = S_list[ai]
                for bi in range(ai + 1, len(S_list)):
                    b = S_list[bi]
                    if (conflict[a][b] >> x) & 1:
                        can = False
                        break
                if not can:
                    break

            if can:
                S_new = S_current | {x}
                # Recurse with remaining candidates
                remaining = candidates[i + 1:]
                result = branch_and_bound(S_new, remaining, depth + 1)
                if len(result) > len(best):
                    best = result

            # Also try not adding x (skip)
            # This is implicitly handled by the loop structure

        if len(best) > len(best_set):
            best_set = set(best)
            best_size = len(best)

        return best

    # Use B&B on a reduced candidate set
    if time.time() < deadline - 15 and best_size > 0:
        # Start from best_set, try to add more
        S_list = list(best_set)
        candidates = [x for x in all_nums if x not in best_set]

        # Sort candidates by conflict count (fewer conflicts first)
        candidates.sort(key=lambda x: count_conflicts(x, S_list))

        # Limit candidates to top 30 for B&B
        limited_candidates = candidates[:30]

        if len(limited_candidates) > 0:
            result = branch_and_bound(best_set, limited_candidates)
            update_best(result)

    # Final polish: try to add any remaining elements
    if best_size > 0 and time.time() < deadline - 3:
        S_list = list(best_set)
        for x in all_nums:
            if x not in best_set:
                if can_add(x, best_set, S_list):
                    best_set.add(x)
                    S_list.append(x)

    # Verify no violations
    S_list = list(best_set)
    violations = find_all_violations(S_list)
    if violations:
        # Remove elements to fix violations
        while violations and time.time() < deadline - 1:
            # Count conflicts per element
            conf_count = {}
            for a, b, c in violations:
                for v in [a, b, c]:
                    conf_count[v] = conf_count.get(v, 0) + 1
            # Remove element with max conflicts
            max_c = -1
            to_remove = None
            for v in best_set:
                cc = conf_count.get(v, 0)
                if cc > max_c:
                    max_c = cc
                    to_remove = v
            if to_remove:
                best_set.remove(to_remove)
            S_list = list(best_set)
            violations = find_all_violations(S_list)

    return list(best_set)
# EVOLVE-BLOCK-END
