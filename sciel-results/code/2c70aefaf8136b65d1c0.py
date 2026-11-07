import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 256

    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Small primes (p^2 <= 256, i.e., p <= 16): can appear with powers > 1
    # Large primes (p^2 > 256, i.e., p > 16): can only appear to the first power
    small_primes = [p for p in primes if p * p <= N]   # 2,3,5,7,11,13
    large_primes = [p for p in primes if p * p > N]

    # For each small prime, list all its powers <= N
    small_prime_powers = {}
    for p in small_primes:
        powers = []
        val = p
        while val <= N:
            powers.append(val)
            val *= p
        small_prime_powers[p] = powers

    # Precompute all possible "skeletons": products of one power from a subset of small primes
    # A skeleton is a product of distinct small-prime powers, each from a different small prime
    # We enumerate all such products via recursion
    skeletons_set = set()
    skeletons_set.add(1)  # empty product

    def gen_skeletons(idx, current_product):
        if time.time() > deadline:
            return
        if idx == len(small_primes):
            skeletons_set.add(current_product)
            return
        p = small_primes[idx]
        # Option: skip this prime
        gen_skeletons(idx + 1, current_product)
        # Option: multiply by each power of p
        for pow_val in small_prime_powers[p]:
            new_prod = current_product * pow_val
            if new_prod <= N:
                gen_skeletons(idx + 1, new_prod)

    gen_skeletons(0, 1)
    skeletons = sorted(skeletons_set)

    # Now we need to select a subset of these skeletons such that:
    # 1. Each small prime's power is used in at most one skeleton (i.e., for each small prime p,
    #    at most one skeleton in our selection has p dividing it)
    # 2. We assign large primes to skeletons to maximize total sum
    #
    # Actually, the constraint is: each small prime can appear in at most one selected skeleton.
    # And each large prime can be assigned to at most one skeleton (or left alone).
    #
    # Let's think of it differently:
    # We select a collection of pairwise coprime numbers from 1..256.
    # Each number has a unique prime factorization.
    # The constraint is that no prime factor appears in more than one number.
    #
    # This is equivalent to: partition a subset of primes (with powers) into groups,
    # where each group's product is <= 256, and we want to maximize the sum of products + 1.
    #
    # Since small primes can have powers, let's enumerate all valid "atoms":
    # An atom is a number of the form (product of small prime powers) * (optional large prime)
    # such that the number <= 256.
    #
    # Actually, let's use a cleaner formulation:
    # We have a set of "items" to assign. Each item is either:
    #   - A small prime power p^k (for small prime p, power k>=1)
    #   - A large prime p
    # We partition a subset of these items into groups, each group's product <= 256.
    # Maximize sum of group products + 1.
    #
    # But this is complex. Let's use the skeleton approach more carefully.
    #
    # Key insight: For small primes, the "power" choice is coupled with which skeleton it goes into.
    # For large primes, they can either be standalone or multiplied into a skeleton.
    #
    # Let's enumerate all possible "skeletons" (products of small prime powers) as above.
    # For each skeleton s, we can optionally multiply it by one large prime p (if s*p <= 256).
    #
    # The constraint is:
    # - Each small prime's power is used in at most one skeleton (across all selected skeletons)
    # - Each large prime is used in at most one skeleton (or standalone)
    #
    # This is a complex combinatorial problem. Given the small size, let's use DFS with pruning.
    #
    # Alternative approach: Since the number of small primes is only 6, we can enumerate all
    # possible "assignments" of small prime powers to skeletons, then solve for large primes.
    #
    # Let me think about this differently. The optimal solution will include 1 and a set of
    # pairwise coprime numbers. Let's enumerate all maximal such sets using a different strategy.
    #
    # Strategy: Enumerate all possible selections of small-prime-power assignments, then for each,
    # greedily/optimally assign large primes.
    #
    # For small primes, each small prime p can contribute p^k for some k>=0 (k=0 means unused).
    # But p^k must be part of some number in the set. Multiple small primes can combine in one number.
    #
    # Let's enumerate all possible "base numbers" formed by small primes:
    # A base number is a product of powers of small primes, <= 256.
    # For each base number, we track which small primes it uses (and their exponents).
    #
    # Then we select a set of base numbers such that no small prime is used in more than one.
    # For each selected base number, we can optionally multiply by a large prime.
    #
    # Number of base numbers: let's count.
    # 2^a * 3^b * 5^c * 7^d * 11^e * 13^f <= 256
    # This is manageable.

    # Generate all base numbers (products of small prime powers) with their prime usage
    base_numbers = []  # list of (value, frozenset of (prime, exponent) pairs)
    def gen_base(idx, current_val, usage):
        if time.time() > deadline:
            return
        if idx == len(small_primes):
            if current_val > 1:  # skip 1 (empty product) as a base, we'll handle 1 separately
                base_numbers.append((current_val, frozenset(usage)))
            return
        p = small_primes[idx]
        # Try each power of p
        val = 1
        exp = 0
        while val <= N // current_val:
            if current_val * val <= N:
                new_usage = list(usage)
                if exp > 0:
                    new_usage.append((p, exp))
                gen_base(idx + 1, current_val * val, new_usage)
            val *= p
            exp += 1

    gen_base(0, 1, [])

    # Deduplicate base numbers by value (keep track of usage)
    # Actually, each value has a unique prime factorization, so no duplicates.
    # But let's verify: for a given value, the usage is determined. So we can just use value -> usage dict.
    base_to_usage = {}
    for val, usage in base_numbers:
        base_to_usage[val] = usage

    # Now enumerate all valid sets of base numbers (no shared small primes)
    # This is a set packing problem on 6 elements, so it's small.
    # Use DFS to enumerate all maximal (or all) valid combinations.

    valid_base_sets = []  # list of (tuple of base values, total small prime usage as frozenset)

    def gen_base_sets(idx, selected_bases, used_primes):
        if time.time() > deadline:
            return
        if idx == len(base_numbers):
            if selected_bases:
                valid_base_sets.append(tuple(selected_bases))
            return
        val, usage = base_numbers[idx]
        # Option: don't select this base
        gen_base_sets(idx + 1, selected_bases, used_primes)
        # Option: select this base if no conflict
        usage_set = frozenset(usage)
        if not used_primes & usage_set:
            gen_base_sets(idx + 1, selected_bases + [val], used_primes | usage_set)

    gen_base_sets(0, [], frozenset())

    # For each valid set of base numbers, we need to assign large primes optimally.
    # For a set of bases B = {b1, b2, ..., bk}, we can assign each large prime p to at most one base bi
    # (if bi * p <= 256) or leave p standalone.
    #
    # The contribution of assigning p to bi is: bi * p instead of bi + p, so gain = bi * p - bi - p = (bi-1)(p-1) - 1
    # Wait: if we don't assign p, we get bi + p. If we assign p to bi, we get bi*p.
    # Gain = bi*p - bi - p = bi*p - bi - p.
    # We want to maximize total sum, which is sum(bi) + sum(p for unassigned) + sum(bi*p - bi - p for assigned)
    # = sum(bi) + sum_all_large_primes + sum(gains for assigned)
    # where sum_all_large_primes is the sum of all large primes (since unassigned ones contribute p,
    # and assigned ones contribute bi*p instead of bi + p, so the net change from baseline is bi*p - bi - p).
    #
    # Actually, let's define:
    # Baseline: sum of all bases + sum of all large primes (all standalone)
    # If we assign p to bi: replace bi + p with bi*p. Change = bi*p - bi - p.
    # Total = baseline + sum(changes for assigned pairs)
    #
    # So we want to maximize sum of changes, where change(bi, p) = bi*p - bi - p.
    #
    # This is a maximum weight matching problem: bases on one side, large primes on the other,
    # edge weight = change(bi, p) if bi*p <= 256, else no edge.
    # We want to find a matching that maximizes total weight. We can also leave nodes unmatched.
    #
    # Since |bases| <= 6 and |large_primes| is ~40, we can use DFS or Hungarian algorithm.
    # Given the small size, DFS with pruning is fine.

    sum_all_large = sum(large_primes)
    best_total = -1
    best_result = [1]

    for base_set in valid_base_sets:
        if time.time() > deadline:
            break

        base_sum = sum(base_set)
        baseline = base_sum + sum_all_large

        # Solve max weight matching
        m = len(base_set)
        # For each base, list of (gain, prime) for large primes that can be assigned
        candidates = []
        for b in base_set:
            cand = []
            for p in large_primes:
                if b * p <= N:
                    gain = b * p - b - p
                    if gain > 0:  # only positive gains are useful
                        cand.append((gain, p))
            cand.sort(reverse=True)
            candidates.append(cand[:15])  # keep top 15

        # DFS for max weight matching
        best_gain = 0
        best_assignment = [None] * m

        def dfs_match(idx, used_primes, current_gain, assignment):
            nonlocal best_gain, best_assignment
            if time.time() > deadline:
                return
            if idx == m:
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_assignment = assignment.copy()
                return

            b = base_set[idx]

            # Option: don't assign any prime to this base
            assignment[idx] = None
            dfs_match(idx + 1, used_primes, current_gain, assignment)

            # Option: assign a prime
            for gain, p in candidates[idx]:
                if p not in used_primes:
                    used_primes.add(p)
                    assignment[idx] = p
                    dfs_match(idx + 1, used_primes, current_gain + gain, assignment)
                    used_primes.remove(p)
            assignment[idx] = None

        dfs_match(0, set(), 0, [None] * m)

        total = baseline + best_gain

        if total > best_total:
            best_total = total
            # Reconstruct result
            res = [1]
            used_large = set()
            for i, b in enumerate(base_set):
                p = best_assignment[i]
                if p is not None:
                    res.append(b * p)
                    used_large.add(p)
                else:
                    res.append(b)
            for p in large_primes:
                if p not in used_large:
                    res.append(p)

            # Validate
            if len(set(res)) != len(res):
                continue
            if any(x < 1 or x > N for x in res):
                continue
            valid = True
            for i in range(len(res)):
                for j in range(i + 1, len(res)):
                    if gcd(res[i], res[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_result = res

    # Also try the trivial case: just 1 and all large primes standalone
    trivial = [1] + large_primes
    trivial_sum = sum(trivial)
    if trivial_sum > best_total:
        best_result = trivial
        best_total = trivial_sum

    # Ensure we return a proper list of Python ints
    return [int(x) for x in best_result]
