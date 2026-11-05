"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill

    MAX_VAL = 10000
    N = 12

    # Precompute smallest prime factor (SPF) for fast factorization
    def build_spf(limit):
        spf = list(range(limit + 1))
        for i in range(2, int(limit**0.5) + 1):
            if spf[i] == i:
                for j in range(i*i, limit+1, i):
                    if spf[j] == j:
                        spf[j] = i
        return spf

    spf = build_spf(2 * MAX_VAL)

    def prime_factors_fast(n):
        """Return frozenset of prime factors of n using precomputed SPF."""
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return frozenset(factors)

    def count_distinct_primes_in_sums(nums):
        """Count distinct primes across all pairwise sums."""
        primes = set()
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                s = nums[i] + nums[j]
                primes.update(prime_factors_fast(s))
        return len(primes)

    # Precompute prime factors for all sums up to 2*MAX_VAL
    sum_prime_factors = {}
    for s in range(2, 2 * MAX_VAL + 1):
        sum_prime_factors[s] = prime_factors_fast(s)

    def get_sum_factors(a, b):
        """Get precomputed prime factors of a+b."""
        return sum_prime_factors[a + b]

    def build_compatibility_graph(candidate_set, prime_set, max_val):
        """
        Build a graph where nodes are candidates and edges connect compatible pairs.
        Two numbers x,y are compatible if all prime factors of x+y are in prime_set.
        """
        n = len(candidate_set)
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                s = candidate_set[i] + candidate_set[j]
                factors = sum_prime_factors[s]
                if factors.issubset(prime_set):
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def find_clique_backtrack(adj, target_size, deadline, max_time=5.0):
        """
        Find a clique of size target_size in the graph using backtracking.
        Returns the list of node indices if found, else None.
        """
        n = len(adj)
        best_clique = []
        found = [False]
        start = time.time()

        def backtrack(clique, candidates):
            if time.time() - start > max_time:
                return
            if len(clique) >= target_size:
                found[0] = True
                best_clique.clear()
                best_clique.extend(clique)
                return
            if not candidates:
                return
            # Try each candidate
            for idx in range(len(candidates)):
                node = candidates[idx]
                # Check if node is compatible with all in clique
                compatible = True
                for c in clique:
                    if c not in adj[node]:
                        compatible = False
                        break
                if not compatible:
                    continue
                # Recurse
                new_candidates = [c for c in candidates[idx+1:] if c in adj[node]]
                backtrack(clique + [node], new_candidates)
                if found[0]:
                    return

        backtrack([], list(range(n)))
        return list(best_clique) if found[0] else None

    def greedy_clique(adj, target_size):
        """
        Greedily find a clique using degree-based heuristic.
        """
        n = len(adj)
        selected = []
        remaining = set(range(n))

        for _ in range(target_size):
            if not remaining:
                break
            # Pick node with highest degree within remaining
            best_node = None
            best_degree = -1
            for node in remaining:
                deg = len(adj[node] & set(selected)) + len(adj[node] & remaining)
                # Prefer nodes connected to all selected
                if all(s in adj[node] for s in selected):
                    deg_to_remaining = len(adj[node] & remaining)
                    if deg_to_remaining > best_degree:
                        best_degree = deg_to_remaining
                        best_node = node
            if best_node is None:
                break
            selected.append(best_node)
            remaining.discard(best_node)
            # Remove nodes not connected to best_node
            remaining = {r for r in remaining if best_node in adj[r]}

        if len(selected) >= target_size:
            return selected[:target_size]
        return None

    # Strategy: Try small prime sets, build compatibility graph, find clique
    prime_set_candidates = [
        frozenset([2, 3]),
        frozenset([2, 3, 5]),
        frozenset([2, 3, 5, 7]),
        frozenset([2, 3, 5, 7, 11]),
        frozenset([2, 3, 5, 7, 11, 13]),
    ]

    best_result = None
    best_score = float('inf')

    for prime_set in prime_set_candidates:
        if time.time() > deadline - 10:
            break

        # Generate candidate numbers: those that can form valid sums
        # A number x is a candidate if there exists some y such that all prime factors of x+y are in prime_set
        # Simplification: just use numbers 1..max_val and check compatibility
        for max_val in [300, 600, 1000, 2000, 5000, 10000]:
            if time.time() > deadline - 5:
                break

            candidates = list(range(1, min(max_val, MAX_VAL) + 1))
            n_cand = len(candidates)

            # Build adjacency matrix (compatibility graph)
            # Use a bitset approach for speed: adj[i] is a set of j where i,j are compatible
            adj = [set() for _ in range(n_cand)]

            for i in range(n_cand):
                if time.time() > deadline - 3:
                    break
                xi = candidates[i]
                for j in range(i+1, n_cand):
                    xj = candidates[j]
                    s = xi + xj
                    factors = sum_prime_factors[s]
                    if factors.issubset(prime_set):
                        adj[i].add(j)
                        adj[j].add(i)

            # Try greedy first (fast)
            clique = greedy_clique(adj, N)
            if clique is not None:
                nums = sorted([candidates[i] for i in clique])
                score = count_distinct_primes_in_sums(nums)
                if score < best_score:
                    best_score = score
                    best_result = nums
                    if score == len(prime_set):
                        return best_result

            # Try backtracking if greedy failed and we have time
            if clique is None and time.time() < deadline - 8:
                clique = find_clique_backtrack(adj, N, deadline, max_time=3.0)
                if clique is not None:
                    nums = sorted([candidates[i] for i in clique])
                    score = count_distinct_primes_in_sums(nums)
                    if score < best_score:
                        best_score = score
                        best_result = nums
                        if score == len(prime_set):
                            return best_result

    # Local search refinement if we have a result and time
    if best_result is not None and time.time() < deadline - 10:
        current = best_result[:]
        current_score = best_score

        # Generate a pool of replacement candidates
        smooth_nums = []
        for a in range(14):
            for b in range(14):
                for c in range(14):
                    n = (2 ** a) * (3 ** b) * (5 ** c)
                    if 1 <= n <= MAX_VAL:
                        smooth_nums.append(n)
        smooth_nums = sorted(set(smooth_nums))
        pool = set(smooth_nums) | set(range(1, 200))

        for iteration in range(200):
            if time.time() > deadline - 3:
                break

            improved = False
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                for new_val in sorted(pool):
                    if new_val in current:
                        continue
                    trial = current[:]
                    trial[i] = new_val
                    trial.sort()
                    trial_score = count_distinct_primes_in_sums(trial)
                    if trial_score < current_score:
                        current = trial
                        current_score = trial_score
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        if current_score < best_score:
            best_score = current_score
            best_result = current[:]

    if best_result is not None:
        return best_result

    # Final fallback
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# EVOLVE-BLOCK-END
