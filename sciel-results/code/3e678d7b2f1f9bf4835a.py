"""Pairwise coprime integers below 512 maximizing sum 1/(512 - a).

Strategy:
  - Every integer a in [1,511] has a smallest prime factor spf(a).
  - In any pairwise-coprime set, at most one element can have a given spf.
  - For each prime p <= 23, the best (highest-value) element with spf = p
    must be included in any optimal solution (it dominates all others with
    the same spf and is coprime to everything coprime to p).
  - 511 = 7*73 has spf 7; it is the highest-value element with spf 7.
  - For primes p in (23, 512), we have freedom to pick 0 or 1 element
    with spf = p. The optimal choice is a small knapsack / DP over these
    primes, since the number of such primes is small (~85) and for each
    we only need to consider the top few candidates.
"""

def construct():
    from math import gcd
    import time

    deadline = time.time() + 110  # 110s safety margin under 120s limit
    N = 512

    # Sieve smallest prime factor
    spf = list(range(N))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N, i):
                if spf[j] == j:
                    spf[j] = i

    # Group numbers by their smallest prime factor
    # For each prime p, collect all a in [1, 511] with spf(a) = p
    from collections import defaultdict
    groups = defaultdict(list)
    for a in range(1, N):
        p = spf[a]
        groups[p].append(a)

    # For each prime, sort candidates by value 1/(512-a) descending
    for p in groups:
        groups[p].sort(key=lambda a: 1.0 / (N - a), reverse=True)

    # Identify primes that are "forced": for small primes, the best element
    # with that spf must be in any optimal solution.
    # Actually, for ALL primes p, the best element with spf=p dominates all
    # others with the same spf. But we can only pick at most one per spf.
    # The question is whether to pick one at all.
    #
    # Key insight: if p is small (p <= 511/2 = 255.5), then 2p <= 511, so
    # the largest multiple of p below 512 that has spf=p... hmm this is
    # getting complex. Let me think differently.
    #
    # For a prime p, the elements with spf=p are: p itself, and composites
    # a = p * m where m >= 2, gcd(m, p) = 1, and all prime factors of m
    # are >= p (so that spf(a) = p).
    #
    # The value of a is 1/(512-a), maximized when a is as large as possible.
    #
    # For the DP: we consider primes p > 23 (primes where we have real
    # choice). For p <= 23, we argue the best candidate must be included
    # because excluding it would lose too much value and we can't compensate.
    #
    # Actually, let's just do a proper DP over ALL primes.
    # Number of primes < 512: about 97. For each, we consider top K=3 candidates.
    # DP state: for each prime, either skip or pick one candidate.
    # This is a 0-1 knapsack-like problem with small per-item choices.

    # Get all primes < 512
    primes = [p for p in range(2, N) if spf[p] == p]

    # For each prime, get top candidates (by value)
    K = 5  # top K candidates per prime
    prime_candidates = []
    for p in primes:
        if p in groups:
            cands = groups[p][:K]
        else:
            cands = [p]  # at least p itself
        prime_candidates.append((p, cands))

    # DP: dp[i] = best value using first i primes
    # We need to track the actual choices for reconstruction
    # Since n_primes ~ 97 and K=5, state space is manageable
    # But storing full lists in DP is expensive. Let's use parent pointers.

    n_primes = len(prime_candidates)
    # dp[j] = (best_value, list_of_choices) for first j primes
    # Actually let's do: dp[i][k] not needed; it's just sequential.

    # Better: dp[i] = (best_value, choices_tuple) after processing first i primes
    # choices_tuple stores the chosen a for each prime (or 0 for skip)
    # This is O(n_primes * product of choices) which is too large.

    # Instead: greedy + local search, or a smarter DP.
    # For each prime, the marginal value of picking the best candidate vs skipping
    # is positive (1/(512-a) > 0), so we should always pick the best candidate
    # for every prime. Wait - but two different primes' best candidates might
    # not be coprime to each other!
    #
    # Example: best with spf=3 might be 3*167=501, best with spf=5 might be
    # 5*101=505. gcd(501,505)=? 501=3*167, 505=5*101. gcd=1. OK.
    # But what about spf=3: 3*167=501 vs spf=7: 7*71=497? gcd(501,497)=?
    # 501=3*167, 497=7*71. gcd=1.
    # Actually, if spf(a)=p and spf(b)=q with p!=q, then gcd(a,b) could still
    # be > 1 if a and b share a larger prime factor.
    # Example: a = 3*11 = 33 (spf=3), b = 5*11 = 55 (spf=5). gcd=11.
    #
    # So the candidates for different primes CAN conflict. This makes it a
    # proper combinatorial optimization problem.

    # Given the complexity, let's use a greedy approach with local search:
    # 1. Start with best candidate for each prime.
    # 2. If conflicts (non-coprime pairs) exist, resolve by removing the
    #    lower-value element from each conflicting pair.
    # 3. Local search: try swapping, adding, removing to improve.

    # Step 1: Initial assignment - best candidate per prime
    initial = []
    for p, cands in prime_candidates:
        initial.append((p, cands[0]))  # (prime, best_a)

    # Check for conflicts and resolve
    def resolve_conflicts(assignment):
        """Resolve conflicts by removing lower-value elements."""
        chosen = [a for p, a in assignment]
        changed = True
        while changed:
            changed = False
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    if chosen[i] is None or chosen[j] is None:
                        continue
                    g = gcd(chosen[i], chosen[j])
                    if g > 1:
                        # Remove the one with lower value
                        vi = 1.0 / (N - chosen[i])
                        vj = 1.0 / (N - chosen[j])
                        if vi < vj:
                            chosen[i] = None
                        else:
                            chosen[j] = None
                        changed = True
                        break
                if changed:
                    break
        return [a for a in chosen if a is not None]

    current = resolve_conflicts(initial)
    current_set = set(current)

    # Compute score
    def score(sol):
        return sum(1.0 / (N - a) for a in sol)

    best_score = score(current)
    best_solution = list(current)

    # Local search: try to improve by:
    # 1. Adding back removed elements if they don't conflict
    # 2. Swapping elements
    # 3. Trying second-best candidates for each prime

    # Build list of all candidate a values (top K per prime)
    all_candidates = set()
    for p, cands in prime_candidates:
        for a in cands:
            all_candidates.add(a)

    # Also include some high-value numbers not in top K
    for a in range(511, 0, -1):
        all_candidates.add(a)
        if len(all_candidates) > 300:
            break

    all_candidates = list(all_candidates)
    all_candidates.sort(key=lambda a: 1.0 / (N - a), reverse=True)

    # Local search iterations
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1

        # Try adding each candidate not in current solution
        for a in all_candidates:
            if a in current_set:
                continue
            # Check if coprime to all current
            if all(gcd(a, b) == 1 for b in current):
                new_sol = current + [a]
                ns = score(new_sol)
                if ns > best_score + 1e-15:
                    best_score = ns
                    best_solution = new_sol
                    current = new_sol
                    current_set = set(current)
                    improved = True
                    break

        # Try swapping: remove one, add another
        if not improved:
            for i, a_remove in enumerate(current):
                if time.time() > deadline:
                    break
                for a_add in all_candidates[:100]:
                    if a_add in current_set:
                        continue
                    if a_add == a_remove:
                        continue
                    # Check a_add coprime to all others
                    others = [b for j, b in enumerate(current) if j != i]
                    if all(gcd(a_add, b) == 1 for b in others):
                        new_sol = [b for j, b in enumerate(current) if j != i] + [a_add]
                        ns = score(new_sol)
                        if ns > best_score + 1e-15:
                            best_score = ns
                            best_solution = new_sol
                            current = new_sol
                            current_set = set(current)
                            improved = True
                            break
                if improved:
                    break

        # Try removing one element (if it allows adding two better ones)
        if not improved and len(current) > 10:
            for i, a_remove in enumerate(current):
                if time.time() > deadline:
                    break
                temp_current = [b for j, b in enumerate(current) if j != i]
                temp_set = set(temp_current)
                # Try to add two elements
                best_pair_gain = 0
                best_pair = None
                for a1 in all_candidates[:50]:
                    if a1 in temp_set:
                        continue
                    if not all(gcd(a1, b) == 1 for b in temp_current):
                        continue
                    for a2 in all_candidates[:50]:
                        if a2 in temp_set or a2 == a1:
                            continue
                        if not all(gcd(a2, b) == 1 for b in temp_current):
                            continue
                        if gcd(a1, a2) != 1:
                            continue
                        gain = (1.0/(N-a1) + 1.0/(N-a2)) - 1.0/(N-a_remove)
                        if gain > best_pair_gain:
                            best_pair_gain = gain
                            best_pair = (a1, a2)
                if best_pair and best_pair_gain > 1e-15:
                    new_sol = temp_current + [best_pair[0], best_pair[1]]
                    ns = score(new_sol)
                    if ns > best_score + 1e-15:
                        best_score = ns
                        best_solution = new_sol
                        current = new_sol
                        current_set = set(current)
                        improved = True
                        break

    # Final validation
    sol = best_solution
    assert len(set(sol)) == len(sol), "duplicates"
    assert all(1 <= a <= 511 for a in sol), "out of range"
    assert all(gcd(a, b) == 1 for i, a in enumerate(sol) for b in sol[i+1:]), "not pairwise coprime"

    return sol
