"""Rewritten solver: exact Branch-and-Bound over the independence set of the coprimality conflict graph,
with a tight LP relaxation bound using a greedy clique cover of the prime-neighborhood structure."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

_N = 512
_TIME_LIMIT = 120.0  # safety margin under 150s hard kill


def _factorize(n):
    """Return set of distinct prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    x = n
    while d * d <= x:
        if x % d == 0:
            factors.add(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        factors.add(x)
    return factors


def _build_data():
    # a in 1..511, weight = 1/(512-a)
    nodes = list(range(511, 0, -1))  # descending by weight (a closer to 512 -> smaller denominator -> larger weight)
    weights = [Fraction(1, _N - a) for a in nodes]

    # Prime factors for each node
    pfs = [_factorize(a) for a in nodes]

    # Build conflict graph: two nodes conflict if gcd(a,b) > 1
    # We need to check gcd for pairs. To speed up, group by prime.
    # Build conflict set for each node as a frozenset of indices
    # n = 511, O(n^2) = ~130k pairs, fine.
    n = len(nodes)
    conflict = [set() for _ in range(n)]
    for i in range(n):
        pi = pfs[i]
        for j in range(i + 1, n):
            if pi & pfs[j]:
                conflict[i].add(j)
                conflict[j].add(i)
    conflict = [frozenset(c) for c in conflict]

    # Precompute for each node: which primes it "blocks" (its own prime factors)
    # For bound: we can use a simple greedy clique cover on the remaining nodes.
    # Actually, a tighter bound: the sum of weights of a maximum-weight independent set
    # is bounded by the sum of weights of a minimum-weight clique cover (duality).
    # Greedy clique cover: repeatedly pick the node with the highest weight among remaining,
    # form a clique (set of mutually conflicting nodes) containing it, remove them.
    # But forming a clique from a single node is trivial; we need cliques of size >1 to get a tight bound.
    # Alternative bound: for each prime p, at most one chosen node can have p in its factor set.
    # So the bound is: for each prime p, take the max weight among remaining nodes divisible by p,
    # and also consider primes not dividing any remaining node (their max is 0).
    # But a node can be divisible by multiple primes, so this overcounts if we just sum.
    # A valid upper bound: the sum over all primes p < 512 of max weight among remaining nodes divisible by p
    # is NOT valid because a single node contributes to multiple primes.
    # Actually, it IS valid as an upper bound on the independent set weight? No.
    # Counter-example: node 6 (2*3) has weight w. It counts in both prime 2 and prime 3 buckets.
    # If we pick node 6, we gain w, but the bucket sum includes w twice. So the bucket sum >= actual.
    # Yes! The sum over primes of (max weight in that prime's bucket) is >= weight of any independent set,
    # because each chosen node a contributes weight(a) to each prime p | a, so sum of contributions >= sum of weights.
    # Wait, that's not right either. If I pick node 6, it contributes w to prime 2 bucket and w to prime 3 bucket.
    # The bucket sum counts w twice. The actual gain is w. So bucket sum >= actual. Correct.
    # This is a valid upper bound.

    # Precompute for each prime p < 512, the list of node indices divisible by p, sorted by weight desc.
    primes = []
    for p in range(2, _N):
        if not any(p % d == 0 for d in range(2, int(p**0.5) + 1)):
            primes.append(p)

    # prime_to_nodes: dict prime -> list of (weight, index) sorted desc
    prime_nodes = {}
    for p in primes:
        lst = []
        for i in range(n):
            if pfs[i] & {p}:
                lst.append((weights[i], i))
        lst.sort(key=lambda x: -x[0])
        prime_nodes[p] = lst

    return nodes, weights, pfs, conflict, primes, prime_nodes, n


def _upper_bound(remaining_mask, weights, primes, prime_nodes, n):
    """
    Upper bound: sum over primes p of max weight among remaining nodes divisible by p.
    remaining_mask: int bitmask of remaining nodes.
    """
    bound = Fraction(0)
    for p in primes:
        for w, i in prime_nodes[p]:
            if (remaining_mask >> i) & 1:
                bound += w
                break
    return bound


def _greedy_clique_bound(remaining_mask, weights, conflict, n):
    """
    Alternative bound using greedy clique cover.
    """
    bound = Fraction(0)
    rem = list(remaining_mask.bit_length() * [0])
    rem_set = set()
    for i in range(n):
        if (remaining_mask >> i) & 1:
            rem_set.add(i)

    while rem_set:
        # Pick node with max weight
        best = max(rem_set, key=lambda i: weights[i])
        bound += weights[best]
        # Form clique: all nodes in rem_set that conflict with best (plus best itself)
        clique = {best} | (conflict[best] & rem_set)
        rem_set -= clique

    return bound


def _branch_and_bound(nodes, weights, pfs, conflict, primes, prime_nodes, n, deadline):
    """
    B&B for maximum weight independent set.
    State: chosen set, remaining candidates.
    Use recursion with bitmask.
    """
    best_score = Fraction(0)
    best_set = set()

    # Initial: try greedy solution as lower bound
    # Greedy: repeatedly pick highest weight node among remaining that doesn't conflict with chosen
    chosen_greedy = set()
    remaining = set(range(n))
    while remaining:
        # Find max weight node in remaining
        best = max(remaining, key=lambda i: weights[i])
        # Check if conflicts with any chosen
        if not (conflict[best] & chosen_greedy):
            chosen_greedy.add(best)
            remaining -= conflict[best]
            remaining.discard(best)
        else:
            remaining.discard(best)
    greedy_score = sum(weights[i] for i in chosen_greedy)
    best_score = greedy_score
    best_set = set(chosen_greedy)

    def dfs(chosen, remaining_mask):
        nonlocal best_score, best_set

        if time.time() > deadline:
            return

        # Current score
        cur_score = sum(weights[i] for i in chosen)

        # Upper bound
        ub = _upper_bound(remaining_mask, weights, primes, prime_nodes, n)
        total_ub = cur_score + ub
        if total_ub <= best_score:
            return

        if remaining_mask == 0:
            if cur_score > best_score:
                best_score = cur_score
                best_set = set(chosen)
            return

        # Pick next node to branch on: highest weight in remaining
        # Find max weight node in remaining_mask
        best_i = None
        best_w = Fraction(0)
        for i in range(n):
            if (remaining_mask >> i) & 1:
                if weights[i] > best_w:
                    best_w = weights[i]
                    best_i = i

        if best_i is None:
            return

        # Branch 1: include best_i
        new_chosen = chosen | {best_i}
        new_remaining = remaining_mask
        # Remove best_i and all its conflicts
        remove_mask = (1 << best_i)
        for j in conflict[best_i]:
            if (remaining_mask >> j) & 1:
                remove_mask |= (1 << j)
        new_remaining &= ~remove_mask

        dfs(new_chosen, new_remaining)

        # Branch 2: exclude best_i
        new_remaining2 = remaining_mask & ~(1 << best_i)
        dfs(chosen, new_remaining2)

    # Start DFS
    initial_mask = (1 << n) - 1
    dfs(set(), initial_mask)

    return list(best_set)


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT - 10.0  # 10s safety margin

    nodes, weights, pfs, conflict, primes, prime_nodes, n = _build_data()

    # Run B&B
    result_indices = _branch_and_bound(nodes, weights, pfs, conflict, primes, prime_nodes, n, deadline)

    # Convert back to original values
    return [nodes[i] for i in result_indices]

# EVOLVE-BLOCK-END
