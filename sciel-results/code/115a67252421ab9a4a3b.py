"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill

    N = 256
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]
    P = len(primes)

    # prime_idx: map prime -> index
    prime_idx = {p: i for i, p in enumerate(primes)}

    # For each number 1..N, its set of prime indices (empty for 1)
    num_primes = [[] for _ in range(N + 1)]
    num_primes[1] = []
    for i in range(2, N + 1):
        x = i
        seen = set()
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                seen.add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            seen.add(x)
        num_primes[i] = [prime_idx[p] for p in seen]

    # Numbers grouped by their prime-set "signature"
    # For each distinct frozenset of prime indices, keep the max number with that signature
    from collections import defaultdict
    best_by_sig = {}
    for a in range(1, N + 1):
        sig = tuple(sorted(num_primes[a]))
        if sig not in best_by_sig or a > best_by_sig[sig]:
            best_by_sig[sig] = a

    # Also, for each single prime, we want the best number that uses ONLY that prime (a prime power)
    # But our signature approach already handles this.
    # Build candidate list: for each signature, the best number.
    # Actually, we should keep ALL numbers because sometimes a smaller number with same primes
    # doesn't help. But the max number per signature dominates.
    # However, we also need to consider that a number with signature {p} (like 256=2^8) conflicts
    # with any number having p. So keeping only max per signature is optimal.
    candidates = list(best_by_sig.values())
    candidates = [c for c in candidates if c >= 1]
    # Ensure 1 is in candidates
    if 1 not in candidates:
        candidates.append(1)

    # Sort candidates: 1 first (always includable), then others
    # We'll do B&B on the non-1 candidates, and always include 1.
    non_one = [c for c in candidates if c != 1]
    non_one.sort(reverse=True)  # larger first for better initial bound

    # For each candidate, its prime set as a bitmask (Python int)
    cand_primes_mask = []
    for c in non_one:
        mask = 0
        for pi in num_primes[c]:
            mask |= (1 << pi)
        cand_primes_mask.append(mask)

    n_cand = len(non_one)

    # Precompute conflict: for each candidate i, which candidates j conflict with i
    # conflict_mask[i] = bitmask of candidates that conflict with i
    conflict_mask = [0] * n_cand
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if cand_primes_mask[i] & cand_primes_mask[j]:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Global best solution
    best_sum = 1  # at least we can pick {1}
    best_set = [1]

    # Upper bound function: given a set of available candidate indices (as a bitmask),
    # compute an upper bound on the sum we can add.
    # Use a greedy fractional packing: repeatedly pick the candidate with max value/conflict_degree
    def upper_bound(available_mask, used_primes_mask):
        """
        available_mask: bitmask of candidate indices still available
        used_primes_mask: bitmask of prime indices already used
        Returns an upper bound on the sum of values we can add.
        """
        total = 0
        avail = available_mask
        # Greedy: pick candidates in order of value / (1 + number of conflicts with remaining)
        # But for speed, use a simpler bound: sum of all values of candidates that don't conflict with used_primes
        # and then apply a packing bound.
        
        # Simple bound: sum of values of all candidates that are compatible with used_primes
        # This is an overestimate but fast.
        s = 0
        for i in range(n_cand):
            if (avail >> i) & 1:
                if not (cand_primes_mask[i] & used_primes_mask):
                    s += non_one[i]
        return s

    # Better upper bound: use a greedy independent set heuristic
    def greedy_bound(available_mask, used_primes_mask):
        """
        Returns an upper bound using a greedy heuristic that is admissible (>= true max).
        Actually, for B&B we need an upper bound that is >= the true max.
        The sum of all compatible values is a valid upper bound.
        Let's use a tighter one: greedy maximal independent set sum.
        """
        # Greedy: repeatedly pick the highest-value candidate that doesn't conflict with chosen ones
        # This gives a lower bound, not upper. For upper bound, use the sum of all compatible.
        # Let's use: for each prime, the max value of any candidate using that prime, sum over primes.
        # This is a valid upper bound because each prime can be used at most once.
        bound = 0
        for pi in range(P):
            pbit = 1 << pi
            if used_primes_mask & pbit:
                continue
            max_val = 0
            for i in range(n_cand):
                if (available_mask >> i) & 1:
                    if cand_primes_mask[i] & pbit:
                        if non_one[i] > max_val:
                            max_val = non_one[i]
            bound += max_val
        return bound

    # Branch and bound
    # We process candidates in order of their "constraint" (most constrained first)
    # Order: candidates with more prime factors first (they conflict with more)
    order = list(range(n_cand))
    order.sort(key=lambda i: (-len(num_primes[non_one[i]]), -non_one[i]))

    # Remap: new index -> old index
    new_to_old = [order[i] for i in range(n_cand)]
    old_to_new = [0] * n_cand
    for i, old_i in enumerate(order):
        old_to_new[old_i] = i

    # Rebuild arrays in new order
    new_non_one = [non_one[old_i] for old_i in order]
    new_mask = [cand_primes_mask[old_i] for old_i in order]
    new_conflict = [0] * n_cand
    for i in range(n_cand):
        old_i = order[i]
        for j in range(n_cand):
            old_j = order[j]
            if (conflict_mask[old_i] >> old_j) & 1:
                new_conflict[i] |= (1 << j)

    # Now B&B
    # State: position in order, used_primes_mask, current_sum, current_set
    sys.setrecursionlimit(10000)

    def solve(pos, used_primes_mask, cur_sum, cur_set):
        nonlocal best_sum, best_set
        if time.time() > deadline:
            return

        if pos == n_cand:
            if cur_sum + 1 > best_sum:  # +1 for the number 1
                best_sum = cur_sum + 1
                best_set = [1] + list(cur_set)
            return

        # Upper bound: current sum + greedy bound on remaining
        # Available: all candidates from pos onwards
        avail_mask = ((1 << n_cand) - 1) & (~((1 << pos) - 1))
        ub = cur_sum + 1 + greedy_bound(avail_mask, used_primes_mask)
        if ub <= best_sum:
            return

        # Try including candidate pos
        i = pos
        if not (new_mask[i] & used_primes_mask):
            new_used = used_primes_mask | new_mask[i]
            new_sum = cur_sum + new_non_one[i]
            new_set = cur_set + [new_non_one[i]]
            # Check if any future candidate conflicts
            # Actually, we just need to ensure future candidates don't share primes with new_used
            # This is handled by the used_primes_mask check in recursion
            solve(pos + 1, new_used, new_sum, new_set)

        # Try excluding candidate pos
        solve(pos + 1, used_primes_mask, cur_sum, cur_set)

    # Initial call
    solve(0, 0, 0, [])

    # Always include 1
    result = sorted(best_set)
    # Ensure all are Python ints
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
