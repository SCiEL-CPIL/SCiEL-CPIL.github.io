import math
import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85  # safety margin under 90s

    N = 256

    # Sieve for primes
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    primes = sieve(N)
    prime_set = set(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Prime powers: largest p^k <= N for each prime p
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Precompute prime factorization (as frozenset of distinct primes) for all numbers 1..N
    F = [None] * (N + 1)
    F[1] = frozenset()
    for i in range(2, N + 1):
        x = i
        factors = set()
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.add(d)
                while x % d == 0:
                    x //= d
            else:
                d += 1
        if x > 1:
            factors.add(x)
        F[i] = frozenset(factors)

    # Build list of "items": each item is a number with its prime-factor set.
    # We want to select a subset of numbers maximizing sum, with disjoint prime factors.
    # Base items: 1 (empty factor set) and each prime power.
    # Alternative items: composites that may replace multiple prime powers.

    # For each composite c (non-prime, non-prime-power), it has factor set F[c] with >= 2 primes.
    # It can replace the prime powers of exactly those primes.
    # Gain = c - sum(prime_powers[p] for p in F[c])

    # Collect all candidate "moves": a move is (factor_set, value, gain)
    # where factor_set is the set of primes used, value is the number, gain is value - base_cost.

    # Base solution: 1 + all prime powers
    base_numbers = [1] + [prime_powers[p] for p in primes]
    base_sum = sum(base_numbers)

    # Build all composites with 2-4 distinct prime factors
    composite_candidates = {}  # frozenset -> list of (value, gain)
    for c in range(2, N + 1):
        if c in prime_set:
            continue
        ps = F[c]
        if len(ps) < 2 or len(ps) > 4:
            continue
        base_cost = sum(prime_powers[p] for p in ps)
        gain = c - base_cost
        if gain > 0:
            if ps not in composite_candidates:
                composite_candidates[ps] = []
            composite_candidates[ps].append((c, gain))

    # For each factor set, keep only the best value (highest gain)
    best_composite = {}  # frozenset -> (value, gain)
    for ps, vals in composite_candidates.items():
        best_val = max(vals, key=lambda x: x[0])
        best_composite[ps] = best_val

    # Now we have a set packing problem:
    # Select a set of disjoint prime subsets S1, S2, ... to maximize total gain.
    # Each subset Si has a gain g(Si) = best_composite[Si][1].
    # Primes not in any selected subset use their prime power.

    # Since we have 54 primes and subsets of size 2-4, this is a weighted set packing.
    # Use greedy + local search with bitmask for efficiency.

    # Map primes to bit positions
    prime_to_bit = {p: (1 << i) for i, p in enumerate(primes)}

    # Convert factor sets to bitmasks
    def factor_set_to_mask(ps):
        mask = 0
        for p in ps:
            mask |= prime_to_bit[p]
        return mask

    # List of (mask, value, gain) for all positive-gain composites
    moves = []
    for ps, (val, gain) in best_composite.items():
        mask = factor_set_to_mask(ps)
        moves.append((mask, val, gain))

    # Sort by gain descending
    moves.sort(key=lambda x: -x[2])

    # Greedy initialization
    def greedy_select(moves, all_prime_mask):
        selected = []
        used_mask = 0
        for mask, val, gain in moves:
            if mask & used_mask == 0:
                selected.append((mask, val, gain))
                used_mask |= mask
        return selected, used_mask

    # Build solution from selected moves
    def build_solution(selected_moves, all_primes, prime_powers):
        result = [1]
        used_mask = 0
        for mask, val, gain in selected_moves:
            result.append(val)
            used_mask |= mask
        for p in all_primes:
            bit = prime_to_bit[p]
            if not (used_mask & bit):
                result.append(prime_powers[p])
        return result

    # Try multiple greedy orderings
    best_sum = base_sum
    best_sol = base_numbers[:]
    best_selected = []
    best_used_mask = 0

    # Order 1: by gain descending
    sel1, mask1 = greedy_select(moves, 0)
    sol1 = build_solution(sel1, primes, prime_powers)
    s1 = sum(sol1)
    if s1 > best_sum:
        best_sum = s1
        best_sol = sol1[:]
        best_selected = sel1[:]
        best_used_mask = mask1

    # Order 2: by gain/size ratio
    moves_by_ratio = sorted(moves, key=lambda x: -x[2] / x[0].bit_count())
    sel2, mask2 = greedy_select(moves_by_ratio, 0)
    sol2 = build_solution(sel2, primes, prime_powers)
    s2 = sum(sol2)
    if s2 > best_sum:
        best_sum = s2
        best_sol = sol2[:]
        best_selected = sel2[:]
        best_used_mask = mask2

    # Order 3: random shuffles
    import random
    rng = random.Random(42)
    for trial in range(20):
        shuffled = moves[:]
        rng.shuffle(shuffled)
        sel_t, mask_t = greedy_select(shuffled, 0)
        sol_t = build_solution(sel_t, primes, prime_powers)
        st = sum(sol_t)
        if st > best_sum:
            best_sum = st
            best_sol = sol_t[:]
            best_selected = sel_t[:]
            best_used_mask = mask_t
        if time.time() > deadline - 10:
            break

    # Local search: 1-for-1 swaps
    def local_search(selected, used_mask, current_sum):
        improved = True
        max_iter = 200
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            if time.time() > deadline - 5:
                break

            # Try removing one selected move and adding another
            for i in range(len(selected)):
                if time.time() > deadline - 5:
                    break
                rem_mask, rem_val, rem_gain = selected[i]
                rem_used = used_mask ^ rem_mask  # remove from used

                # Try adding each move not currently selected
                for j in range(len(moves)):
                    m_mask, m_val, m_gain = moves[j]
                    if m_mask & rem_used:
                        continue
                    # Check it's not the same as removed
                    if m_mask == rem_mask and m_val == rem_val:
                        continue
                    new_sum = current_sum - rem_val + m_val
                    if new_sum > current_sum:
                        # Apply the swap
                        selected[i] = (m_mask, m_val, m_gain)
                        used_mask = rem_used | m_mask
                        current_sum = new_sum
                        improved = True
                        break

            # Try adding a new move (if there's an unused prime)
            if not improved:
                for j in range(len(moves)):
                    m_mask, m_val, m_gain = moves[j]
                    if m_mask & used_mask:
                        continue
                    # Check not already selected
                    already = False
                    for sm, sv, sg in selected:
                        if sm == m_mask and sv == m_val:
                            already = True
                            break
                    if already:
                        continue
                    new_sum = current_sum + m_val
                    if new_sum > current_sum:
                        selected.append((m_mask, m_val, m_gain))
                        used_mask |= m_mask
                        current_sum = new_sum
                        improved = True
                        break

            # Try removing a move if it has negative net effect (shouldn't happen since gain>0)
            # But removing frees primes for potential better composites - handled by swaps above

        return selected, used_mask, current_sum

    best_selected, best_used_mask, best_sum = local_search(best_selected, best_used_mask, best_sum)

    # Rebuild solution
    best_sol = build_solution(best_selected, primes, prime_powers)
    best_sum = sum(best_sol)

    # Additional local search: try individual number replacements
    # For each number in the solution, try replacing it with a larger number that doesn't conflict
    def improve_by_replacement(sol):
        improved = True
        max_iter = 50
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            if time.time() > deadline - 3:
                break

            for i in range(len(sol)):
                if time.time() > deadline - 3:
                    break
                if sol[i] == 1:
                    continue
                # Remove sol[i], find best replacement
                rest = [x for j, x in enumerate(sol) if j != i]
                rest_factors = set()
                for x in rest:
                    rest_factors |= F[x]

                best_repl = None
                best_repl_val = 0
                for c in range(2, N + 1):
                    if F[c] & rest_factors:
                        continue
                    if c > best_repl_val:
                        best_repl_val = c
                        best_repl = c

                if best_repl is not None and best_repl > sol[i]:
                    sol[i] = best_repl
                    improved = True

        return sol

    best_sol = improve_by_replacement(best_sol)
    best_sum = sum(best_sol)

    # Verify pairwise coprimality
    def verify(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    if not verify(best_sol):
        # Fallback: rebuild from selected moves + prime powers
        fallback = build_solution(best_selected, primes, prime_powers)
        if not verify(fallback):
            best_sol = base_numbers[:]
        else:
            best_sol = fallback

    # Ensure distinct and in range
    best_sol = list(set(x for x in best_sol if 1 <= x <= 256))

    # Final type check: all must be Python int
    best_sol = [int(x) for x in best_sol]

    return best_sol
