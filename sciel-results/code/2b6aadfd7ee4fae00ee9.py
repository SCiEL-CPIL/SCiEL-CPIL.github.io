import math
import time
import random

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

    # Base solution: 1 + all prime powers
    base_numbers = [1] + [prime_powers[p] for p in primes]
    base_sum = sum(base_numbers)

    # Generate candidate moves
    # A move is defined by a set of primes P and a composite c <= N
    # such that prime_factors(c) == P.
    # We want to maximize c - sum(prime_powers[p] for p in P).
    
    moves = []
    
    # Helper to get bitmask for a set of primes
    def get_mask(ps):
        mask = 0
        for p in ps:
            mask |= (1 << prime_index[p])
        return mask

    # We can iterate over all c from 2 to N
    # and group by their factor set.
    best_c_for_ps = {}
    
    for c in range(2, N + 1):
        if c in prime_set:
            continue
        ps = F[c]
        if len(ps) < 2:
            continue
        # Check if c is the max for this ps
        if ps not in best_c_for_ps or c > best_c_for_ps[ps]:
            best_c_for_ps[ps] = c

    for ps, c in best_c_for_ps.items():
        base_cost = sum(prime_powers[p] for p in ps)
        gain = c - base_cost
        if gain > 0:
            mask = get_mask(ps)
            moves.append((mask, c, gain, ps))

    # Sort moves by gain descending for branch and bound efficiency
    moves.sort(key=lambda x: -x[2])
    
    # Precompute suffix sums of gains for pruning
    n_moves = len(moves)
    suffix_gains = [0] * (n_moves + 1)
    for i in range(n_moves - 1, -1, -1):
        suffix_gains[i] = suffix_gains[i+1] + moves[i][2]

    best_total_gain = 0
    best_selected_moves = []
    
    # Branch and Bound
    # State: idx, used_mask, current_gain, selected
    def solve(idx, used_mask, current_gain, selected):
        nonlocal best_total_gain, best_selected_moves
        
        if time.time() > deadline - 1:
            return

        # Pruning
        if current_gain + suffix_gains[idx] <= best_total_gain:
            return

        if idx == n_moves:
            if current_gain > best_total_gain:
                best_total_gain = current_gain
                best_selected_moves = selected[:]
            return

        # Option 1: Skip move idx
        solve(idx + 1, used_mask, current_gain, selected)

        # Option 2: Take move idx
        mask, c, gain, ps = moves[idx]
        if mask & used_mask == 0:
            selected.append((mask, c, gain, ps))
            solve(idx + 1, used_mask | mask, current_gain + gain, selected)
            selected.pop()

    solve(0, 0, 0, [])

    # Construct final solution from B&B result
    final_sol = set(base_numbers)
    
    # Remove prime powers for primes used in selected moves
    used_primes_in_moves = set()
    for mask, c, gain, ps in best_selected_moves:
        for p in ps:
            used_primes_in_moves.add(p)
            final_sol.remove(prime_powers[p])
        
    # Add composites
    for mask, c, gain, ps in best_selected_moves:
        final_sol.add(c)
        
    # Convert to list of ints
    result = [int(x) for x in final_sol]
    
    # Verification
    def verify(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    if not verify(result):
        # Fallback to base if something went wrong (should not happen)
        result = [int(x) for x in base_numbers]
    
    # Now try to improve with local search
    # We'll try swapping out one number for another to see if we can gain
    def try_improve(sol):
        sol_set = set(sol)
        sol_list = list(sol)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i, num in enumerate(sol_list):
                # Try removing num and adding a different number
                # We need to find a number not in sol_set that is coprime to all others in sol_set (except num)
                # Actually, we need to find a number x such that:
                # 1. x not in sol_set
                # 2. gcd(x, y) == 1 for all y in sol_set - {num}
                # 3. x > num (to improve sum)
                
                # This is expensive. Let's try a simpler approach:
                # For each num in sol, try to replace it with a larger number that is coprime to all others.
                
                others = [y for y in sol_list if y != num]
                
                # Try numbers from 256 down to num+1
                for x in range(256, num, -1):
                    if x in sol_set:
                        continue
                    # Check if x is coprime to all others
                    ok = True
                    for y in others:
                        if math.gcd(x, y) != 1:
                            ok = False
                            break
                    if ok:
                        # Replace num with x
                        sol_set.remove(num)
                        sol_set.add(x)
                        sol_list.remove(num)
                        sol_list.append(x)
                        improved = True
                        break
        return sol_list

    # Apply local search
    result = try_improve(result)
    
    # Verify again
    if not verify(result):
        # If local search broke something, fall back to B&B result
        final_sol = set(base_numbers)
        used_primes_in_moves = set()
        for mask, c, gain, ps in best_selected_moves:
            for p in ps:
                used_primes_in_moves.add(p)
                final_sol.remove(prime_powers[p])
        for mask, c, gain, ps in best_selected_moves:
            final_sol.add(c)
        result = [int(x) for x in final_sol]
    
    return result
