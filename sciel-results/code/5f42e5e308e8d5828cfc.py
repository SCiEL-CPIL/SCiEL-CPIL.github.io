import math
import time
import random

def construct():
    start = time.time()
    deadline = start + 82  # safety margin under 90s

    N = 256

    # ---- Sieve primes ----
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    primes = sieve(N)
    prime_set = set(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # ---- Largest prime power <= N for each prime ----
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # ---- Prime factorization (frozenset of distinct primes) for 1..N ----
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

    # ---- For each prime-subset, the best composite using exactly those primes ----
    # A composite c with factor set ps beats the base (sum of prime powers) iff
    # gain = c - sum(prime_powers[p] for p in ps) > 0.
    best_for_subset = {}  # frozenset(primes) -> (value, gain)
    for c in range(2, N + 1):
        if c in prime_set:
            continue
        ps = F[c]
        if len(ps) < 2:
            continue
        if len(ps) > 5:
            continue
        base_cost = sum(prime_powers[p] for p in ps)
        gain = c - base_cost
        if gain > 0:
            prev = best_for_subset.get(ps)
            if prev is None or c > prev[0]:
                best_for_subset[ps] = (c, gain)

    # ---- Base solution: 1 + all prime powers ----
    base_numbers = [1] + [prime_powers[p] for p in primes]
    base_sum = sum(base_numbers)

    # ---- Build move list (mask, value, gain) ----
    def factor_set_to_mask(ps):
        mask = 0
        for p in ps:
            mask |= 1 << prime_index[p]
        return mask

    moves = []
    for ps, (val, gain) in best_for_subset.items():
        mask = factor_set_to_mask(ps)
        moves.append((mask, val, gain))
    moves.sort(key=lambda x: -x[2])
    num_moves = len(moves)

    # Upper bound on total gain for a set of moves (ignores disjointness)
    total_gain = sum(m[2] for m in moves)

    # ---- Branch-and-bound to find the OPTIMAL disjoint set of moves ----
    # State: (free_mask, idx). Recurse with memoized upper-bound to prune.
    memo = {}
    best = {'sum': 0, 'sel': []}
    cur = []

    def bnb(free_mask, idx, cur_sum):
        if cur_sum + total_gain <= best['sum']:
            return
        if time.time() > deadline - 6:
            return
        if idx >= num_moves:
            if cur_sum > best['sum']:
                best['sum'] = cur_sum
                best['sel'] = cur[:]
            return

        # memoized upper bound on achievable gain from idx..end given free_mask
        key = (free_mask, idx)
        ub = memo.get(key)
        if ub is None:
            if idx >= num_moves:
                ub = 0
            else:
                mask, val, gain = moves[idx]
                if mask & free_mask == 0:
                    ub = max(gain + bnb(free_mask ^ mask, idx + 1, 0),
                             bnb(free_mask, idx + 1, 0))
                else:
                    ub = bnb(free_mask, idx + 1, 0)
            memo[key] = ub
        if cur_sum + ub <= best['sum']:
            return

        mask, val, gain = moves[idx]
        if mask & free_mask == 0:
            # take
            cur.append((mask, val, gain))
            bnb(free_mask ^ mask, idx + 1, cur_sum + gain)
            cur.pop()
        # skip
        bnb(free_mask, idx + 1, cur_sum)

    full_mask = (1 << num_primes) - 1
    bnb(full_mask, 0, 0)

    # ---- Build solution from selected moves ----
    def build_solution(selected_moves):
        result = [1]
        used_mask = 0
        for mask, val, gain in selected_moves:
            result.append(val)
            used_mask |= mask
        for i, p in enumerate(primes):
            if not (used_mask & (1 << i)):
                result.append(prime_powers[p])
        return result

    selected = best['sel'][:]
    best_sol = build_solution(selected)
    best_sum = sum(best_sol)
    if best_sum < base_sum:
        best_sol = base_numbers[:]
        best_sum = base_sum

    # ---- Fast direct-replacement polish (structural: per-element max search) ----
    def improve_by_replacement(sol):
        improved = True
        max_iter = 40
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            if time.time() > deadline - 3:
                break
            # precompute factor sets
            fs = [F[x] for x in sol]
            for i in range(len(sol)):
                if time.time() > deadline - 3:
                    break
                if sol[i] == 1:
                    continue
                rest = set()
                for j in range(len(sol)):
                    if j != i:
                        rest |= fs[j]
                best_repl = None
                best_repl_val = sol[i]
                for c in range(2, N + 1):
                    if F[c] & rest:
                        continue
                    if c > best_repl_val:
                        best_repl_val = c
                        best_repl = c
                if best_repl is not None and best_repl > sol[i]:
                    sol[i] = best_repl
                    fs[i] = F[best_repl]
                    improved = True
        return sol

    best_sol = improve_by_replacement(best_sol)
    best_sum = sum(best_sol)
    if time.time() < deadline - 5:
        best_sol = improve_by_replacement(best_sol)
        best_sum = sum(best_sol)

    # ---- Verify pairwise coprimality & validity ----
    def verify(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    if not verify(best_sol):
        fb = build_solution(selected)
        best_sol = fb if verify(fb) else base_numbers[:]
        best_sum = sum(best_sol)

    # ---- Finalize: distinct, in range, pure Python ints ----
    best_sol = list({x for x in best_sol if 1 <= x <= 256})
    best_sol = [int(x) for x in best_sol]
    if not best_sol:
        best_sol = [1]
    return best_sol
