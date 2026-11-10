import time
import random
from math import gcd, isqrt
from fractions import Fraction

def _get_prime_factors(n):
    """Return sorted list of unique prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def _sieve_primes(n):
    """Return list of primes < n."""
    if n <= 2:
        return []
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n, i):
                sieve[j] = False
    return [i for i in range(n) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    N = 512

    # Precompute values and prime factors
    values = [Fraction(0)] * N
    prime_factors = [[] for _ in range(N)]

    for a in range(1, N):
        values[a] = Fraction(1, N - a)
        prime_factors[a] = _get_prime_factors(a)

    primes = _sieve_primes(N)
    prime_set = set(primes)

    # All composites (non-1, non-prime) in 1..511
    composites = [a for a in range(1, N) if a != 1 and a not in prime_set]
    # Sort composites by value descending (closest to 511 first)
    composites_by_value = sorted(composites, key=lambda x: N - x)

    def greedy_init(order=None):
        """Greedy: start with all primes, add composites in given order if no conflict."""
        selected = set(primes)
        prime_to_sel = {}
        for b in selected:
            for p in prime_factors[b]:
                prime_to_sel.setdefault(p, set()).add(b)

        if order is None:
            order = composites_by_value

        for c in order:
            if c in selected:
                continue
            # Check conflicts
            conflict = False
            for p in prime_factors[c]:
                if p in prime_to_sel:
                    conflict = True
                    break
            if not conflict:
                selected.add(c)
                for p in prime_factors[c]:
                    prime_to_sel.setdefault(p, set()).add(c)

        return selected, prime_to_sel

    def get_conflict_set(c, prime_to_sel):
        """Get set of selected elements conflicting with c."""
        cs = set()
        for p in prime_factors[c]:
            if p in prime_to_sel:
                cs.update(prime_to_sel[p])
        return cs

    def apply_add_remove(selected, prime_to_sel, to_add, to_remove):
        """Apply a move."""
        for b in to_remove:
            for p in prime_factors[b]:
                if p in prime_to_sel:
                    prime_to_sel[p].discard(b)
                    if not prime_to_sel[p]:
                        del prime_to_sel[p]
        for c in to_add:
            for p in prime_factors[c]:
                prime_to_sel.setdefault(p, set()).add(c)
        selected.difference_update(to_remove)
        selected.update(to_add)

    def score_of(selected):
        return sum(values[a] for a in selected)

    def local_search_1_for_k(selected, prime_to_sel, deadline):
        """First-improvement: try adding one unselected, removing its conflicts."""
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            for c in composites_by_value:
                if c in selected:
                    continue
                conflict_set = get_conflict_set(c, prime_to_sel)
                if not conflict_set:
                    # No conflict, just add
                    gain = values[c]
                    if gain > 0:
                        apply_add_remove(selected, prime_to_sel, [c], [])
                        improved = True
                        break
                else:
                    conflict_cost = sum(values[b] for b in conflict_set)
                    gain = values[c] - conflict_cost
                    if gain > 0:
                        apply_add_remove(selected, prime_to_sel, [c], list(conflict_set))
                        improved = True
                        break
        return score_of(selected)

    def local_search_2_for_k(selected, prime_to_sel, deadline):
        """Best-improvement: try adding two unselected composites, removing conflicts."""
        # Only consider top composites for speed
        top = [c for c in composites_by_value if c not in selected][:60]
        improved = True
        while improved:
            if time.time() > deadline - 8:
                break
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None

            for i in range(len(top)):
                c1 = top[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top)):
                    c2 = top[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue

                    conflict_set = set()
                    for p in prime_factors[c1]:
                        if p in prime_to_sel:
                            conflict_set.update(prime_to_sel[p])
                    for p in prime_factors[c2]:
                        if p in prime_to_sel:
                            conflict_set.update(prime_to_sel[p])

                    # Remove c1, c2 from conflict if they were in it (they're not selected)
                    conflict_set.discard(c1)
                    conflict_set.discard(c2)

                    conflict_cost = sum(values[b] for b in conflict_set)
                    gain = values[c1] + values[c2] - conflict_cost

                    if gain > best_gain:
                        best_gain = gain
                        best_add = [c1, c2]
                        best_remove = list(conflict_set)

            if best_add is not None:
                apply_add_remove(selected, prime_to_sel, best_add, best_remove)
                improved = True

        return score_of(selected)

    def sa_1_for_1(selected, prime_to_sel, deadline, rng):
        """Simulated annealing with 1-for-1 swaps."""
        T = 0.05
        T_min = 0.001
        cooling = 0.995
        sa_deadline = deadline - 3

        while T > T_min and time.time() < sa_deadline:
            if not selected:
                break
            a_out = rng.choice(list(selected))

            # Find candidates that can replace a_out
            # A candidate a_in is valid if after removing a_out, a_in has no conflict
            # with remaining selected
            unselected = [a for a in composites_by_value if a not in selected]
            if not unselected:
                break

            a_in = rng.choice(unselected)

            # Check validity: a_in must be coprime to all selected except possibly a_out
            ok = True
            for p in prime_factors[a_in]:
                if p in prime_to_sel:
                    conflicts = prime_to_sel[p]
                    # All conflicts must be a_out only
                    for b in conflicts:
                        if b != a_out:
                            ok = False
                            break
                    if not ok:
                        break

            if not ok:
                continue

            gain = values[a_in] - values[a_out]
            # Acceptance
            if gain > 0 or rng.random() < (2.718281828 ** (gain / T)):
                apply_add_remove(selected, prime_to_sel, [a_in], [a_out])

            T *= cooling

        return score_of(selected)

    def sa_1_for_k(selected, prime_to_sel, deadline, rng):
        """SA with 1-for-k moves: add one, remove all its conflicts."""
        T = 0.05
        T_min = 0.001
        cooling = 0.995
        sa_deadline = deadline - 2

        while T > T_min and time.time() < sa_deadline:
            unselected = [a for a in composites_by_value if a not in selected]
            if not unselected:
                break

            a_in = rng.choice(unselected)
            conflict_set = get_conflict_set(a_in, prime_to_sel)

            if not conflict_set:
                gain = values[a_in]
            else:
                conflict_cost = sum(values[b] for b in conflict_set)
                gain = values[a_in] - conflict_cost

            if gain > 0 or rng.random() < (2.718281828 ** (gain / T)):
                apply_add_remove(selected, prime_to_sel, [a_in], list(conflict_set))

            T *= cooling

        return score_of(selected)

    # Main: multi-restart
    rng = random.Random(42)
    best_score = Fraction(0)
    best_selected = set()
    best_prime_to_sel = {}

    # Run multiple restarts
    num_restarts = 30
    restart_deadline = deadline - 15  # Reserve time for final polish

    for r in range(num_restarts):
        if time.time() > restart_deadline:
            break

        # Perturb the order of composites
        if r == 0:
            order = composites_by_value[:]
        else:
            order = composites_by_value[:]
            # Randomly shuffle a portion
            k = min(len(order), rng.randint(10, 50))
            for _ in range(k):
                i = rng.randint(0, len(order) - 1)
                j = rng.randint(0, len(order) - 1)
                order[i], order[j] = order[j], order[i]

        selected, prime_to_sel = greedy_init(order)
        s = local_search_1_for_k(selected, prime_to_sel, deadline)
        s = local_search_2_for_k(selected, prime_to_sel, deadline)
        s = sa_1_for_1(selected, prime_to_sel, deadline, rng)
        s = local_search_1_for_k(selected, prime_to_sel, deadline)
        s = sa_1_for_k(selected, prime_to_sel, deadline, rng)
        s = local_search_1_for_k(selected, prime_to_sel, deadline)

        if s > best_score:
            best_score = s
            best_selected = set(selected)
            best_prime_to_sel = {p: set(v) for p, v in prime_to_sel.items()}

    # Final polish on best solution
    if time.time() < deadline - 5:
        local_search_1_for_k(best_selected, best_prime_to_sel, deadline)
        local_search_2_for_k(best_selected, best_prime_to_sel, deadline)
        local_search_1_for_k(best_selected, best_prime_to_sel, deadline)

    # Final verification
    result = list(best_selected)
    for a in result:
        if a < 1 or a > 511:
            raise ValueError(f"Out of range: {a}")
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                raise ValueError(f"Not coprime: {result[i]}, {result[j]}")

    return result
