"""Covering 1..L by prime residue classes with reciprocal budget 1 (SA + local search)."""
import time
import random
from fractions import Fraction

MAXL = 100000
DEADLINE = 140.0  # safety margin before the 150s kill


def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = 0
    return [i for i in range(2, n + 1) if sieve[i]]


def _covered_prefix_len(covered):
    """Return the largest L such that covered[1..L] are all True."""
    L = 0
    for i in range(1, MAXL + 1):
        if covered[i]:
            L = i
        else:
            break
    return L


def _recip_sum(prime_set):
    """Return the sum of 1/p for p in prime_set as a Fraction."""
    total = Fraction(0, 1)
    for p in prime_set:
        total += Fraction(1, p)
    return total


def construct():
    t0 = time.time()
    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}

    # ---- Fast incremental coverage via per-position count array ----
    # cnt[x] = number of classes covering x; covered[x] = cnt[x] > 0.
    # add/remove are O(MAXL/p) each.

    def add_class(cnt, covered, p, r):
        start = r if r >= 1 else p
        for x in range(start, MAXL + 1, p):
            if cnt[x] == 0:
                covered[x] = 1
            cnt[x] += 1

    def remove_class(cnt, covered, p, r):
        start = r if r >= 1 else p
        for x in range(start, MAXL + 1, p):
            cnt[x] -= 1
            if cnt[x] == 0:
                covered[x] = 0

    def swap_residue(cnt, covered, p, old_r, new_r):
        remove_class(cnt, covered, p, old_r)
        add_class(cnt, covered, p, new_r)

    # ---- Greedy initialization ----
    def greedy_init(seed):
        rng = random.Random(seed)
        state = {}
        budget = Fraction(1, 1)
        cnt = [0] * (MAXL + 1)
        covered = bytearray(MAXL + 1)
        L = 0
        remaining = primes[:]
        rng.shuffle(remaining)

        for _ in range(len(primes)):
            if time.time() > DEADLINE - 30:
                break
            best_gain = 0
            best_p = best_r = None
            best_newL = 0

            for p in remaining:
                if p in state:
                    continue
                if budget < recip[p]:
                    continue
                # Try residue covering L+1
                r = (L + 1) % p
                add_class(cnt, covered, p, r)
                newL = 0
                for i in range(L + 1, MAXL + 1):
                    if covered[i]:
                        newL = i
                    else:
                        break
                gain = newL - L
                remove_class(cnt, covered, p, r)

                if gain > best_gain:
                    best_gain = gain
                    best_p, best_r = p, r
                    best_newL = newL

            if best_p is None or best_gain == 0:
                break

            add_class(cnt, covered, best_p, best_r)
            state[best_p] = best_r
            budget -= recip[best_p]
            L = best_newL

        return state, budget, L

    # ---- Simulated annealing refinement ----
    def simulated_annealing(state, budget, cnt, covered, L, rng, max_iters=50000):
        """SA to improve L by exploring residue changes and prime swaps."""
        best_state = dict(state)
        best_budget = budget
        best_L = L
        best_cnt = cnt[:]
        best_covered = covered[:]

        temp = 10.0
        cooling = 0.9999
        min_temp = 0.1

        for it in range(max_iters):
            if time.time() > DEADLINE - 10:
                break
            if temp < min_temp:
                break

            # Choose a random move
            move_type = rng.random()
            if move_type < 0.6:
                # Change residue of an existing prime
                if not state:
                    continue
                p = rng.choice(list(state.keys()))
                old_r = state[p]
                new_r = rng.randrange(p)
                if new_r == old_r:
                    continue
                swap_residue(cnt, covered, p, old_r, new_r)
                state[p] = new_r
                new_L = _covered_prefix_len(covered)
                delta = new_L - L
                # Accept if better or with probability exp(delta/temp)
                if delta >= 0 or rng.random() < (temp / max(1, abs(delta)) if delta < 0 else 1):
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_state = dict(state)
                        best_budget = budget
                        best_cnt = cnt[:]
                        best_covered = covered[:]
                else:
                    swap_residue(cnt, covered, p, new_r, old_r)
                    state[p] = old_r
            elif move_type < 0.8:
                # Add a new prime if budget allows
                candidates = [p for p in primes if p not in state and budget >= recip[p]]
                if not candidates:
                    continue
                p = rng.choice(candidates)
                r = rng.randrange(p)
                add_class(cnt, covered, p, r)
                state[p] = r
                budget -= recip[p]
                new_L = _covered_prefix_len(covered)
                delta = new_L - L
                if delta >= 0 or rng.random() < (temp / max(1, abs(delta)) if delta < 0 else 1):
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_state = dict(state)
                        best_budget = budget
                        best_cnt = cnt[:]
                        best_covered = covered[:]
                else:
                    remove_class(cnt, covered, p, r)
                    del state[p]
                    budget += recip[p]
            else:
                # Remove a prime to free budget
                if not state:
                    continue
                p = rng.choice(list(state.keys()))
                r = state[p]
                remove_class(cnt, covered, p, r)
                del state[p]
                budget += recip[p]
                new_L = _covered_prefix_len(covered)
                delta = new_L - L
                if delta >= 0 or rng.random() < (temp / max(1, abs(delta)) if delta < 0 else 1):
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_state = dict(state)
                        best_budget = budget
                        best_cnt = cnt[:]
                        best_covered = covered[:]
                else:
                    add_class(cnt, covered, p, r)
                    state[p] = r
                    budget -= recip[p]

            temp *= cooling

        # Restore best state
        state.clear()
        state.update(best_state)
        budget = best_budget
        cnt[:] = best_cnt
        covered[:] = best_covered
        L = best_L

        return state, budget, L

    # ---- Run greedy with multiple seeds, keep best ----
    best_state = None
    best_budget = Fraction(1, 1)
    best_L = 0

    for seed in range(8):
        if time.time() > DEADLINE - 40:
            break
        state, budget, L = greedy_init(seed)
        if L > best_L:
            best_L = L
            best_state = state
            best_budget = budget

    if best_state is None:
        # Fallback
        primes13 = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes13)]

    # ---- SA refinement from the best greedy solution ----
    state = dict(best_state)
    budget = best_budget
    cnt = [0] * (MAXL + 1)
    covered = bytearray(MAXL + 1)
    for p, r in state.items():
        add_class(cnt, covered, p, r)
    L = _covered_prefix_len(covered)

    # Run SA with a few different randomizations
    for trial in range(4):
        if time.time() > DEADLINE - 15:
            break

        rng = random.Random(100 + trial * 7)
        s, b, L_new = simulated_annealing(state, budget, cnt, covered, L, rng, max_iters=20000)

        if L_new > best_L:
            best_state = s
            best_budget = b
            best_L = L_new
            state = dict(s)
            budget = b
            cnt = [0] * (MAXL + 1)
            covered = bytearray(MAXL + 1)
            for p, r in state.items():
                add_class(cnt, covered, p, r)

    # ---- Final verification and return ----
    result = [[p, r] for p, r in best_state.items()]

    # Verify budget constraint
    total = Fraction(0, 1)
    for p, _ in result:
        total += Fraction(1, p)
    if total > Fraction(1, 1):
        # Shouldn't happen, but fallback
        primes13 = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes13)]

    return result
