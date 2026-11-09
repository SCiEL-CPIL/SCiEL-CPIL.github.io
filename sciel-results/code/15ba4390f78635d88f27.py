"""Covering 1..L by prime residue classes with reciprocal budget 1.

Structural change: reformulate as a gap-filling CSP.
State is the set of uncovered integers in 1..L (maintained via a coverage-count
array). Moves are: (a) pick a gap x, (b) choose a prime p (used or unused) and
residue r = x % p that maximizes the number of gaps it fills, subject to the
reciprocal budget. This is the "gap-filling" model from the directive.

The refinement machinery (simulated annealing with incremental updates,
recompute-on-remove, best-state tracking) is retained from the current program,
but the move generation is now gap-driven rather than prime-driven.
"""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
import sys

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin before 150s kill
    MAXL = 100000
    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}

    # ---- State representation ----
    # cover_count[x] = number of classes covering x (for x in 1..MAXL)
    # state: dict p -> r
    # budget: Fraction remaining
    # L: current covered prefix length

    def build_initial(seed):
        """Greedy gap-filling initial construction."""
        random.seed(seed)
        cover_count = [0] * (MAXL + 1)
        state = {}
        budget = Fraction(1, 1)
        L = 0

        for _ in range(len(primes)):
            if time.time() > DEADLINE:
                break
            # Find current L
            while L < MAXL and cover_count[L + 1] > 0:
                L += 1
            if L >= MAXL:
                break
            # Gap at L+1 (first uncovered)
            gap = L + 1
            # Find best (p, r) that covers this gap
            best_gain = 0
            best_p = None
            best_r = None
            for p in primes:
                if p in state:
                    continue
                if budget < recip[p]:
                    continue
                r = gap % p
                # Count how many of the next K positions would be newly covered
                K = min(3000, MAXL - gap + 1)
                gain = 0
                for x in range(gap, min(gap + K, MAXL) + 1):
                    if cover_count[x] == 0 and (x - r) % p == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            if best_p is None or best_gain == 0:
                break
            p, r = best_p, best_r
            state[p] = r
            budget -= recip[p]
            for x in range(r if r >= 1 else p, MAXL + 1, p):
                cover_count[x] += 1
            # Advance L
            while L < MAXL and cover_count[L + 1] > 0:
                L += 1

        return state, cover_count, L, budget

    # Try several seeds, keep best
    best_state = None
    best_cover = None
    best_L = 0
    best_budget = Fraction(1, 1)
    for seed in range(16):
        if time.time() > DEADLINE:
            break
        state, cover, L, budget = build_initial(seed)
        if L > best_L:
            best_L = L
            best_state = state
            best_cover = cover
            best_budget = budget

    if best_state is None:
        primes13 = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes13)]

    # ---- Simulated annealing with gap-filling moves ----
    state = dict(best_state)
    cover_count = list(best_cover)
    L = best_L
    budget = best_budget
    best_state_sa = dict(state)
    best_L_sa = L

    def recompute_full():
        nonlocal cover_count, L
        cover_count = [0] * (MAXL + 1)
        for p, r in state.items():
            for x in range(r if r >= 1 else p, MAXL + 1, p):
                cover_count[x] += 1
        L = 0
        for i in range(1, MAXL + 1):
            if cover_count[i] > 0:
                L = i
            else:
                break

    def add_prime(p, r):
        nonlocal cover_count, L, budget
        if p in state:
            return False
        if budget < recip[p]:
            return False
        for x in range(r if r >= 1 else p, MAXL + 1, p):
            cover_count[x] += 1
        state[p] = r
        budget -= recip[p]
        newL = L
        for i in range(L + 1, MAXL + 1):
            if cover_count[i] > 0:
                newL = i
            else:
                break
        L = newL
        return True

    def remove_prime(p):
        nonlocal cover_count, L, budget
        if p not in state:
            return False
        r = state[p]
        for x in range(r if r >= 1 else p, MAXL + 1, p):
            cover_count[x] -= 1
        del state[p]
        budget += recip[p]
        recompute_full()
        return True

    def change_residue(p, new_r):
        nonlocal cover_count, L
        if p not in state:
            return False
        old_r = state[p]
        if old_r == new_r:
            return False
        # Decrement old
        for x in range(old_r if old_r >= 1 else p, MAXL + 1, p):
            cover_count[x] -= 1
        # Increment new
        for x in range(new_r if new_r >= 1 else p, MAXL + 1, p):
            cover_count[x] += 1
        state[p] = new_r
        # Recompute L
        L = 0
        for i in range(1, MAXL + 1):
            if cover_count[i] > 0:
                L = i
            else:
                break
        return True

    def gap_positions(limit=3000):
        gaps = []
        for i in range(1, MAXL + 1):
            if cover_count[i] == 0:
                gaps.append(i)
                if len(gaps) >= limit:
                    break
        return gaps

    rng = random.Random(98765)
    T = 5.0
    T_min = 0.01
    alpha = 0.9997

    iter_count = 0
    while time.time() < DEADLINE:
        T = max(T_min, T * alpha)
        move_type = rng.random()

        if move_type < 0.50:
            # Gap-filling: pick a gap, choose prime+residue to cover it
            gaps = gap_positions(2000)
            if gaps:
                g = rng.choice(gaps[:min(len(gaps), 1000)])
                # Collect candidate primes (used and unused)
                candidates = []
                for p in primes:
                    if p in state:
                        # Could change residue
                        candidates.append((p, 'change'))
                    else:
                        if budget >= recip[p]:
                            candidates.append((p, 'add'))
                if not candidates:
                    continue
                # Score each candidate by how many gaps it fills
                best_score = -1
                best_p = None
                best_r = None
                best_action = None
                # Sample a subset if too many
                if len(candidates) > 60:
                    candidates = rng.sample(candidates, 60)
                for p, action in candidates:
                    r = g % p
                    # Estimate gain: count newly covered in a window
                    window = min(2000, MAXL - g + 1)
                    gain = 0
                    if action == 'add':
                        for x in range(g, g + window):
                            if cover_count[x] == 0 and (x - r) % p == 0:
                                gain += 1
                    else:
                        old_r = state[p]
                        if old_r == r:
                            continue
                        for x in range(g, g + window):
                            if cover_count[x] == 0 and (x - r) % p == 0:
                                gain += 1
                    if gain > best_score:
                        best_score = gain
                        best_p = p
                        best_r = r
                        best_action = action
                if best_p is None or best_score <= 0:
                    continue
                old_L = L
                if best_action == 'add':
                    if add_prime(best_p, best_r):
                        delta = L - old_L
                        if delta < 0 and rng.random() > (0.1 ** (abs(delta) / 100 + 1)):
                            remove_prime(best_p)
                else:
                    old_r = state[best_p]
                    change_residue(best_p, best_r)
                    delta = L - old_L
                    if delta < 0 and rng.random() > (0.1 ** (abs(delta) / 100 + 1)):
                        change_residue(best_p, old_r)

        elif move_type < 0.75:
            # Random residue change on a used prime
            if state:
                p = rng.choice(list(state.keys()))
                old_r = state[p]
                new_r = rng.randrange(p)
                old_L = L
                change_residue(p, new_r)
                delta = L - old_L
                if delta < 0 and rng.random() > (0.1 ** (abs(delta) / 50 + 1)):
                    change_residue(p, old_r)

        elif move_type < 0.90:
            # Add a random prime with random residue
            unused = [p for p in primes if p not in state and budget >= recip[p]]
            if unused:
                p = rng.choice(unused)
                r = rng.randrange(p)
                old_L = L
                if add_prime(p, r):
                    delta = L - old_L
                    if delta < 0 and rng.random() > 0.5:
                        remove_prime(p)

        else:
            # Remove a prime (free budget)
            if len(state) > 5:
                p = rng.choice(list(state.keys()))
                remove_prime(p)

        iter_count += 1
        if L > best_L_sa:
            best_L_sa = L
            best_state_sa = dict(state)
            if rng.random() < 0.005:
                state.clear()
                state.update(best_state_sa)
                recompute_full()

    # Return best found
    if best_L_sa > best_L:
        result = [[p, r] for p, r in best_state_sa.items()]
        return result
    else:
        result = [[p, r] for p, r in best_state.items()]
        return result
# EVOLVE-BLOCK-END
