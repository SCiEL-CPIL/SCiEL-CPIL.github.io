"""Covering 1..L by prime residue classes with reciprocal budget 1 (primes <= 1000)."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

MAXL = 100000
DEADLINE = 132.0  # safety margin before the 150s kill

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = 0
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    t0 = time.time()
    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}

    # ---- Fast incremental coverage via per-position count array ----
    # cnt[x] = number of classes covering x; covered[x] = cnt[x] > 0.
    # add/remove/residue-change are O(p) each, so we can do many moves.
    def make_covered(state):
        cnt = [0] * (MAXL + 1)
        for p, r in state.items():
            for x in range(r if r >= 1 else p, MAXL + 1, p):
                cnt[x] += 1
        covered = bytearray(MAXL + 1)
        for x in range(1, MAXL + 1):
            if cnt[x]:
                covered[x] = 1
        return cnt, covered

    def compute_L(covered):
        L = 0
        for i in range(1, MAXL + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def add_class(cnt, covered, p, r):
        for x in range(r if r >= 1 else p, MAXL + 1, p):
            if cnt[x] == 0:
                covered[x] = 1
            cnt[x] += 1

    def remove_class(cnt, covered, p, r):
        for x in range(r if r >= 1 else p, MAXL + 1, p):
            cnt[x] -= 1
            if cnt[x] == 0:
                covered[x] = 0

    def swap_residue(cnt, covered, p, old_r, new_r):
        # remove old, add new (order-safe since both are the same prime)
        remove_class(cnt, covered, p, old_r)
        add_class(cnt, covered, p, new_r)

    # ---- Greedy initialization (multi-seed) ----
    def greedy_init(seed):
        rng = random.Random(seed)
        state = {}
        budget = Fraction(1, 1)
        cnt, covered = [0] * (MAXL + 1), bytearray(MAXL + 1)
        L = 0
        remaining = primes[:]
        rng.shuffle(remaining)
        # repeatedly add the (p,r) with best marginal prefix extension
        for _ in range(len(primes)):
            if time.time() > DEADLINE - 20:
                break
            best_gain = 0
            best_p = best_r = None
            for p in remaining:
                if p in state:
                    continue
                if budget < recip[p]:
                    continue
                # try a few candidate residues, prefer one covering L+1
                cands = [(L + 1) % p]
                # also try residues that cover near-current gaps (cheap sample)
                for _c in range(2):
                    cands.append(rng.randrange(p))
                seen = set()
                for r in cands:
                    if r in seen:
                        continue
                    seen.add(r)
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

    best_state = None
    best_budget = Fraction(1, 1)
    best_L = 0
    for seed in range(8):
        if time.time() > DEADLINE - 25:
            break
        state, budget, L = greedy_init(seed)
        if L > best_L:
            best_L = L
            best_state = state
            best_budget = budget

    if best_state is None:
        # fallback: original seed-like construction
        primes13 = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes13)]

    state = best_state
    budget = best_budget
    cnt, covered = make_covered(state)
    L = compute_L(covered)
    best_state = dict(state)
    best_L_sa = L

    def gaps_list():
        return [i for i in range(1, MAXL + 1) if not covered[i]]

    # ---- Simulated annealing with gap-focused moves ----
    rng = random.Random(2024)
    T = 1.0
    T_min = 1e-4
    alpha = 0.99985
    used_primes = list(state.keys())

    while time.time() < DEADLINE:
        T = max(T_min, T * alpha)
        mt = rng.random()
        old_L = L
        if mt < 0.45:
            # change residue of a used prime to cover a gap
            gaps = gaps_list()
            if gaps:
                g = rng.choice(gaps[:4000])
                cands = [p for p in used_primes if (g % p) != state[p]]
                if cands:
                    p = rng.choice(cands)
                    new_r = g % p
                    old_r = state[p]
                    swap_residue(cnt, covered, p, old_r, new_r)
                    state[p] = new_r
                    newL = 0
                    for i in range(1, MAXL + 1):
                        if covered[i]:
                            newL = i
                        else:
                            break
                    L = newL
                    delta = L - old_L
                    if delta < 0 and rng.random() > (0.08 ** (delta / 40.0 + 1.0)):
                        # revert
                        swap_residue(cnt, covered, p, new_r, old_r)
                        state[p] = old_r
                        L = old_L
        elif mt < 0.75:
            # add a new prime (if budget allows)
            if budget > 0:
                avail = [p for p in primes if p not in state and budget >= recip[p]]
                if avail:
                    p = rng.choice(avail)
                    # residue covering the first gap
                    gaps = gaps_list()
                    if gaps:
                        r = gaps[0] % p
                    else:
                        r = rng.randrange(p)
                    add_class(cnt, covered, p, r)
                    state[p] = r
                    used_primes.append(p)
                    budget -= recip[p]
                    newL = 0
                    for i in range(L + 1, MAXL + 1):
                        if covered[i]:
                            newL = i
                        else:
                            break
                    L = newL
                    if L < old_L and rng.random() > (0.05 ** 1.0):
                        # revert add
                        remove_class(cnt, covered, p, r)
                        del state[p]
                        used_primes.remove(p)
                        budget += recip[p]
                        L = old_L
        else:
            # remove a prime to free budget (only if we have slack in budget)
            if len(state) > 3:
                # remove a prime whose reciprocal is large to free budget
                p = max(state.keys(), key=lambda q: 1.0 / q)
                r = state[p]
                remove_class(cnt, covered, p, r)
                del state[p]
                used_primes.remove(p)
                budget += recip[p]
                newL = 0
                for i in range(1, MAXL + 1):
                    if covered[i]:
                        newL = i
                    else:
                        break
                L = newL
                delta = L - old_L
                if delta < 0 and rng.random() > (0.05 ** (delta / 60.0 + 1.0)):
                    # revert remove
                    add_class(cnt, covered, p, r)
                    state[p] = r
                    used_primes.append(p)
                    budget -= recip[p]
                    L = old_L

        if L > best_L_sa:
            best_L_sa = L
            best_state = dict(state)

    # ---- Final: trim to keep budget <= 1 (should already hold) and return ----
    total = Fraction(0, 1)
    for p in best_state:
        total += recip[p]
    if total > 1:
        # drop largest-reciprocal primes until within budget
        for p in sorted(best_state.keys(), key=lambda q: 1.0 / q, reverse=True):
            if total <= 1:
                break
            del best_state[p]
            total -= recip[p]
    return [[p, r] for p, r in best_state.items()]
# EVOLVE-BLOCK-END
