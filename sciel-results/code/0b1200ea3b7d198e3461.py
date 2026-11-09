"""Fresh design: distinct-prime residue classes; SA local search to maximize covered prefix."""
import time
import random
import math
from fractions import Fraction

_DEADLINE = 140.0
_START = None

def _elapsed():
    return time.time() - _START

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _eval(state):
    # state: list of (p, r)
    covered = bytearray(1001)
    for p, r in state:
        if r <= 1000:
            for x in range(r, 1001, p):
                covered[x] = 1
    L = 0
    while L < 1000 and covered[L + 1]:
        L += 1
    return L

def _budget(state):
    return sum(Fraction(1, p) for p, r in state)

def construct():
    global _START
    _START = time.time()

    primes = _primes_upto(1000)
    small = [p for p in primes if p <= 80]
    large = [p for p in primes if p > 80]

    def _rand_prime(exclude):
        pool = [p for p in (small if random.random() < 0.6 else large) if p not in exclude]
        return random.choice(pool)

    def _init_state():
        # Start with a set of small primes, residues chosen to cover low numbers.
        state = []
        used = set()
        for p in small:
            if _budget(state) + Fraction(1, p) <= 1:
                r = random.randrange(1, p)
                state.append((p, r))
                used.add(p)
        return state

    def _local_search(state):
        # For each class, try all residues, keep the one giving best prefix.
        # Also try swapping primes.
        improved = True
        it = 0
        while improved and _elapsed() < _DEADLINE:
            improved = False
            it += 1
            order = list(range(len(state)))
            random.shuffle(order)
            for i in order:
                p, r = state[i]
                best_r = r
                best_L = _eval(state)
                for nr in range(p):
                    state[i] = (p, nr)
                    L = _eval(state)
                    if L > best_L:
                        best_L = L
                        best_r = nr
                state[i] = (p, best_r)
                if best_r != r:
                    improved = True
        return state

    def _sa(state):
        cur = state[:]
        cur_L = _eval(cur)
        best = cur[:]
        best_L = cur_L
        temp = 300.0
        cool = 0.99
        while temp > 0.5 and _elapsed() < _DEADLINE:
            used = set(p for p, r in cur)
            # choose an operation
            op = random.random()
            if op < 0.5 and len(cur) > 0:
                # modify a class
                i = random.randrange(len(cur))
                old_p, old_r = cur[i]
                new_p = _rand_prime(used - {old_p})
                new_r = random.randrange(new_p)
                new_budget = _budget(cur) - Fraction(1, old_p) + Fraction(1, new_p)
                if new_budget > 1:
                    continue
                cur[i] = (new_p, new_r)
                new_L = _eval(cur)
                d = new_L - cur_L
                if d >= 0 or random.random() < math.exp(d / temp):
                    cur_L = new_L
                    if new_L > best_L:
                        best_L = new_L
                        best = cur[:]
                else:
                    cur[i] = (old_p, old_r)
            elif op < 0.75:
                # add a class
                new_p = _rand_prime(used)
                new_r = random.randrange(new_p)
                if _budget(cur) + Fraction(1, new_p) <= 1:
                    cur.append((new_p, new_r))
                    new_L = _eval(cur)
                    d = new_L - cur_L
                    if d >= 0 or random.random() < math.exp(d / temp):
                        cur_L = new_L
                        if new_L > best_L:
                            best_L = new_L
                            best = cur[:]
                    else:
                        cur.pop()
            else:
                # remove a class
                if len(cur) > 1:
                    i = random.randrange(len(cur))
                    old = cur.pop(i)
                    new_L = _eval(cur)
                    d = new_L - cur_L
                    if d >= 0 or random.random() < math.exp(d / temp):
                        cur_L = new_L
                        if new_L > best_L:
                            best_L = new_L
                            best = cur[:]
                    else:
                        cur.insert(i, old)
            temp *= cool
        return best, best_L

    best_state = None
    best_L = 0
    for restart in range(200):
        if _elapsed() > _DEADLINE - 5:
            break
        state = _init_state()
        state = _local_search(state)
        L = _eval(state)
        if L > best_L:
            best_L = L
            best_state = state
        # SA from this state
        if _elapsed() > _DEADLINE - 10:
            break
        s, L2 = _sa(state)
        if L2 > best_L:
            best_L = L2
            best_state = s
        if best_L >= 999:
            break

    if best_state is None:
        best_state = _init_state()
        best_state = _local_search(best_state)

    # Final polish: local search on best
    if _elapsed() < _DEADLINE - 3:
        best_state = _local_search(best_state)

    # Validate budget
    if _budget(best_state) > 1:
        # drop classes until valid (keep those with largest p? no, keep small)
        # Just return empty-safe fallback
        best_state = [(p, r) for p, r in best_state if True]
        # recompute: remove largest-1/p contributors? Actually 1/p smaller for large p.
        # To reduce budget, remove classes with small p (large 1/p).
        best_state.sort(key=lambda pr: pr[0])  # ascending p => largest 1/p first
        # remove until valid
        while best_state and _budget(best_state) > 1:
            best_state.pop(0)

    return [[p, r] for p, r in best_state]
