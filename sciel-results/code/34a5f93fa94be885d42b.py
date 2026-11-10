"""Rewritten program: greedy removed-residue-class selection with admissibility repair."""
import time

N = 5000


def _primes(limit):
    sieve = bytearray([1]) * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:limit + 1:step] = bytearray(len(range(start, limit + 1, step)))
    return [p for p in range(limit + 1) if sieve[p]]


def construct():
    t0 = time.time()
    DEADLINE = 300.0

    primes = _primes(N)
    P = len(primes)
    pset = set(primes)
    n = N + 1

    # residue vector per integer, and class membership per (prime, residue)
    rvec = []
    for x in range(n):
        rv = [0] * P
        for pi, p in enumerate(primes):
            rv[pi] = x % p
        rvec.append(rv)

    classes = [[[] for _ in range(p)] for p in primes]
    for x in range(n):
        rv = rvec[x]
        for pi, p in enumerate(primes):
            classes[pi][rv[pi]].append(x)

    # initial admissible set: primes p with p*p > N (each is alone in its class 0) plus 1
    # (1 is never a prime, so it shares no class 0 with any prime)
    init = [1] + [p for p in primes if p * p > N]
    removed = [0] * P  # removed[pi] = -1 means "not removed"
    removed = [-1] * P
    for pi, p in enumerate(primes):
        if p * p > N:
            removed[pi] = 0

    def _is_admissible(rem):
        for pi, p in enumerate(primes):
            if p > len(init):
                continue
            r = rem[pi]
            if r == -1:
                # class not removed: must be empty in init
                for v in classes[pi][0]:
                    if v in init:
                        return False
            else:
                for v in classes[pi][r]:
                    if v in init:
                        return False
        return True

    if not _is_admissible(removed):
        return init

    best = list(init)
    best_len = len(init)

    # Greedy: repeatedly pick the (pi, r) removal that minimizes newly-deleted count.
    # A prime pi needs a removal if its current class is not empty in init.
    while time.time() < DEADLINE:
        # find primes that currently have a non-empty class in init (need removal)
        need = []
        for pi, p in enumerate(primes):
            if p > len(init):
                continue
            r = removed[pi]
            r = 0 if r == -1 else r
            nonempty = False
            for v in classes[pi][r]:
                if v in init:
                    nonempty = True
                    break
            if nonempty:
                need.append(pi)
        if not need:
            break

        # evaluate candidate removals for each needed prime
        best_new = None
        best_choice = None
        for pi in need:
            p = primes[pi]
            for r in range(p):
                if r == removed[pi]:
                    continue
                # count new deletions
                cnt = 0
                for v in classes[pi][r]:
                    if v in init:
                        cnt += 1
                if best_new is None or cnt < best_new:
                    best_new = cnt
                    best_choice = (pi, r)
        if best_choice is None:
            break
        pi, r = best_choice
        removed[pi] = r
        # delete elements in this class
        for v in classes[pi][r]:
            if v in init:
                init.remove(v)
        if len(init) > best_len:
            best_len = len(init)
            best = list(init)

    # verify final admissibility
    if _is_admissible(removed):
        if len(init) > best_len:
            best = list(init)
    else:
        # fall back to best known admissible set
        pass

    return best
