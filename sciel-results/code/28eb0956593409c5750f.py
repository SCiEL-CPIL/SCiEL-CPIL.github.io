"""Construct an additive basis A subset of [0,128] covering [128,256] by pair sums,
minimizing the peak (max) representation multiplicity in the target interval."""
import time
import random

N = 128
TLO = 128
THI = 256
TARGETS = list(range(TLO, THI + 1))  # 129 targets


def count_multiplicities(A):
    """Return list of representation counts for each target in [128,256]."""
    Aset = set(A)
    counts = [0] * (THI + 1)
    for i, a in enumerate(A):
        # a + a
        s = a + a
        if TLO <= s <= THI:
            counts[s] += 1
        for b in A[i + 1:]:
            s = a + b
            if TLO <= s <= THI:
                counts[s] += 1
    return counts


def peak_of(counts):
    return max(counts[t] for t in TARGETS)


def coverage_ok(counts):
    return all(counts[t] >= 1 for t in TARGETS)


def peak_and_ok(A):
    counts = count_multiplicities(A)
    return peak_of(counts), coverage_ok(counts)


def greedy_construct():
    """Greedy: start with interval, then try to reduce peak by moving elements."""
    A = list(range(64, 129))  # 65 elements, known valid, peak 33
    return A


def local_search(A, time_limit):
    """Local search via add/remove/swap to reduce peak while keeping coverage."""
    A = sorted(A)
    counts = count_multiplicities(A)
    cur_peak = peak_of(counts)
    start = time.time()
    best = (cur_peak, A[:])

    def evaluate(newA):
        nonlocal cur_peak, counts
        nc = count_multiplicities(newA)
        if not coverage_ok(nc):
            return None
        np = peak_of(nc)
        return np, nc

    while time.time() - start < time_limit:
        improved = False
        # candidate moves: remove an element, add an element, swap
        # Try to remove elements that contribute to peak sums
        Aset = set(A)
        for a in A[:]:
            if time.time() - start >= time_limit:
                break
            if a in Aset:
                newA = [x for x in A if x != a]
                res = evaluate(newA)
                if res is not None:
                    np, nc = res
                    if np < cur_peak:
                        A = newA
                        Aset = set(A)
                        cur_peak = np
                        counts = nc
                        best = (cur_peak, A[:])
                        improved = True
                        break
        if improved:
            continue
        # Try adding low elements to spread out representations
        for cand in range(0, 64):
            if time.time() - start >= time_limit:
                break
            if cand not in Aset:
                newA = A + [cand]
                res = evaluate(newA)
                if res is not None:
                    np, nc = res
                    if np < cur_peak:
                        A = newA
                        Aset = set(A)
                        cur_peak = np
                        counts = nc
                        best = (cur_peak, A[:])
                        improved = True
                        break
        if improved:
            continue
        # Try swap: remove a, add b
        for a in A[:]:
            if time.time() - start >= time_limit:
                break
            for b in range(0, 64):
                if b == a or b in Aset:
                    continue
                newA = [x for x in A if x != a] + [b]
                res = evaluate(newA)
                if res is not None:
                    np, nc = res
                    if np < cur_peak:
                        A = newA
                        Aset = set(A)
                        cur_peak = np
                        counts = nc
                        best = (cur_peak, A[:])
                        improved = True
                        break
            if improved:
                break
        if not improved:
            # try random restarts / perturbations
            for _ in range(200):
                if time.time() - start >= time_limit:
                    break
                # random add then random remove
                newA = A[:]
                # add a random low element
                cand = random.randint(0, 63)
                if cand not in newA:
                    newA.append(cand)
                # remove a random element (not needed for coverage)
                rem = random.choice(newA)
                newA = [x for x in newA if x != rem]
                res = evaluate(newA)
                if res is not None:
                    np, nc = res
                    if np < cur_peak:
                        A = newA
                        Aset = set(A)
                        cur_peak = np
                        counts = nc
                        best = (cur_peak, A[:])
                        improved = True
                        break
        if not improved:
            break
    return best[1]


def construct():
    start = time.time()
    deadline = 75.0  # safety margin before 90s kill
    A = greedy_construct()
    A = local_search(A, deadline - (time.time() - start))
    return A


# Quick self-test when run directly
if __name__ == "__main__":
    A = construct()
    counts = count_multiplicities(A)
    print("size:", len(A))
    print("peak:", peak_of(counts))
    print("coverage:", coverage_ok(counts))
