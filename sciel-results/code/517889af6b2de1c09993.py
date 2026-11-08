import time
import math
import random
import numpy as np
from itertools import combinations


def compute_score(U):
    U = sorted(set(U))
    if len(U) < 2:
        return 0
    maxU = U[-1]
    if maxU == 0:
        return 0
    n = len(U)
    # Use numpy for speed
    Ua = np.array(U, dtype=np.int64)
    sums = Ua[:, None] + Ua[None, :]
    diffs = Ua[:, None] - Ua[None, :]
    sum_set = set(sums.tolist())
    diff_set = set(diffs.tolist())
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom


def compute_score_fast(U):
    """Faster score using numpy, assumes U is sorted list of unique ints."""
    if len(U) < 2:
        return 0
    maxU = U[-1]
    if maxU == 0:
        return 0
    Ua = np.array(U, dtype=np.int64)
    sums = Ua[:, None] + Ua[None, :]
    diffs = Ua[:, None] - Ua[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0
    return 1 + math.log(ratio) / math.log(2 * maxU + 1)


def construct():
    start = time.time()
    deadline = start + 145  # safety margin under 150s

    best_score = 0
    best_U = [0, 1, 3]

    def consider(U):
        nonlocal best_score, best_U
        U = sorted(set(U))
        if len(U) < 2:
            return
        s = compute_score_fast(U)
        if s > best_score:
            best_score = s
            best_U = list(U)

    # --- Phase 1: Bose-Chowla sets for small primes ---
    # For prime p, a Bose-Chowla set of size p-1 has all nonzero differences distinct mod p.
    # We construct it greedily and verify all integer differences are distinct.
    def bose_chowla(p):
        """Find a Bose-Chowla set of size p-1 in {1,...,p-1} with 0, or None."""
        # Greedy: pick elements so that all pairwise differences are distinct mod p
        # and no difference is 0 mod p.
        used_diffs = set()
        S = [0]
        for x in range(1, p):
            # check if adding x keeps all pairwise diffs distinct mod p
            ok = True
            for y in S:
                d = (x - y) % p
                if d == 0:
                    ok = False
                    break
                if d in used_diffs:
                    ok = False
                    break
            if ok:
                for y in S:
                    d = (x - y) % p
                    used_diffs.add(d)
                S.append(x)
            if len(S) == p - 1:
                return S
        return None

    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]

    for p in primes:
        if time.time() > deadline - 30:
            break
        S = bose_chowla(p)
        if S is not None:
            # Verify all integer differences are distinct (not just mod p)
            S_sorted = sorted(S)
            diffs = set()
            ok = True
            for i in range(len(S_sorted)):
                for j in range(len(S_sorted)):
                    d = S_sorted[i] - S_sorted[j]
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
                if not ok:
                    break
            if ok:
                consider(S_sorted)
        if time.time() > deadline - 20:
            break

    # --- Phase 2: Local search / polish from best Bose-Chowla ---
    if time.time() < deadline - 15:
        random.seed(12345)
        U = list(best_U)
        max_val = max(U) if U else 10
        n_iter = 0
        stall = 0
        while time.time() < deadline - 10:
            n_iter += 1
            U_new = U[:]
            op = random.random()
            if op < 0.35:
                # add element
                if len(U_new) < 60:
                    v = random.randint(1, int(2 * max_val + 10))
                    if v not in U_new:
                        U_new.append(v)
                        U_new = sorted(set(U_new))
            elif op < 0.6:
                # remove element (not 0)
                if len(U_new) > 2:
                    idx = random.randint(1, len(U_new) - 1)
                    U_new.pop(idx)
            elif op < 0.85:
                # modify element (not 0)
                idx = random.randint(1, len(U_new) - 1)
                U_new[idx] = random.randint(1, int(2 * max_val + 20))
                U_new = sorted(set(U_new))
            else:
                # small perturbation of a random element
                idx = random.randint(1, len(U_new) - 1)
                delta = random.choice([-2, -1, 1, 2, 3, -3])
                U_new[idx] = max(1, U_new[idx] + delta)
                U_new = sorted(set(U_new))

            if len(U_new) < 2:
                continue
            s = compute_score_fast(U_new)
            if s > best_score:
                best_score = s
                best_U = U_new
                U = U_new[:]
                max_val = max(U)
                stall = 0
            else:
                stall += 1
                if stall > 200:
                    # occasionally restart from a random Bose-Chowla
                    stall = 0
                    if random.random() < 0.3:
                        p = random.choice([7, 11, 13, 17, 19, 23, 29, 31])
                        S = bose_chowla(p)
                        if S is not None:
                            U = sorted(S)
                            max_val = max(U)

    # --- Phase 3: Final high-precision polish ---
    if time.time() < deadline - 5:
        U = list(best_U)
        max_val = max(U) if U else 10
        # Try all single-element perturbations with finer steps
        for _ in range(3):
            improved = False
            for i in range(1, len(U)):
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    U_new = U[:]
                    U_new[i] = max(1, U[i] + delta)
                    U_new = sorted(set(U_new))
                    if len(U_new) < 2:
                        continue
                    s = compute_score_fast(U_new)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = U_new
                        U = U_new[:]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    return best_U


if __name__ == "__main__":
    result = construct()
    print("Best set:", result)
    print("Score:", compute_score(result))
