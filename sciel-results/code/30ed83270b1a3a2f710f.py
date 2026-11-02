import time
import random
import math
from itertools import combinations

def _score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    mx = max(U)
    if mx == 0:
        return 1.0
    n = len(U)
    # sumset
    sset = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sset.add(ui + U[j])
    # diffset
    dset = set()
    for i in range(n):
        ui = U[i]
        for j in range(n):
            dset.add(ui - U[j])
    ratio = len(dset) / len(sset)
    if ratio <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s kill

    best_set = [0, 1, 3]
    best_score = _score_set(best_set)

    # Precompute a few strong known constructions
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 4],
        [0, 1, 3, 4, 9],
        [0, 1, 3, 4, 9, 10],
    ]
    for c in candidates:
        s = _score_set(c)
        if s > best_score:
            best_score = s
            best_set = list(c)

    # Mian-Chowla-like greedy with randomization
    def greedy(start_set, max_elems, max_val, seed):
        rng = random.Random(seed)
        U = list(start_set)
        while len(U) < max_elems:
            # candidate next elements
            base = max(U) + 1
            cand_range = range(base, max_val + 1)
            # score each candidate by resulting improvement
            best_cand = None
            best_s = -1.0
            for c in cand_range:
                trial = U + [c]
                s = _score_set(trial)
                if s > best_s:
                    best_s = s
                    best_cand = c
            if best_cand is None:
                break
            U.append(best_cand)
        return U

    # Try a few greedy constructions with different seeds/params
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Vary the parameters
        seed = attempts
        max_elems = min(6, 3 + (attempts % 4))
        max_val = 30 + 10 * (attempts % 5)
        U = greedy([0, 1], max_elems, max_val, seed)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
            print(f"attempt {attempts}: score {best_score:.6f}, set {U}")

        # Also try local search: swap elements
        if best_score > 1.001:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                U = list(best_set)
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    orig = U[i]
                    for new_val in range(1, max(U) + 5):
                        if new_val == orig:
                            continue
                        trial = U[:]
                        trial[i] = new_val
                        if len(set(trial)) != len(trial):
                            continue
                        s = _score_set(trial)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = trial
                            improved = True
                            break
                    if improved:
                        break

    # Final: try random perturbations of best
    if time.time() < deadline - 5:
        rng = random.Random(42)
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            U = list(best_set)
            # randomly modify one element
            idx = rng.randint(1, len(U) - 1)
            old = U[idx]
            U[idx] = old + rng.randint(-3, 3)
            if U[idx] < 0:
                U[idx] = 0
            if len(set(U)) != len(U):
                continue
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)

    return best_set
