import time
import math
import random
import numpy as np
from itertools import combinations

# Use numpy for fast sum/diff set computation
def compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    maxU = U[-1]
    if maxU == 0:
        return 0.0

    # Use numpy int64 for speed (safe for reasonable sizes)
    if n <= 200 and maxU <= 2**62:
        arr = np.array(U, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        sum_count = len(np.unique(sums))
        diff_count = len(np.unique(diffs))
    else:
        # Fallback to Python sets for very large values
        sum_set = set()
        diff_set = set()
        for i in range(n):
            ui = U[i]
            for j in range(n):
                sum_set.add(ui + U[j])
                diff_set.add(ui - U[j])
        sum_count = len(sum_set)
        diff_count = len(diff_set)

    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * maxU + 1)
    return score


def exhaustive_search_small(n, max_val, deadline):
    """Exhaustive search for small sets of size n"""
    best_score = 0.0
    best_set = None
    candidates = list(range(1, max_val + 1))
    for combo in combinations(candidates, n - 1):
        if time.time() > deadline:
            break
        U = sorted([0] + list(combo))
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    return best_score, best_set


def parametric_families(deadline):
    """Try various parametric families"""
    best_score = 0.0
    best_set = None

    def try_set(U):
        nonlocal best_score, best_set
        if not U or 0 not in U:
            return
        U = sorted(set(U))
        if len(U) < 2:
            return
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U

    # Family 1: arithmetic progression
    for n in range(3, 60):
        if time.time() > deadline:
            break
        try_set(list(range(n)))

    # Family 2: Mersenne-like (2^k - 1)
    for n in range(3, 20):
        if time.time() > deadline:
            break
        try_set([0] + [2**k - 1 for k in range(1, n)])

    # Family 3: powers of 2
    for n in range(3, 20):
        if time.time() > deadline:
            break
        try_set([0] + [2**k for k in range(1, n)])

    # Family 4: triangular numbers
    for n in range(3, 40):
        if time.time() > deadline:
            break
        try_set([0] + [k * (k + 1) // 2 for k in range(1, n)])

    # Family 5: quadratic
    for n in range(3, 40):
        if time.time() > deadline:
            break
        try_set([0] + [k * k // 2 for k in range(1, n)])

    # Family 6: Mixed dense + sparse
    for base_n in range(3, 15):
        if time.time() > deadline:
            break
        for gap in range(2, 30):
            if time.time() > deadline:
                break
            for extra in range(0, 8):
                if time.time() > deadline:
                    break
                U = list(range(base_n))
                for i in range(extra):
                    U.append(base_n * gap * (i + 1))
                try_set(U)

    # Family 7: geometric progression
    for b in range(2, 15):
        if time.time() > deadline:
            break
        for n in range(3, 15):
            if time.time() > deadline:
                break
            try_set([0] + [b**k for k in range(1, n)])

    # Family 8: two-segment
    for m in range(3, 12):
        if time.time() > deadline:
            break
        for b in range(2, 10):
            if time.time() > deadline:
                break
            for ext in range(0, 5):
                if time.time() > deadline:
                    break
                U = list(range(m))
                for i in range(ext):
                    U.append(m * b * (i + 1))
                try_set(U)

    return best_score, best_set


def hill_climb(initial_set, deadline, max_iter=5000):
    """Local search via element replacement"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    best_score = compute_score(U)
    best_set = U[:]

    max_val = max(U)
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        improved = False
        n = len(U)

        for i in range(n):
            if time.time() > deadline:
                break
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_val = random.randint(0, max_val * 2)
                if new_val in U and new_val != U[i]:
                    continue
                new_U = U[:i] + [new_val] + U[i+1:]
                new_U = sorted(set(new_U))
                if 0 not in new_U:
                    new_U = sorted([0] + [x for x in new_U if x > 0])
                if len(new_U) < 2:
                    continue
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(U)
                    improved = True
                    break

        if not improved:
            if len(U) < 50:
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    new_val = random.randint(1, max_val * 3)
                    new_U = sorted(U + [new_val])
                    s = compute_score(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = new_U
                        U = new_U
                        max_val = max(U)
                        improved = True
                        break

            if not improved:
                if len(U) > 3:
                    for i in range(len(U)):
                        if time.time() > deadline:
                            break
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = compute_score(new_U)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = new_U
                            U = new_U
                            max_val = max(U)
                            improved = True
                            break

            if not improved:
                break

    return best_score, best_set


def simulated_annealing(initial_set, deadline, max_iter=10000):
    """Simulated annealing to escape local optima"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U

    current_score = compute_score(U)
    best_score = current_score
    best_set = U[:]

    T0 = 0.5
    T_min = 0.001
    cooling_rate = 0.9995

    T = T0
    stagnant = 0

    for it in range(max_iter):
        if time.time() > deadline:
            break

        n = len(U)
        max_val = max(U)

        move_type = random.random()

        if move_type < 0.6:
            # Replace one element
            i = random.randint(1, n - 1)
            if random.random() < 0.3:
                new_val = random.randint(0, max_val * 2)
            elif random.random() < 0.5:
                new_val = random.randint(0, max_val)
            else:
                new_val = max(0, U[i] + random.randint(-max_val // 2, max_val // 2))

            if new_val in U and new_val != U[i]:
                continue

            new_U = sorted(set(U[:i] + [new_val] + U[i+1:]))
            if 0 not in new_U:
                new_U = sorted([0] + [x for x in new_U if x > 0])

        elif move_type < 0.8:
            # Add an element
            if len(U) >= 50:
                continue
            if random.random() < 0.3:
                new_val = max_val + random.randint(1, max(1, max_val // 3))
            else:
                new_val = random.randint(1, max_val * 2)

            if new_val in U:
                continue

            new_U = sorted(U + [new_val])

        elif move_type < 0.9:
            # Remove an element
            if len(U) <= 3:
                continue
            i = random.randint(1, n - 1)
            new_U = U[:i] + U[i+1:]

        else:
            # Swap two elements
            if n < 4:
                continue
            i, j = random.sample(range(1, n), 2)
            new_i = random.randint(0, max_val * 2)
            new_j = random.randint(0, max_val * 2)

            new_U = sorted(set(U[:i] + [new_i] + U[i+1:j] + [new_j] + U[j+1:]))
            if 0 not in new_U:
                new_U = sorted([0] + [x for x in new_U if x > 0])

        if len(new_U) < 2:
            continue

        new_score = compute_score(new_U)

        delta = new_score - current_score
        if delta > 0 or (T > T_min and random.random() < math.exp(delta / T)):
            current_score = new_score
            U = new_U

            if current_score > best_score + 1e-12:
                best_score = current_score
                best_set = U[:]
                stagnant = 0
            else:
                stagnant += 1
        else:
            stagnant += 1

        T *= cooling_rate

        if stagnant > 200 and T < 0.01:
            break

    return best_score, best_set


def polish_pass(U, deadline):
    """Final high-precision polish: tight local search with small perturbations"""
    U = sorted(set(U))
    if len(U) < 2:
        return U
    best_score = compute_score(U)
    best_set = U[:]
    max_val = max(U)

    # Try small integer perturbations around each element
    for i in range(1, len(U)):
        if time.time() > deadline:
            break
        for delta in [-3, -2, -1, 1, 2, 3]:
            if time.time() > deadline:
                break
            new_val = U[i] + delta
            if new_val < 0:
                continue
            if new_val in U and new_val != U[i]:
                continue
            new_U = U[:i] + [new_val] + U[i+1:]
            new_U = sorted(set(new_U))
            if 0 not in new_U:
                continue
            if len(new_U) < 2:
                continue
            s = compute_score(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_set = new_U
                U = new_U
                max_val = max(U)

    return best_set


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)

    best_score = 0.0
    best_set = [0, 1, 3]

    # 1. Small exhaustive search
    for n in range(3, 7):
        if time.time() > deadline:
            break
        max_val = 30 if n <= 5 else 20
        s, U = exhaustive_search_small(n, max_val, deadline)
        if s > best_score:
            best_score = s
            best_set = U

    # 2. Parametric families
    s, U = parametric_families(deadline)
    if s > best_score:
        best_score = s
        best_set = U

    # 3. Hill climb from best
    if time.time() < deadline - 10:
        s, U = hill_climb(best_set, deadline, max_iter=3000)
        if s > best_score:
            best_score = s
            best_set = U

    # 4. Simulated annealing from best
    if time.time() < deadline - 10:
        random.seed(42)
        s, U = simulated_annealing(best_set, deadline, max_iter=5000)
        if s > best_score:
            best_score = s
            best_set = U

    # 5. Hill climb from random starts
    if time.time() < deadline - 5:
        random.seed(42)
        for seed in range(5):
            if time.time() > deadline:
                break
            random.seed(seed)
            n = random.randint(4, 12)
            U = sorted(random.sample(range(1, 100), n - 1) + [0])
            s, U2 = hill_climb(U, deadline, max_iter=1000)
            if s > best_score:
                best_score = s
                best_set = U2

    # 6. Simulated annealing from random starts
    if time.time() < deadline - 5:
        random.seed(123)
        for seed in range(3):
            if time.time() > deadline:
                break
            random.seed(seed + 100)
            n = random.randint(4, 12)
            U = sorted(random.sample(range(1, 100), n - 1) + [0])
            s, U2 = simulated_annealing(U, deadline, max_iter=2000)
            if s > best_score:
                best_score = s
                best_set = U2

    # 7. Final polish pass
    if time.time() < deadline - 2:
        best_set = polish_pass(best_set, deadline)
        final_score = compute_score(best_set)
        if final_score > best_score:
            best_score = final_score

    return best_set


# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
