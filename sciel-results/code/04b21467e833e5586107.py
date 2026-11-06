"""Combined program: structured (geometric) seeds + incremental objective + SA polish."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    n = 20
    max_val = 10000

    # ---- candidate pool: 2,3-smooth (Hamming) numbers <= max_val ----
    smooths = set()
    p2 = 1
    while p2 <= max_val:
        p3 = p2
        while p3 <= max_val:
            smooths.add(p3)
            p3 *= 3
        p2 *= 2
    smooths = sorted(smooths)

    # ---- fast incremental objective: max(|A+A|, |A*A|) ----
    def compute_objective(A):
        nA = len(A)
        sums = set()
        prods = set()
        for i in range(nA):
            ai = A[i]
            sums.add(ai + ai)
            prods.add(ai * ai)
            for j in range(i + 1, nA):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))

    # ---- structured seeds (strongest idea from inspiration 1) ----
    # Pure geometric {b, b^2, ..., b^20}: sumset == productset == n(n+1)/2 = 210
    best_set = None
    best_score = float('inf')

    def consider(A):
        nonlocal best_set, best_score
        if len(set(A)) != n or max(A) > max_val or min(A) < 1:
            return
        s = compute_objective(A)
        if s < best_score:
            best_score = s
            best_set = sorted(A)

    # Family 1: pure geometric
    for b in range(2, 13):
        A = [b ** i for i in range(1, n + 1)]
        if max(A) <= max_val:
            consider(A)

    # Family 2: arithmetic-geometric a*b^i
    for a in range(1, 12):
        for b in range(2, 8):
            A = [a * (b ** i) for i in range(n)]
            if len(set(A)) == n and max(A) <= max_val:
                consider(A)

    # Family 3: smallest 20 smooths (dense structure)
    if len(smooths) >= n:
        consider(smooths[:n])

    if best_set is None:
        best_set = list(range(1, n + 1))
        best_score = compute_objective(best_set)

    # ---- Simulated Annealing with ratio-preserving moves ----
    current_set = best_set[:]
    membership = set(current_set)
    current_score = best_score
    best_set_sa = current_set[:]
    best_score_sa = current_score

    T = 8.0
    T_min = 0.1
    cooling = 0.995
    sa_deadline = deadline - 12

    while time.time() < sa_deadline:
        if T < T_min:
            T = T_min

        i = random.randint(0, n - 1)
        mt = random.random()

        if mt < 0.30:
            # ratio-preserving move: multiply/divide by a small integer
            base = current_set[i]
            if random.random() < 0.5:
                new_val = base * random.choice([2, 3, 4, 5])
            else:
                d = random.choice([2, 3, 4, 5])
                if base % d == 0:
                    new_val = base // d
                else:
                    new_val = base
        elif mt < 0.55:
            # replace with a smooth number
            new_val = random.choice(smooths)
        elif mt < 0.80:
            # small perturbation
            new_val = current_set[i] + random.randint(-8, 8)
        else:
            # swap two elements (keeps multiset, tests ordering-independent structure)
            j = random.randint(0, n - 1)
            if j == i:
                continue
            current_set[i], current_set[j] = current_set[j], current_set[i]
            new_score = compute_objective(current_set)
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_set_sa = current_set[:]
                current_score = new_score
            else:
                current_set[i], current_set[j] = current_set[j], current_set[i]
            T *= cooling
            continue

        if not (1 <= new_val <= max_val) or new_val in membership:
            T *= cooling
            continue

        old_val = current_set[i]
        current_set[i] = new_val
        membership.remove(old_val)
        membership.add(new_val)

        new_score = compute_objective(current_set)
        delta = new_score - current_score

        if delta < 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_set_sa = current_set[:]
        else:
            current_set[i] = old_val
            membership.remove(new_val)
            membership.add(old_val)

        T *= cooling

    # ---- final local-search polish (exact, incremental) ----
    if time.time() < deadline - 2:
        membership = set(best_set_sa)
        current_set = best_set_sa[:]
        current_score = best_score_sa
        for _ in range(40):
            improved = False
            for i in range(n):
                for delta_val in range(-6, 7):
                    if delta_val == 0:
                        continue
                    new_val = current_set[i] + delta_val
                    if not (1 <= new_val <= max_val) or new_val in membership:
                        continue
                    old_val = current_set[i]
                    current_set[i] = new_val
                    membership.remove(old_val)
                    membership.add(new_val)
                    new_score = compute_objective(current_set)
                    if new_score < best_score_sa:
                        best_score_sa = new_score
                        best_set_sa = current_set[:]
                        current_score = new_score
                        improved = True
                    else:
                        current_set[i] = old_val
                        membership.remove(new_val)
                        membership.add(old_val)
            if not improved:
                break

    best_set_sa = sorted(set(best_set_sa))
    return best_set_sa

# EVOLVE-BLOCK-END
