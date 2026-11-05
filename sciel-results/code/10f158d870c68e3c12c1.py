"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    recip = [0.0] + [1.0 / i for i in range(1, n + 1)]

    def can_add(x, in_set):
        """Check if x can be added without creating a 3-AP. in_set is a list of bools."""
        for a in range(1, n + 1):
            if not in_set[a]:
                continue
            # x is middle: a, x, b => b = 2x - a
            b = 2 * x - a
            if b != x and 1 <= b <= n and in_set[b]:
                return False
            # x is endpoint, a is middle: x, a, b => b = 2a - x
            b2 = 2 * a - x
            if b2 != x and 1 <= b2 <= n and in_set[b2]:
                return False
            # x is endpoint, a is other endpoint: a, m, x => m = (a+x)/2
            s = a + x
            if s % 2 == 0:
                m = s // 2
                if m != a and m != x and 1 <= m <= n and in_set[m]:
                    return False
        return True

    def greedy_fill(in_set):
        """Add all valid numbers in increasing order."""
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if not in_set[x] and can_add(x, in_set):
                    in_set[x] = True
                    changed = True
        return in_set

    def get_sum(in_set):
        return sum(recip[i] for i in range(1, n + 1) if in_set[i])

    def get_list(in_set):
        return [i for i in range(1, n + 1) if in_set[i]]

    # Greedy initialization
    in_set = [False] * (n + 1)
    for x in range(1, n + 1):
        if can_add(x, in_set):
            in_set[x] = True
    greedy_fill(in_set)

    best_in_set = in_set[:]
    best_sum = get_sum(best_in_set)

    # Local search: try removing each element and re-filling
    def local_search(current_in_set):
        improved = True
        cur_sum = get_sum(current_in_set)
        while improved and time.time() < deadline - 5.0:
            improved = False
            for rem in range(1, n + 1):
                if time.time() > deadline - 5.0:
                    break
                if not current_in_set[rem]:
                    continue
                trial = current_in_set[:]
                trial[rem] = False
                greedy_fill(trial)
                new_sum = get_sum(trial)
                if new_sum > cur_sum + 1e-12:
                    current_in_set = trial
                    cur_sum = new_sum
                    improved = True
                    break
        return current_in_set, cur_sum

    best_in_set, best_sum = local_search(best_in_set)

    # Randomized restarts / perturbations to escape local optima
    random.seed(12345)
    while time.time() < deadline - 10.0:
        perturbed = best_in_set[:]
        num_removals = random.randint(1, 4)
        candidates = [i for i in range(1, n + 1) if perturbed[i]]
        if not candidates:
            break
        for _ in range(num_removals):
            if not candidates:
                break
            idx = random.randint(0, len(candidates) - 1)
            perturbed[candidates[idx]] = False
            candidates.pop(idx)
        greedy_fill(perturbed)
        perturbed, pert_sum = local_search(perturbed)
        if pert_sum > best_sum + 1e-12:
            best_in_set = perturbed
            best_sum = pert_sum

    # Final verification
    S_list = get_list(best_in_set)
    S_set = set(S_list)
    valid = True
    for i, a in enumerate(S_list):
        for b in S_list[i + 1:]:
            c = 2 * b - a
            if c in S_set:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: rebuild greedily
        S = [False] * (n + 1)
        for x in range(1, n + 1):
            if can_add(x, S):
                S[x] = True
        greedy_fill(S)
        return get_list(S)

    return S_list
# EVOLVE-BLOCK-END
