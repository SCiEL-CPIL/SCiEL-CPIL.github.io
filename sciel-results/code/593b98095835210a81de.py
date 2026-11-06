import time
from math import gcd
from functools import reduce
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin

    # Precompute LCM of 1..24
    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, 25))

    # V[x] = L // x
    V = {x: L // x for x in range(1, 25)}

    S = []
    current_sums = {0}

    # Greedy: iterate from 24 down to 1
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break

        if V[x] not in current_sums:
            S.append(x)
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + V[x])
            current_sums.update(new_sums)

            if len(current_sums) > 5_000_000:
                break

    # Local search: try to add back skipped elements
    selected = set(S)
    skipped = [x for x in range(1, 25) if x not in selected]

    def is_valid_set(S_set):
        """Check if S_set is reciprocal-subsum-free."""
        S_list = list(S_set)
        for i, x in enumerate(S_list):
            # Check if 1/x equals sum of reciprocals of any subset of OTHER elements
            other = [y for y in S_list if y != x]
            n = len(other)
            # Enumerate all subsets of other
            for mask in range(1, 1 << n):
                subset_sum = 0
                for j in range(n):
                    if mask & (1 << j):
                        subset_sum += V[other[j]]
                if subset_sum == V[x]:
                    return False
        return True

    def count_conflicts(S_set):
        """Count number of conflict constraints violated."""
        S_list = list(S_set)
        conflicts = 0
        for i, x in enumerate(S_list):
            other = [y for y in S_list if y != x]
            n = len(other)
            for mask in range(1, 1 << n):
                subset_sum = 0
                for j in range(n):
                    if mask & (1 << j):
                        subset_sum += V[other[j]]
                if subset_sum == V[x]:
                    conflicts += 1
        return conflicts

    # Simulated annealing to improve the set
    best_set = set(S)
    best_size = len(S)
    current_set = set(S)
    current_conflicts = count_conflicts(current_set)

    # If current set is valid, try to add more elements
    if current_conflicts == 0:
        # Try adding each skipped element
        for x in sorted(skipped, reverse=True):
            if time.time() > deadline:
                break
            trial_set = current_set | {x}
            if count_conflicts(trial_set) == 0:
                current_set = trial_set
                best_set = set(current_set)
                best_size = len(current_set)

    # Simulated annealing
    if time.time() < deadline - 5.0:
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        random.seed(42)

        while T > min_T and time.time() < deadline - 2.0:
            # Generate a move: add, remove, or swap
            move_type = random.random()
            if move_type < 0.3 and skipped:
                # Add a random skipped element
                x = random.choice(skipped)
                new_set = current_set | {x}
            elif move_type < 0.6 and current_set:
                # Remove a random element
                x = random.choice(list(current_set))
                new_set = current_set - {x}
            elif current_set and skipped:
                # Swap: remove one, add one
                x_remove = random.choice(list(current_set))
                x_add = random.choice(skipped)
                new_set = (current_set - {x_remove}) | {x_add}
            else:
                T *= cooling
                continue

            new_conflicts = count_conflicts(new_set)
            delta_conflicts = new_conflicts - current_conflicts
            delta_size = len(new_set) - len(current_set)

            # Accept if: fewer conflicts, or same conflicts with larger size, or probabilistic
            if delta_conflicts < 0 or (delta_conflicts == 0 and delta_size > 0):
                current_set = new_set
                current_conflicts = new_conflicts
                if current_conflicts == 0 and len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
            elif random.random() < min(1.0, 1.0 / (1.0 + delta_conflicts * 10.0 - delta_size * 5.0)):
                current_set = new_set
                current_conflicts = new_conflicts

            T *= cooling

    # Final check: ensure best_set is valid
    if count_conflicts(best_set) == 0:
        result = sorted(best_set)
    else:
        # Fallback: use greedy result
        result = sorted(S)

    if not result:
        return [24]

    return result

# EVOLVE-BLOCK-END
