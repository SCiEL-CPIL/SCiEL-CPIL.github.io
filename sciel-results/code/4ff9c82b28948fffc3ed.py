"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    denominators = list(range(2, 49))
    n = len(denominators)

    # Compute LCM of 2..48 for exact integer arithmetic
    lcm_val = 1
    for d in range(2, 49):
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)

    # Precompute numerator for each denominator: 1/d = (lcm_val/d) / lcm_val
    nums = [lcm_val // d for d in denominators]

    # Sort indices by weight descending (largest unit fractions first)
    sorted_indices = sorted(range(n), key=lambda i: -nums[i])

    def greedy_init(order):
        """Greedy: add elements in given order that keep sum < lcm_val."""
        selected = []
        total = 0
        for i in order:
            if total + nums[i] < lcm_val:
                selected.append(i)
                total += nums[i]
        return selected, total

    def local_search(selected, total, time_limit=None):
        """Improve via 1-add, 1-swap, 1-for-2-swap, and 2-for-2-swap moves."""
        if time_limit is None:
            time_limit = deadline - 1.0

        in_set = [False] * n
        for i in selected:
            in_set[i] = True

        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # 1-add: try adding an unselected element
            for j in range(n):
                if not in_set[j]:
                    if total + nums[j] < lcm_val:
                        selected.append(j)
                        in_set[j] = True
                        total += nums[j]
                        improved = True
                        break

            if improved:
                continue

            # 1-swap: remove i, add j where gain > 0 and new total < lcm_val
            best_gain = 0
            best_move = None
            for i in selected:
                vi = nums[i]
                for j in range(n):
                    if not in_set[j]:
                        vj = nums[j]
                        gain = vj - vi
                        if gain > 0:
                            new_total = total - vi + vj
                            if new_total < lcm_val:
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = (i, j)
            if best_move:
                i, j = best_move
                in_set[i] = False
                in_set[j] = True
                selected.remove(i)
                selected.append(j)
                total += nums[j] - nums[i]
                improved = True
                continue

            # 1-for-2-swap: remove i, add j and k
            best_gain = 0
            best_move = None
            for i in selected:
                vi = nums[i]
                base = total - vi
                for j in range(n):
                    if not in_set[j]:
                        vj = nums[j]
                        if base + vj >= lcm_val:
                            continue
                        for k in range(j + 1, n):
                            if not in_set[k]:
                                vk = nums[k]
                                new_total = base + vj + vk
                                if new_total < lcm_val:
                                    gain = vj + vk - vi
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_move = (i, j, k)
            if best_move:
                i, j, k = best_move
                in_set[i] = False
                in_set[j] = True
                in_set[k] = True
                selected.remove(i)
                selected.append(j)
                selected.append(k)
                total += nums[j] + nums[k] - nums[i]
                improved = True
                continue

            # 2-for-2-swap: remove i1, i2, add j, k
            best_gain = 0
            best_move = None
            sel_list = list(selected)
            for idx1, i in enumerate(sel_list):
                vi = nums[i]
                for idx2, i2 in enumerate(sel_list):
                    if idx2 <= idx1:
                        continue
                    vi2 = nums[i2]
                    base = total - vi - vi2
                    for j in range(n):
                        if in_set[j]:
                            continue
                        vj = nums[j]
                        if base + vj >= lcm_val:
                            continue
                        for k in range(j + 1, n):
                            if in_set[k]:
                                continue
                            vk = nums[k]
                            new_total = base + vj + vk
                            if new_total < lcm_val:
                                gain = vj + vk - vi - vi2
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = (i, i2, j, k)
            if best_move:
                i, i2, j, k = best_move
                in_set[i] = False
                in_set[i2] = False
                in_set[j] = True
                in_set[k] = True
                selected.remove(i)
                selected.remove(i2)
                selected.append(j)
                selected.append(k)
                total += nums[j] + nums[k] - nums[i] - nums[i2]
                improved = True
                continue

            # 2-for-3-swap: remove i1, i2, add j, k, m
            best_gain = 0
            best_move = None
            sel_list = list(selected)
            for idx1, i in enumerate(sel_list):
                vi = nums[i]
                for idx2, i2 in enumerate(sel_list):
                    if idx2 <= idx1:
                        continue
                    vi2 = nums[i2]
                    base = total - vi - vi2
                    for j in range(n):
                        if in_set[j]:
                            continue
                        vj = nums[j]
                        if base + vj >= lcm_val:
                            continue
                        for k in range(j + 1, n):
                            if in_set[k]:
                                continue
                            vk = nums[k]
                            base2 = base + vj + vk
                            if base2 >= lcm_val:
                                continue
                            for m in range(k + 1, n):
                                if in_set[m]:
                                    continue
                                vm = nums[m]
                                new_total = base2 + vm
                                if new_total < lcm_val:
                                    gain = vj + vk + vm - vi - vi2
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_move = (i, i2, j, k, m)
            if best_move:
                i, i2, j, k, m = best_move
                in_set[i] = False
                in_set[i2] = False
                in_set[j] = True
                in_set[k] = True
                in_set[m] = True
                selected.remove(i)
                selected.remove(i2)
                selected.append(j)
                selected.append(k)
                selected.append(m)
                total += nums[j] + nums[k] + nums[m] - nums[i] - nums[i2]
                improved = True

        return selected, total

    def polish(selected, total, time_limit):
        """Final high-precision polish: try all single additions and best swaps."""
        in_set = [False] * n
        for i in selected:
            in_set[i] = True

        # Try all possible single additions
        for j in range(n):
            if not in_set[j] and total + nums[j] < lcm_val:
                selected.append(j)
                in_set[j] = True
                total += nums[j]

        # Try best 1-swap
        best_gain = 0
        best_move = None
        for i in selected:
            vi = nums[i]
            for j in range(n):
                if not in_set[j]:
                    vj = nums[j]
                    gain = vj - vi
                    if gain > 0 and total - vi + vj < lcm_val:
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (i, j)
        if best_move:
            i, j = best_move
            in_set[i] = False
            in_set[j] = True
            selected.remove(i)
            selected.append(j)
            total += nums[j] - nums[i]

        # Try best 1-for-2-swap
        best_gain = 0
        best_move = None
        for i in selected:
            vi = nums[i]
            base = total - vi
            for j in range(n):
                if not in_set[j]:
                    vj = nums[j]
                    if base + vj >= lcm_val:
                        continue
                    for k in range(j + 1, n):
                        if not in_set[k]:
                            vk = nums[k]
                            if base + vj + vk < lcm_val:
                                gain = vj + vk - vi
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = (i, j, k)
        if best_move:
            i, j, k = best_move
            in_set[i] = False
            in_set[j] = True
            in_set[k] = True
            selected.remove(i)
            selected.append(j)
            selected.append(k)
            total += nums[j] + nums[k] - nums[i]

        return selected, total

    # Best result tracking
    best_indices = None
    best_sum = 0

    # First try greedy with sorted order
    sel, tot = greedy_init(sorted_indices)
    sel, tot = local_search(sel, tot)
    best_indices = sel
    best_sum = tot

    # Random restarts with shuffled orderings
    random.seed(42)
    while time.time() < deadline - 5.0:
        perm = list(range(n))
        random.shuffle(perm)

        selected, total = greedy_init(perm)
        selected, total = local_search(selected, total)

        if total > best_sum:
            best_indices = selected
            best_sum = total

    # Final polish on the best solution found
    if time.time() < deadline - 2.0 and best_indices:
        best_indices, best_sum = polish(best_indices, best_sum, deadline - 1.0)

    # Fallback: if best_indices is empty, use a single element
    if not best_indices:
        best_indices = [0]  # index 0 corresponds to denominator 2

    return best_indices
# EVOLVE-BLOCK-END
