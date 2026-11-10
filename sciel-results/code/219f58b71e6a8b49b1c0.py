"""Optimized program for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 84.0  # 90s wall-clock, 6s safety margin

    N = 256
    nums = list(range(2, 257))
    n = len(nums)  # 255

    # Precompute conflict: conflict[i][j] = True if lcm(nums[i], nums[j]) <= 256
    conflict = [bytearray(n) for _ in range(n)]
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict[i][j] = 1
                conflict[j][i] = 1

    # Weight: 1/nums[i]
    weights = [1.0 / nums[i] for i in range(n)]

    def total_weight(selected):
        return sum(weights[i] for i in selected)

    def is_valid_add(selected_set, idx):
        """Check if idx can be added to selected_set"""
        for j in selected_set:
            if conflict[idx][j]:
                return False
        return True

    # Greedy initialization: try multiple orderings
    best_set = set()
    best_score = 0.0

    # Ordering 1: by weight descending (smallest nums first)
    order1 = sorted(range(n), key=lambda i: -weights[i])
    # Ordering 2: by number of conflicts ascending (least constrained first)
    conflict_counts = [sum(conflict[i]) for i in range(n)]
    order2 = sorted(range(n), key=lambda i: (conflict_counts[i], -weights[i]))
    # Ordering 3: by weight/conflict_count ratio
    order3 = sorted(range(n), key=lambda i: -(weights[i] / (conflict_counts[i] + 1)))
    # Ordering 4: random-ish deterministic shuffle based on index
    order4 = sorted(range(n), key=lambda i: (i * 7919) % n)
    # Ordering 5: weight * (1 + 1/conflict_count) - favors low conflict, high weight
    order5 = sorted(range(n), key=lambda i: -(weights[i] * (1 + 1.0 / (conflict_counts[i] + 1))))
    # Ordering 6: smallest numbers that are "isolated" (fewest conflicts)
    order6 = sorted(range(n), key=lambda i: (conflict_counts[i] // 10, nums[i]))

    for order in [order1, order2, order3, order4, order5, order6]:
        selected = set()
        for i in order:
            if is_valid_add(selected, i):
                selected.add(i)
        score = total_weight(selected)
        if score > best_score:
            best_score = score
            best_set = set(selected)

    # Local search: try swaps and adds
    selected = set(best_set)
    current_score = best_score

    # Phase 1: Try adding elements that fit
    improved = True
    while improved and time.time() - start_time < DEADLINE - 5:
        improved = False
        for j in range(n):
            if j in selected:
                continue
            if is_valid_add(selected, j):
                selected.add(j)
                current_score += weights[j]
                improved = True
                break

    # Phase 2: Try single swaps (remove i, add j) - use best improvement
    improved = True
    while improved and time.time() - start_time < DEADLINE - 5:
        improved = False
        best_swap_gain = 0.0
        best_swap_i = -1
        best_swap_j = -1
        for i in list(selected):
            for j in range(n):
                if j in selected:
                    continue
                # Check if j can be added after removing i
                can_add = True
                for k in selected:
                    if k != i and conflict[j][k]:
                        can_add = False
                        break
                if can_add:
                    gain = weights[j] - weights[i]
                    if gain > best_swap_gain + 1e-15:
                        best_swap_gain = gain
                        best_swap_i = i
                        best_swap_j = j
        if best_swap_i >= 0:
            selected.remove(best_swap_i)
            selected.add(best_swap_j)
            current_score += best_swap_gain
            improved = True

    # Phase 3: Try 2-for-1 swaps (remove two, add one) - use float pruning
    if time.time() - start_time < DEADLINE - 10:
        improved = True
        while improved and time.time() - start_time < DEADLINE - 10:
            improved = False
            sel_list = list(selected)
            best_21_gain = 0.0
            best_21_i1 = -1
            best_21_i2 = -1
            best_21_j = -1
            for idx_i1 in range(len(sel_list)):
                i1 = sel_list[idx_i1]
                for idx_i2 in range(idx_i1 + 1, len(sel_list)):
                    i2 = sel_list[idx_i2]
                    # Try to find best single element to add
                    for j in range(n):
                        if j in selected:
                            continue
                        # Quick float check: if even the best possible gain isn't positive, skip
                        if weights[j] - weights[i1] - weights[i2] <= best_21_gain + 1e-15:
                            continue
                        # Check if j conflicts with selected minus i1, i2
                        can_add = True
                        for k in selected:
                            if k != i1 and k != i2 and conflict[j][k]:
                                can_add = False
                                break
                        if can_add:
                            gain = weights[j] - weights[i1] - weights[i2]
                            if gain > best_21_gain + 1e-15:
                                best_21_gain = gain
                                best_21_i1 = i1
                                best_21_i2 = i2
                                best_21_j = j
            if best_21_i1 >= 0:
                selected.remove(best_21_i1)
                selected.remove(best_21_i2)
                selected.add(best_21_j)
                current_score += best_21_gain
                improved = True

    # Phase 4: Try 1-for-2 swaps (remove one, add two)
    if time.time() - start_time < DEADLINE - 8:
        improved = True
        while improved and time.time() - start_time < DEADLINE - 8:
            improved = False
            best_12_gain = 0.0
            best_12_i = -1
            best_12_j1 = -1
            best_12_j2 = -1
            for i in list(selected):
                # Find candidates that can be added after removing i
                candidates = []
                for j in range(n):
                    if j in selected:
                        continue
                    can_add = True
                    for k in selected:
                        if k != i and conflict[j][k]:
                            can_add = False
                            break
                    if can_add:
                        candidates.append(j)
                if len(candidates) >= 2:
                    # Try pairs from candidates (limited to top 30 by weight)
                    candidates.sort(key=lambda x: -weights[x])
                    candidates = candidates[:30]
                    for idx_j1 in range(len(candidates)):
                        j1 = candidates[idx_j1]
                        for idx_j2 in range(idx_j1 + 1, len(candidates)):
                            j2 = candidates[idx_j2]
                            if conflict[j1][j2]:
                                continue
                            gain = weights[j1] + weights[j2] - weights[i]
                            if gain > best_12_gain + 1e-15:
                                best_12_gain = gain
                                best_12_i = i
                                best_12_j1 = j1
                                best_12_j2 = j2
            if best_12_i >= 0:
                selected.remove(best_12_i)
                selected.add(best_12_j1)
                selected.add(best_12_j2)
                current_score += best_12_gain
                improved = True

    # Phase 5: Try 2-for-2 swaps (remove two, add two) - limited search
    if time.time() - start_time < DEADLINE - 12:
        improved = True
        while improved and time.time() - start_time < DEADLINE - 12:
            improved = False
            sel_list = list(selected)
            best_22_gain = 0.0
            best_22_i1 = -1
            best_22_i2 = -1
            best_22_j1 = -1
            best_22_j2 = -1
            for idx_i1 in range(len(sel_list)):
                i1 = sel_list[idx_i1]
                for idx_i2 in range(idx_i1 + 1, len(sel_list)):
                    i2 = sel_list[idx_i2]
                    # Find candidates that can be added after removing i1, i2
                    candidates = []
                    for j in range(n):
                        if j in selected:
                            continue
                        can_add = True
                        for k in selected:
                            if k != i1 and k != i2 and conflict[j][k]:
                                can_add = False
                                break
                        if can_add:
                            candidates.append(j)
                    if len(candidates) >= 2:
                        # Try pairs from candidates (limited to top 20 by weight)
                        candidates.sort(key=lambda x: -weights[x])
                        candidates = candidates[:20]
                        for idx_j1 in range(len(candidates)):
                            j1 = candidates[idx_j1]
                            for idx_j2 in range(idx_j1 + 1, len(candidates)):
                                j2 = candidates[idx_j2]
                                if conflict[j1][j2]:
                                    continue
                                gain = weights[j1] + weights[j2] - weights[i1] - weights[i2]
                                if gain > best_22_gain + 1e-15:
                                    best_22_gain = gain
                                    best_22_i1 = i1
                                    best_22_i2 = i2
                                    best_22_j1 = j1
                                    best_22_j2 = j2
            if best_22_i1 >= 0:
                selected.remove(best_22_i1)
                selected.remove(best_22_i2)
                selected.add(best_22_j1)
                selected.add(best_22_j2)
                current_score += best_22_gain
                improved = True

    # Final phase: try adding again after swaps
    improved = True
    while improved and time.time() - start_time < DEADLINE - 2:
        improved = False
        for j in range(n):
            if j in selected:
                continue
            if is_valid_add(selected, j):
                selected.add(j)
                current_score += weights[j]
                improved = True
                break

    return [nums[i] for i in selected]
# EVOLVE-BLOCK-END
