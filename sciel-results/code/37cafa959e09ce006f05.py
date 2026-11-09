import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    denominators = list(range(2, 49))
    n = len(denominators)

    # LCM of 2..48
    lcm_val = 1
    for d in range(2, 49):
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)

    # Precompute numerator for each denominator: 1/d = (lcm/d) / lcm
    nums = [lcm_val // d for d in denominators]

    # We want to maximize sum(nums[i] for i in selected) subject to sum < lcm_val
    
    def greedy_init(order):
        selected = []
        total = 0
        for i in order:
            if total + nums[i] < lcm_val:
                selected.append(i)
                total += nums[i]
        return selected, total

    def local_search(selected, total):
        in_set = [False] * n
        for i in selected:
            in_set[i] = True
            
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            # 1-swap
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
                                    best_move = (i, j, 1)
            if best_move:
                i, j, _ = best_move
                in_set[i] = False
                in_set[j] = True
                selected.remove(i)
                selected.append(j)
                total += nums[j] - nums[i]
                improved = True
                continue

            # 1-for-2 swap
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
                                        best_move = (i, j, k, 2)
            if best_move:
                i, j, k, _ = best_move
                in_set[i] = False
                in_set[j] = True
                in_set[k] = True
                selected.remove(i)
                selected.append(j)
                selected.append(k)
                total += nums[j] + nums[k] - nums[i]
                improved = True
                continue

            # 2-for-1 swap
            best_gain = 0
            best_move = None
            # Iterate over pairs in selected
            # To avoid O(N^2) overhead of list comprehension, use indices
            sel_len = len(selected)
            for a_idx in range(sel_len):
                i = selected[a_idx]
                vi = nums[i]
                for b_idx in range(a_idx + 1, sel_len):
                    j = selected[b_idx]
                    vj = nums[j]
                    base = total - vi - vj
                    for k in range(n):
                        if not in_set[k]:
                            vk = nums[k]
                            new_total = base + vk
                            if new_total < lcm_val:
                                gain = vk - vi - vj
                                if gain > 0:
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_move = (i, j, k, 1)
            if best_move:
                i, j, k, _ = best_move
                in_set[i] = False
                in_set[j] = False
                in_set[k] = True
                selected.remove(i)
                selected.remove(j)
                selected.append(k)
                total += nums[k] - nums[i] - nums[j]
                improved = True
                continue

        return selected, total

    best_indices = None
    best_sum = 0
    
    random.seed(42)
    
    # Initial greedy
    order = list(range(n))
    sel, tot = greedy_init(order)
    sel, tot = local_search(sel, tot)
    best_indices = sel
    best_sum = tot

    # Random restarts
    while time.time() < deadline - 1.0:
        perm = list(range(n))
        random.shuffle(perm)
        sel, tot = greedy_init(perm)
        sel, tot = local_search(sel, tot)
        
        if tot > best_sum:
            best_indices = sel
            best_sum = tot

    return best_indices
