"""Optimized program for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # 90s wall-clock, 5s safety margin
    
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
    
    for order in [order1, order2, order3, order4]:
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
    
    # Phase 2: Try single swaps (remove i, add j)
    improved = True
    while improved and time.time() - start_time < DEADLINE - 5:
        improved = False
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
                    new_score = current_score - weights[i] + weights[j]
                    if new_score > current_score + 1e-15:
                        selected.remove(i)
                        selected.add(j)
                        current_score = new_score
                        improved = True
                        break
            if improved:
                break
    
    # Phase 3: Try 2-swaps (remove i1, i2, add j1, j2) - limited search
    if time.time() - start_time < DEADLINE - 10:
        improved = True
        while improved and time.time() - start_time < DEADLINE - 10:
            improved = False
            sel_list = list(selected)
            for idx_i1 in range(len(sel_list)):
                i1 = sel_list[idx_i1]
                for idx_i2 in range(idx_i1 + 1, len(sel_list)):
                    i2 = sel_list[idx_i2]
                    # Try to find two elements to add
                    candidates = []
                    for j in range(n):
                        if j in selected:
                            continue
                        # Check if j conflicts with selected minus i1, i2
                        can_add = True
                        for k in selected:
                            if k != i1 and k != i2 and conflict[j][k]:
                                can_add = False
                                break
                        if can_add:
                            candidates.append(j)
                    
                    if len(candidates) >= 2:
                        # Try all pairs from candidates (limited)
                        for idx_j1 in range(min(len(candidates), 50)):
                            j1 = candidates[idx_j1]
                            for idx_j2 in range(idx_j1 + 1, min(len(candidates), 50)):
                                j2 = candidates[idx_j2]
                                if conflict[j1][j2]:
                                    continue
                                new_score = current_score - weights[i1] - weights[i2] + weights[j1] + weights[j2]
                                if new_score > current_score + 1e-15:
                                    selected.remove(i1)
                                    selected.remove(i2)
                                    selected.add(j1)
                                    selected.add(j2)
                                    current_score = new_score
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
    
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
