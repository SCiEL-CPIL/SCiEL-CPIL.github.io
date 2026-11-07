import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 256
    
    # Precompute prime factors for each number 1..256
    def prime_factors(x):
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors
    
    # factors[i] = set of prime factors of i (for i in 1..256)
    # factors[1] = empty set
    factors = [set() for _ in range(n + 1)]
    for i in range(2, n + 1):
        factors[i] = prime_factors(i)
    
    # Check compatibility: two numbers are compatible if their prime factor sets are disjoint
    def compatible(a, b):
        if a == b:
            return True
        return len(factors[a] & factors[b]) == 0
    
    # Greedy initial solution: sort by value descending, add if compatible with all current
    def greedy_solution():
        selected = [1]  # 1 is always compatible
        for candidate in range(n, 1, -1):
            if all(compatible(candidate, s) for s in selected):
                selected.append(candidate)
        return selected
    
    # Local search: try to improve by removing one element and adding one or more
    def local_search(selected, time_limit):
        selected_set = set(selected)
        current_sum = sum(selected)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try single swaps: remove i, add j if compatible with all others and improves sum
            for i in list(selected):
                others = [s for s in selected if s != i]
                others_set = set(others)
                for j in range(n, 1, -1):
                    if j in selected_set:
                        continue
                    if all(compatible(j, s) for s in others):
                        new_sum = current_sum - i + j
                        if new_sum > current_sum:
                            selected = others + [j]
                            selected_set = set(selected)
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
            # Try double swaps: remove i, j, add k, l if compatible
            # This is O(n^4) so only do it if time permits
            if not improved and time.time() < time_limit - 1.0:
                for i_idx in range(len(selected)):
                    for j_idx in range(i_idx + 1, len(selected)):
                        i, j = selected[i_idx], selected[j_idx]
                        others = [s for k, s in enumerate(selected) if k != i_idx and k != j_idx]
                        for k in range(n, 1, -1):
                            if k in selected_set:
                                continue
                            if not all(compatible(k, s) for s in others):
                                continue
                            for l in range(n, 1, -1):
                                if l in selected_set or l == k:
                                    continue
                                if not all(compatible(l, s) for s in others) or not compatible(k, l):
                                    continue
                                new_sum = current_sum - i - j + k + l
                                if new_sum > current_sum:
                                    selected = others + [k, l]
                                    selected_set = set(selected)
                                    current_sum = new_sum
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        return selected
    
    # Start with greedy
    best = greedy_solution()
    
    # Local search
    best = local_search(best, deadline)
    
    # Convert to list of Python ints
    result = [int(x) for x in best]
    return result
