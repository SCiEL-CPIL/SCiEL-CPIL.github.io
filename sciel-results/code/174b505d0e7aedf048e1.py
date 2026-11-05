import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    numbers = list(range(2, 257))
    n = len(numbers)
    
    # Precompute conflict sets
    conflict = [set() for _ in range(n)]
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            if math.lcm(a, b) <= N:
                conflict[i].add(j)
                conflict[j].add(i)
    
    weights = [1.0 / x for x in numbers]
    
    def v2(x):
        c = 0
        while x % 2 == 0:
            c += 1
            x //= 2
        return c
    
    v2_vals = [v2(x) for x in numbers]
    
    def local_search(selected, unselected, current_sum):
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for si in sorted(selected, key=lambda x: weights[x]):
                old_val = weights[si]
                best_new = None
                best_gain = 0.0
                selected.remove(si)
                for new_idx in unselected:
                    if not conflict[new_idx].intersection(selected):
                        gain = weights[new_idx] - old_val
                        if gain > best_gain:
                            best_gain = gain
                            best_new = new_idx
                if best_new is not None and best_gain > 1e-15:
                    selected.add(best_new)
                    unselected.remove(best_new)
                    unselected.append(si)
                    current_sum += best_gain
                    improved = True
                else:
                    selected.add(si)
            for i in unselected[:]:
                if not conflict[i].intersection(selected):
                    selected.add(i)
                    unselected.remove(i)
                    current_sum += weights[i]
                    improved = True
        return selected, unselected, current_sum
    
    best_sum = 0.0
    best_set = set()
    
    # Diverse greedy orders
    orders = [
        sorted(range(n), key=lambda i: (-v2_vals[i], -weights[i])),
        sorted(range(n), key=lambda i: -weights[i]),
        sorted(range(n), key=lambda i: (-len(conflict[i]), -weights[i])),
        sorted(range(n), key=lambda i: (v2_vals[i], weights[i])),
        sorted(range(n), key=lambda i: (-weights[i], v2_vals[i])),
    ]
    
    for order in orders:
        if time.time() > deadline - 5.0:
            break
        selected = set()
        for i in order:
            if not conflict[i].intersection(selected):
                selected.add(i)
        unselected = [i for i in range(n) if i not in selected]
        current_sum = sum(weights[i] for i in selected)
        selected, unselected, current_sum = local_search(selected, unselected, current_sum)
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = set(selected)
    
    if best_set:
        return [numbers[i] for i in best_set]
    return [2]
