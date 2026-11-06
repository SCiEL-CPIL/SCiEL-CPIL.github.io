import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin for 90s kill limit
    
    n = 256
    weights = np.array([1.0 / i for i in range(1, n + 1)])
    
    def can_add(S_set, x):
        """Check if adding x to S_set creates a 3-AP"""
        for y in S_set:
            z = 2 * x - y
            if z in S_set:
                return False
            z = 2 * y - x
            if z in S_set:
                return False
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if z in S_set:
                    return False
        return True

    def can_add_fast(S_set, x):
        """Faster check using sorted list"""
        for y in S_set:
            z = 2 * x - y
            if z in S_set:
                return False
            z = 2 * y - x
            if z in S_set:
                return False
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if z in S_set:
                    return False
        return True

    def greedy_init(order):
        """Greedy: add elements in given order if they don't conflict"""
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    def local_search_1swap(S_set, max_iters=5000, deadline=None):
        """1-swap local search: try replacing one element with another"""
        S_list = sorted(S_set)
        current_weight = sum(weights[x - 1] for x in S_list)
        
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
            improved = False
            for i, x in enumerate(S_list):
                S_temp = set(S_list)
                S_temp.remove(x)
                for y in range(1, n + 1):
                    if y in S_temp:
                        continue
                    if can_add(S_temp, y):
                        new_weight = current_weight - weights[x - 1] + weights[y - 1]
                        if new_weight > current_weight + 1e-12:
                            S_temp.add(y)
                            S_list = sorted(S_temp)
                            current_weight = new_weight
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return S_list, current_weight

    def local_search_2swap(S_set, max_iters=2000, deadline=None):
        """2-swap local search: try replacing two elements with two others"""
        S_list = sorted(S_set)
        current_weight = sum(weights[x - 1] for x in S_list)
        all_elements = set(range(1, n + 1))
        
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
            improved = False
            # Try all pairs to remove
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    x1, x2 = S_list[i], S_list[j]
                    S_temp = set(S_list)
                    S_temp.remove(x1)
                    S_temp.remove(x2)
                    
                    # Try all pairs to add
                    candidates = sorted([y for y in all_elements if y not in S_temp],
                                        key=lambda y: -weights[y - 1])
                    found = False
                    for a_idx in range(len(candidates)):
                        y1 = candidates[a_idx]
                        if not can_add(S_temp, y1):
                            continue
                        S_temp1 = S_temp | {y1}
                        for b_idx in range(a_idx + 1, len(candidates)):
                            y2 = candidates[b_idx]
                            if not can_add(S_temp1, y2):
                                continue
                            new_weight = current_weight - weights[x1 - 1] - weights[x2 - 1] + weights[y1 - 1] + weights[y2 - 1]
                            if new_weight > current_weight + 1e-12:
                                S_temp1.add(y2)
                                S_list = sorted(S_temp1)
                                current_weight = new_weight
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if improved:
                    break
            if not improved:
                break
        return S_list, current_weight

    def simulated_annealing(S_set, deadline=None, max_iters=10000):
        """Simulated annealing with random kicks to escape local optima"""
        S_list = sorted(S_set)
        current_weight = sum(weights[x - 1] for x in S_list)
        best_weight = current_weight
        best_list = list(S_list)
        
        T = 0.05  # Initial temperature
        cooling = 0.9995
        all_elements = set(range(1, n + 1))
        
        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break
            
            # Random move: pick random element to remove, random element to add
            if len(S_list) > 1:
                idx = np.random.randint(len(S_list))
                x = S_list[idx]
                S_temp = set(S_list)
                S_temp.remove(x)
                
                candidates = [y for y in all_elements if y not in S_temp]
                if not candidates:
                    continue
                y = candidates[np.random.randint(len(candidates))]
                
                if can_add(S_temp, y):
                    S_temp.add(y)
                    new_weight = sum(weights[z - 1] for z in S_temp)
                    delta = new_weight - current_weight
                    if delta > 0 or np.random.random() < np.exp(delta / T):
                        S_list = sorted(S_temp)
                        current_weight = new_weight
                        if current_weight > best_weight:
                            best_weight = current_weight
                            best_list = list(S_list)
            
            T *= cooling
            if T < 1e-6:
                break
        
        return best_list, best_weight

    # Strategy: Try multiple greedy orderings and local search from each
    best_set = set()
    best_weight = 0.0
    
    # Ordering 1: by weight descending (natural greedy)
    order1 = sorted(range(1, n + 1), key=lambda x: -weights[x - 1])
    S1 = greedy_init(order1)
    S1_list, w1 = local_search_1swap(S1, max_iters=3000, deadline=start_time + limit - 5)
    if w1 > best_weight:
        best_weight = w1
        best_set = set(S1_list)
    
    # Ordering 2: random shuffles
    np.random.seed(42)
    for trial in range(5):
        if time.time() > start_time + limit - 10:
            break
        order = list(range(1, n + 1))
        np.random.shuffle(order)
        S_t = greedy_init(order)
        S_t_list, w_t = local_search_1swap(S_t, max_iters=2000, deadline=start_time + limit - 5)
        if w_t > best_weight:
            best_weight = w_t
            best_set = set(S_t_list)
    
    # Additional refinement: try to add more elements or swap for better ones
    # Multiple passes of local search
    for _ in range(3):
        if time.time() > start_time + limit - 3:
            break
        best_list, best_weight = local_search_1swap(list(best_set), max_iters=5000, deadline=start_time + limit - 2)
    
    # 2-swap refinement
    if time.time() < start_time + limit - 3:
        best_list, best_weight = local_search_2swap(list(best_set), max_iters=1500, deadline=start_time + limit - 2)
    
    # Simulated annealing refinement
    if time.time() < start_time + limit - 2:
        sa_list, sa_weight = simulated_annealing(list(best_set), deadline=start_time + limit - 1, max_iters=5000)
        if sa_weight > best_weight:
            best_weight = sa_weight
            best_set = set(sa_list)
    
    if best_set:
        return sorted(best_set)
    else:
        return [1]
