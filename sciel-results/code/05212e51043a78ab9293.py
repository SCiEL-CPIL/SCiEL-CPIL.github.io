import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85
    rng = random.Random(42)
    
    def compute_counts(A_list):
        counts = np.zeros(257, dtype=int)
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            for j in range(i, n):
                s = ai + A_list[j]
                if s <= 256:
                    counts[s] += 1
        return counts

    def evaluate(A_set):
        if not A_set:
            return float('inf'), False
        A_list = sorted(A_set)
        counts = compute_counts(A_list)
        target_counts = counts[128:257]
        if np.any(target_counts == 0):
            return float('inf'), False
        peak = np.max(target_counts)
        return peak, True

    def greedy_repair(A_set):
        current_A = set(A_set)
        while True:
            A_list = sorted(current_A)
            counts = compute_counts(A_list)
            missing = [t for t in range(128, 257) if counts[t] == 0]
            if not missing:
                return current_A
            if time.time() > deadline - 5:
                return None
            
            best_elem = None
            best_gain = 0
            # To speed up, we can sample candidates if set is large, but 129 is small
            for x in range(129):
                if x in current_A:
                    continue
                gain = 0
                for t in missing:
                    a = t - x
                    if 0 <= a <= 128 and a in current_A:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
            
            if best_elem is None or best_gain == 0:
                # Stuck, try random addition
                candidates = [x for x in range(129) if x not in current_A]
                if candidates:
                    best_elem = rng.choice(candidates)
                else:
                    return None
            
            current_A.add(best_elem)
            
        return current_A

    def local_search(A_set, time_limit):
        current_A = set(A_set)
        current_cost, valid = evaluate(current_A)
        if not valid:
            return current_A, float('inf')
            
        best_A = set(current_A)
        best_cost = current_cost
        T = 2.0
        T_min = 0.01
        cooling = 0.9995
        
        while time.time() < time_limit:
            move_type = rng.choice(['remove', 'swap'])
            new_A = set(current_A)
            
            if move_type == 'remove':
                if len(current_A) > 15:
                    x = rng.choice(list(current_A))
                    new_A.remove(x)
                else:
                    move_type = 'swap'
                    if not current_A: continue
                    x_in = rng.choice(list(current_A))
                    candidates = [x for x in range(129) if x not in current_A]
                    if not candidates: continue
                    x_out = rng.choice(candidates)
                    new_A.remove(x_in)
                    new_A.add(x_out)
            else: # swap
                if current_A:
                    x_in = rng.choice(list(current_A))
                    candidates = [x for x in range(129) if x not in current_A]
                    if not candidates: continue
                    x_out = rng.choice(candidates)
                    new_A.remove(x_in)
                    new_A.add(x_out)
            else:
                continue

            new_cost, new_valid = evaluate(new_A)
            
            if not new_valid:
                new_cost = 1000 + current_cost # Penalize
            
            if new_cost < current_cost:
                current_A = new_A
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(current_A)
            else:
                delta = new_cost - current_cost
                if delta < 0:
                    accept = True
                else:
                    accept = rng.random() < np.exp(-delta / T)
                
                if accept:
                    current_A = new_A
                    current_cost = new_cost
            
            T = max(T_min, T * cooling)
            
        return best_A, best_cost

    best_overall_A = None
    best_overall_cost = float('inf')
    
    # 1. Start from Seed
    seed_A = set(range(64, 129))
    A_res, cost_res = local_search(seed_A, deadline - 10)
    if cost_res < best_overall_cost:
        best_overall_cost = cost_res
        best_overall_A = A_res

    # 2. Random restarts
    for i in range(5):
        if time.time() > deadline - 10:
            break
            
        # Random init
        size = rng.randint(20, 40)
        init_A = set(rng.sample(range(129), size))
        
        # Repair
        repaired_A = greedy_repair(init_A)
        if repaired_A is None:
            continue
            
        # Local Search
        A_res, cost_res = local_search(repaired_A, deadline - 10)
        if cost_res < best_overall_cost:
            best_overall_cost = cost_res
            best_overall_A = A_res

    if best_overall_A is None:
        return list(range(64, 129))
        
    return sorted(best_overall_A)

