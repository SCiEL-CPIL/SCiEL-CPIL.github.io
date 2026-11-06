import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [1.0 / d for d in denominators]
    
    best_deficit_float = float('inf')
    best_indices = []
    best_deficit_frac = Fraction(1) # 1 - 0
    
    def try_update(indices):
        nonlocal best_deficit_float, best_indices, best_deficit_frac
        if not indices:
            return
        # Quick float check
        s_float = sum(reciprocals[i] for i in indices)
        if s_float >= 1.0:
            return
        deficit_float = 1.0 - s_float
        if deficit_float < best_deficit_float:
            # Verify with Fraction
            s_frac = Fraction(0)
            for i in indices:
                s_frac += Fraction(1, denominators[i])
            if s_frac < 1:
                deficit_frac = 1 - s_frac
                if deficit_frac < best_deficit_frac:
                    best_deficit_frac = deficit_frac
                    best_deficit_float = float(deficit_frac)
                    best_indices = list(indices)

    # Initial Candidates
    initial_sets = []
    
    # 1. Greedy small denominators
    curr = []
    s = 0.0
    for i in range(n):
        if s + reciprocals[i] < 1.0:
            curr.append(i)
            s += reciprocals[i]
    initial_sets.append(set(curr))
    
    # 2. Greedy large denominators
    curr = []
    s = 0.0
    for i in range(n-1, -1, -1):
        if s + reciprocals[i] < 1.0:
            curr.append(i)
            s += reciprocals[i]
    initial_sets.append(set(curr))
    
    # 3. Known good: {2, 3, 7, 43}
    initial_sets.append({0, 1, 5, 41})
    
    # 4. Random
    for _ in range(20):
        size = random.randint(3, 15)
        rand_set = set(random.sample(range(n), size))
        s = sum(reciprocals[i] for i in rand_set)
        if s < 1.0:
            initial_sets.append(rand_set)
        else:
            while s >= 1.0 and rand_set:
                max_idx = max(rand_set, key=lambda x: reciprocals[x])
                rand_set.remove(max_idx)
                s -= reciprocals[max_idx]
            if rand_set:
                initial_sets.append(rand_set)
                
    # Local Search Function
    def local_search(start_set, time_limit):
        current_set = set(start_set)
        current_sum = sum(reciprocals[i] for i in current_set)
        
        if current_sum >= 1.0:
            while current_sum >= 1.0 and current_set:
                max_idx = max(current_set, key=lambda x: reciprocals[x])
                current_set.remove(max_idx)
                current_sum -= reciprocals[max_idx]
            if not current_set:
                return None, None
            
        best_local_sum = current_sum
        best_local_set = set(current_set)
        
        no_improve_count = 0
        max_no_improve = 20
        
        while time.time() < time_limit:
            improved = False
            best_move_sum = -1.0
            best_move_set = None
            
            in_list = list(current_set)
            out_list = [i for i in range(n) if i not in current_set]
            
            # Swaps
            for i in in_list:
                limit = 1.0 / (1.0 - current_sum + reciprocals[i])
                best_j = -1
                best_j_rec = -1.0
                for j in out_list:
                    if denominators[j] < denominators[i]:
                        if denominators[j] > limit:
                            if reciprocals[j] > best_j_rec:
                                best_j_rec = reciprocals[j]
                                best_j = j
                
                if best_j != -1:
                    new_sum = current_sum - reciprocals[i] + best_j_rec
                    if new_sum > best_move_sum:
                        best_move_sum = new_sum
                        best_move_set = current_set.copy()
                        best_move_set.remove(i)
                        best_move_set.add(best_j)
            
            # Adds
            limit_add = 1.0 / (1.0 - current_sum)
            for j in out_list:
                if denominators[j] > limit_add:
                    new_sum = current_sum + reciprocals[j]
                    if new_sum > best_move_sum:
                        best_move_sum = new_sum
                        best_move_set = current_set.copy()
                        best_move_set.add(j)
            
            if best_move_set is not None:
                current_set = best_move_set
                current_sum = best_move_sum
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            if current_sum > best_local_sum:
                best_local_sum = current_sum
                best_local_set = set(current_set)
                
        return best_local_set, best_local_sum

    # Run search
    for s in initial_sets:
        if time.time() > deadline:
            break
        best_set, best_sum = local_search(s, deadline)
        if best_set:
            try_update(best_set)
            
    # Random restarts
    while time.time() < deadline - 2.0:
        size = random.randint(3, 10)
        rand_set = set(random.sample(range(n), size))
        s = sum(reciprocals[i] for i in rand_set)
        if s >= 1.0:
            while s >= 1.0 and rand_set:
                max_idx = max(rand_set, key=lambda x: reciprocals[x])
                rand_set.remove(max_idx)
                s -= reciprocals[max_idx]
        if not rand_set:
            continue
        best_set, best_sum = local_search(rand_set, deadline)
        if best_set:
            try_update(best_set)
            
    return best_indices
