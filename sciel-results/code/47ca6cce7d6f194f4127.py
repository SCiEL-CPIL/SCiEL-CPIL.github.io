import time
import random
import math

def score(U):
    if not U:
        return 1.0
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
        U = sorted(set(U))
    
    max_u = max(U)
    if max_u == 0:
        return 1.0
        
    sum_set = set()
    diff_set = set()
    
    n = len(U)
    for i in range(n):
        ui = U[i]
        for j in range(n):
            uj = U[j]
            sum_set.add(ui + uj)
            diff_set.add(ui - uj)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 1.0
        
    ratio = size_diff / size_sum
    log_ratio = math.log(ratio)
    log_max = math.log(2 * max_u + 1)
    
    if log_max == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_max

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Multiple restarts with different initializations
    best_global_U = [0, 1, 3]
    best_global_score = score(best_global_U)
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Initialize
        strategy = random.choice(['geo', 'random'])
        if strategy == 'geo':
            base = random.choice([2, 3, 4, 5, 6])
            size = random.randint(10, 30)
            U = [0]
            val = 1
            for _ in range(size - 1):
                U.append(val)
                val *= base
        else:
            # Random set with small diameter
            max_val = random.randint(10, 100)
            size = random.randint(10, 30)
            U = [0] + random.sample(range(1, max_val + 1), size - 1)
            U.sort()
            
        current_U = list(U)
        current_score = score(current_U)
        
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_U = current_U
            
        # Simulated Annealing
        temperature = 0.1
        cooling = 0.99
        min_temp = 0.001
        
        while time.time() < deadline and temperature > min_temp:
            # Generate neighbor
            move_type = random.choice(['add', 'remove', 'shift'])
            neighbor = list(current_U)
            
            if move_type == 'add' and len(neighbor) < 100:
                max_val = max(neighbor)
                new_val = random.randint(1, max(1, int(3 * max_val)))
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
            elif move_type == 'remove' and len(neighbor) > 5:
                non_zero = [x for x in neighbor if x != 0]
                if non_zero:
                    remove_val = random.choice(non_zero)
                    neighbor.remove(remove_val)
            elif move_type == 'shift' and len(neighbor) > 1:
                non_zero = [x for x in neighbor if x != 0]
                if non_zero:
                    idx = random.randint(0, len(non_zero) - 1)
                    old_val = non_zero[idx]
                    delta = random.randint(-10, 10)
                    new_val = old_val + delta
                    if new_val >= 0 and new_val not in neighbor:
                        neighbor.remove(old_val)
                        neighbor.append(new_val)
                        neighbor.sort()
            
            if neighbor == current_U:
                continue
                
            new_score = score(neighbor)
            delta_score = new_score - current_score
            
            if delta_score > 0 or random.random() < math.exp(delta_score / temperature):
                current_U = neighbor
                current_score = new_score
                
                if current_score > best_global_score:
                    best_global_score = current_score
                    best_global_U = current_U
                    
            temperature *= cooling
            
    return best_global_U
