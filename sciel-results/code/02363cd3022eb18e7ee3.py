import time
import random
import math

def score(U):
    if not U:
        return 1.0
    U = sorted(set(U))
    if 0 not in U:
        # Task requires containing 0
        U = [0] + U
        U = sorted(set(U))
    
    max_u = max(U)
    if max_u == 0:
        return 1.0
        
    # Check constraint
    if max_u > 5000000:
        return -1000.0
        
    # Calculate sumset and difference set sizes
    # Using sets for efficiency
    sum_set = set()
    diff_set = set()
    
    n = len(U)
    # Optimize by iterating
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
    if ratio <= 0:
        return -1000.0
        
    log_ratio = math.log(ratio)
    log_max = math.log(2 * max_u + 1)
    
    if log_max == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_max

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    MAX_VAL = 5000000
    
    # Helper to generate valid geometric progression
    def gen_geo(base, max_elements):
        U = [0]
        current_val = 1
        for _ in range(max_elements - 1):
            if current_val > MAX_VAL:
                break
            U.append(current_val)
            current_val *= base
        return U

    # Initialize with a geometric progression
    # Base 2 allows more elements (~22), Base 3 fewer (~14)
    # Let's start with base 2, size 20
    base = 2
    initial_size = 20
    U = gen_geo(base, initial_size)
    
    best_U = list(U)
    best_score = score(best_U)
    
    # Local Search
    # We will try random moves: add, remove, shift
    # We need to be careful with the size of U to keep score calculation fast.
    # If U gets too large, score calculation becomes slow.
    # Let's cap the size of U to, say, 100.
    max_size = 100
    
    no_improve_count = 0
    step = 0
    
    while time.time() < deadline:
        step += 1
        current_U = list(best_U)
        current_score = best_score
        
        # Choose a random move
        move_type = random.choice(['add', 'remove', 'shift'])
        
        if move_type == 'add' and len(current_U) < max_size:
            # Add a random element
            # Range should be around the current max but capped
            max_val = max(current_U)
            # Try adding something in [1, min(3*max_val, MAX_VAL)]
            upper_bound = min(int(3 * max_val), MAX_VAL)
            if upper_bound >= 1:
                new_val = random.randint(1, upper_bound)
                if new_val not in current_U:
                    current_U.append(new_val)
                    current_U.sort()
                    
        elif move_type == 'remove' and len(current_U) > 5:
            # Remove a random element (not 0)
            non_zero = [x for x in current_U if x != 0]
            if non_zero:
                remove_val = random.choice(non_zero)
                current_U.remove(remove_val)
                
        elif move_type == 'shift':
            # Shift a random element
            non_zero = [x for x in current_U if x != 0]
            if non_zero:
                idx = random.randint(0, len(non_zero) - 1)
                old_val = non_zero[idx]
                # Shift by a small amount
                delta = random.randint(-5, 5)
                new_val = old_val + delta
                if 0 <= new_val <= MAX_VAL and new_val not in current_U:
                    current_U.remove(old_val)
                    current_U.append(new_val)
                    current_U.sort()
        
        if len(current_U) != len(best_U) or current_U != best_U:
            new_score = score(current_U)
            if new_score > best_score:
                best_score = new_score
                best_U = current_U
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If stuck, maybe restart with a different base or random set?
                if no_improve_count > 200:
                    # Reset to a new geometric progression
                    base = random.choice([2, 3, 4])
                    # Calculate max safe size for this base
                    # log_base(MAX_VAL)
                    safe_size = int(math.log(MAX_VAL, base)) + 1
                    initial_size = random.randint(10, min(30, safe_size))
                    U = gen_geo(base, initial_size)
                    best_U = list(U)
                    best_score = score(best_U)
                    no_improve_count = 0
                    current_U = list(best_U)

    return best_U
