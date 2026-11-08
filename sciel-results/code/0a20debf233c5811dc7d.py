import math
import random
import time

def score_set(U):
    if not U:
        return 0
    U_set = set(U)
    if 0 not in U_set:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
    return 1 + math.log(ratio) / denom

def construct():
    best_score = 0
    best_set = [0, 1, 3]
    
    start_time = time.time()
    time_limit = 145
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 2, 5],
        [0, 1, 2, 6],
        [0, 1, 2, 7],
        [0, 1, 3, 7],
        [0, 1, 4, 7],
        [0, 1, 2, 4, 7],
        [0, 1, 3, 4, 7],
        [0, 1, 2, 5, 9],
        [0, 1, 3, 7, 11],
    ]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c
    
    current_set = best_set[:]
    current_score = best_score
    
    while time.time() - start_time < time_limit:
        # Random restart
        if random.random() < 0.05:
            current_set = best_set[:]
            current_score = best_score
        
        # Generate move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        max_val = max(current_set) + 10 if current_set else 20
        if max_val > 200:
            max_val = 200
            
        new_set = current_set[:]
        
        if move_type == 'add':
            # Bias towards small numbers
            if random.random() < 0.5:
                x = random.randint(1, 10)
            else:
                x = random.randint(1, max_val)
            if x not in new_set:
                new_set.append(x)
                new_set.sort()
        elif move_type == 'remove':
            if len(new_set) > 1:
                idx = random.randint(1, len(new_set) - 1)
                new_set.pop(idx)
        elif move_type == 'swap':
            if len(new_set) > 1:
                idx = random.randint(1, len(new_set) - 1)
                if random.random() < 0.5:
                    new_x = random.randint(1, 10)
                else:
                    new_x = random.randint(1, max_val)
                if new_x not in new_set:
                    new_set[idx] = new_x
                    new_set.sort()
        
        if new_set == current_set:
            continue
            
        new_score = score_set(new_set)
        
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            elapsed = time.time() - start_time
            T = max(0.01, 1.0 - elapsed / time_limit)
            if random.random() < math.exp((new_score - current_score) / T):
                current_set = new_set
                current_score = new_score

    return best_set
