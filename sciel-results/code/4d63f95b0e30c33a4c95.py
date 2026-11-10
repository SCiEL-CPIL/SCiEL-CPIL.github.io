import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    n = 192
    max_size = 48
    
    def can_add(S, x):
        if x in S:
            return False
        S_list = list(S)
        
        # Check if x divides sum of 3 in S
        for b, c, d in combinations(S_list, 3):
            if (b + c + d) % x == 0:
                return False
        
        # Check if x divides sum of 2 in S (for sums x+b+c)
        for b, c in combinations(S_list, 2):
            if (b + c) % x == 0:
                return False
        
        # Check if any a in S divides x + b + c
        for a in S_list:
            for b, c in combinations(S_list, 2):
                if (x + b + c) % a == 0:
                    return False
                    
        return True

    def is_valid_set(S):
        """Full verification, used for safety."""
        S_list = list(S)
        for a in S_list:
            others = [v for v in S_list if v != a]
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True

    best_set = set()
    best_size = 0
    
    # Initial Greedy
    candidates = list(range(n, 0, -1))
    random.shuffle(candidates)
    current_set = []
    for x in candidates:
        if len(current_set) >= max_size:
            break
        if can_add(current_set, x):
            current_set.append(x)
            
    if len(current_set) > best_size:
        best_set = set(current_set)
        best_size = len(best_set)
        
    all_numbers = set(range(1, n + 1))
    
    while time.time() - start_time < deadline:
        # Try to add
        added = False
        for x in list(all_numbers - best_set):
            if len(best_set) >= max_size:
                break
            if can_add(best_set, x):
                best_set.add(x)
                added = True
                break
                
        if added:
            if len(best_set) > best_size:
                best_size = len(best_set)
            continue
        
        # Try to swap 1 for 2
        found_swap = False
        # Randomly pick a to remove
        if best_set:
            a = random.choice(list(best_set))
            temp_set = best_set - {a}
            candidates_add = list(all_numbers - temp_set)
            random.shuffle(candidates_add)
            
            for b in candidates_add:
                if can_add(temp_set, b):
                    new_temp = temp_set | {b}
                    for c in candidates_add:
                        if c == b:
                            continue
                        if can_add(new_temp, c):
                            new_set = new_temp | {c}
                            if is_valid_set(new_set):
                                best_set = new_set
                                found_swap = True
                                break
                    if found_swap:
                        break
                            
        if found_swap:
            if len(best_set) > best_size:
                best_size = len(best_set)
            continue
            
        # If stuck, restart greedy
        current_set = []
        candidates = list(range(n, 0, -1))
        random.shuffle(candidates)
        for x in candidates:
            if len(current_set) >= max_size:
                break
            if can_add(current_set, x):
                current_set.append(x)
                
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(best_set)
            
    return list(best_set)
