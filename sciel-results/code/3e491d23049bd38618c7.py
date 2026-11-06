import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    N = 1000
    best_set = []
    best_size = 0
    
    def is_valid_addition(x, chosen, counts):
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        if counts.get(2 * x, 0) >= 2:
            return False
        return True

    def add_to_set(x, chosen, counts):
        for y in chosen:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        counts[2 * x] = counts.get(2 * x, 0) + 1
        chosen.append(x)

    def remove_from_set(x, chosen, counts):
        for y in chosen:
            if y == x: continue
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        counts[2 * x] -= 1
        if counts[2 * x] == 0:
            del counts[2 * x]
        chosen.remove(x)

    # Initial deterministic greedy to get a baseline
    chosen = []
    counts = {}
    for x in range(1, N + 1):
        if is_valid_addition(x, chosen, counts):
            add_to_set(x, chosen, counts)
    best_set = chosen.copy()
    best_size = len(chosen)
    
    # Local Search / Hill Climbing
    current_set = best_set.copy()
    current_counts = {}
    for x in current_set:
        for y in current_set:
            if x <= y:
                s = x + y
                current_counts[s] = current_counts.get(s, 0) + 1
                
    all_elements = list(range(1, N + 1))
    no_improvement_count = 0
    max_no_improvement = 5000
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            random.shuffle(all_elements)
            current_set = []
            current_counts = {}
            for x in all_elements:
                if is_valid_addition(x, current_set, current_counts):
                    add_to_set(x, current_set, current_counts)
            no_improvement_count = 0
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            continue

        x = random.randint(1, N)
        if x not in current_set:
            if is_valid_addition(x, current_set, current_counts):
                add_to_set(x, current_set, current_counts)
                no_improvement_count = 0
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                continue
            else:
                no_improvement_count += 1
        
        if current_set:
            y = random.choice(current_set)
            remove_from_set(y, current_set, current_counts)
            
            z = random.randint(1, N)
            attempts = 0
            while z in current_set and attempts < 10:
                z = random.randint(1, N)
                attempts += 1
            
            if z not in current_set:
                if is_valid_addition(z, current_set, current_counts):
                    add_to_set(z, current_set, current_counts)
                    no_improvement_count = max(0, no_improvement_count - 1)
                else:
                    add_to_set(y, current_set, current_counts)
                    no_improvement_count += 1
            else:
                add_to_set(y, current_set, current_counts)
                no_improvement_count += 1

    return best_set
