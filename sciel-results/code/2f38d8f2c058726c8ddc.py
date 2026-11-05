import random
import time

def construct():
    N = 1000
    max_sum = 2 * N
    start_time = time.time()
    deadline = start_time + 145
    
    best_size = 0
    best_set = []
    
    def is_valid_addition(x, counts, chosen_set):
        if counts[2*x] >= 2:
            return False
        for y in chosen_set:
            if counts[x + y] >= 2:
                return False
        return True

    def add_element(x, counts, chosen_list, chosen_set):
        counts[2*x] += 1
        for y in chosen_list:
            counts[x + y] += 1
        chosen_list.append(x)
        chosen_set.add(x)

    def remove_element(x, counts, chosen_list, chosen_set):
        chosen_list.remove(x)
        chosen_set.remove(x)
        counts[2*x] -= 1
        for y in chosen_list:
            counts[x + y] -= 1

    for restart in range(1000): # More restarts, faster checks
        if time.time() > deadline:
            break
            
        counts = [0] * (max_sum + 1)
        chosen_list = []
        chosen_set = set()
        
        # Greedy Phase
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(x, counts, chosen_set):
                add_element(x, counts, chosen_list, chosen_set)
        
        # Local Search Phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add a valid element
            # Full scan to ensure we don't miss any
            added = False
            for v in range(1, N + 1):
                if v not in chosen_set:
                    if is_valid_addition(v, counts, chosen_set):
                        add_element(v, counts, chosen_list, chosen_set)
                        improved = True
                        added = True
                        break
                        
            if not added:
                # 2. Try swaps
                # Try a few random swaps
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    if not chosen_list:
                        break
                    u = random.choice(chosen_list)
                    remove_element(u, counts, chosen_list, chosen_set)
                    
                    # Try to add a random v
                    found_swap = False
                    for _ in range(10): # 10 random trials
                        v = random.randint(1, N)
                        if v in chosen_set:
                            continue
                        if is_valid_addition(v, counts, chosen_set):
                            add_element(v, counts, chosen_list, chosen_set)
                            found_swap = True
                            improved = True
                            break
                    
                    if not found_swap:
                        add_element(u, counts, chosen_list, chosen_set)

        if len(chosen_list) > best_size:
            best_size = len(chosen_list)
            best_set = list(chosen_list)

    return best_set
