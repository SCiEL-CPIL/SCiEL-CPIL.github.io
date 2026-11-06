import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 1000
    max_sum = 2 * N
    count_arr = [0] * (max_sum + 1)
    
    # 1. Greedy Initialization
    current_set_list = []
    current_set_set = set()
    
    for x in range(1, N + 1):
        # Check validity
        valid = True
        s = 2 * x
        if count_arr[s] >= 2:
            valid = False
        if valid:
            for y in current_set_list:
                s = x + y
                if count_arr[s] >= 2:
                    valid = False
                    break
        if valid:
            # Update counts
            count_arr[2 * x] += 1
            for y in current_set_list:
                count_arr[x + y] += 1
            current_set_list.append(x)
            current_set_set.add(x)
            
    best_set = list(current_set_list)
    best_cost = sum(max(0, c - 2) for c in count_arr)
    current_cost = best_cost
    
    # 2. Local Search
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    while time.time() < deadline:
        r = random.random()
        if r < 0.5:
            # Add
            x = random.randint(1, N)
            if x not in current_set_set:
                # Calculate cost increase
                delta_cost = 0
                sums_to_update = []
                
                s = 2 * x
                if count_arr[s] >= 2:
                    delta_cost += (count_arr[s] + 1) - 2
                sums_to_update.append(s)
                
                for y in current_set_list:
                    s = x + y
                    if count_arr[s] >= 2:
                        delta_cost += (count_arr[s] + 1) - 2
                    sums_to_update.append(s)
                
                if delta_cost == 0 or random.random() < math.exp(-delta_cost / T):
                    for s in sums_to_update:
                        count_arr[s] += 1
                    current_set_list.append(x)
                    current_set_set.add(x)
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_set = list(current_set_list)
                        
        elif r < 0.7:
            # Remove
            if current_set_list:
                x = random.choice(current_set_list)
                delta_cost = 0
                sums_to_update = []
                
                s = 2 * x
                if count_arr[s] > 2:
                    delta_cost -= 1
                sums_to_update.append(s)
                
                for y in current_set_list:
                    if y != x:
                        s = x + y
                        if count_arr[s] > 2:
                            delta_cost -= 1
                        sums_to_update.append(s)
                        
                if delta_cost == 0 or random.random() < math.exp(-delta_cost / T):
                    for s in sums_to_update:
                        count_arr[s] -= 1
                    current_set_list.remove(x)
                    current_set_set.discard(x)
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_set = list(current_set_list)
        else:
            # Do nothing or restart? 
            pass

        T *= cooling
        if T < T_min:
            T = T_min

    return best_set
