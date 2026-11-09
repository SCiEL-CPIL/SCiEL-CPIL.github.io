import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    N = 1000
    
    def compute_initial_counts(S):
        counts = {}
        for i, x in enumerate(S):
            for y in S[i:]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
        return counts

    def can_add(x, S_list, S_set, counts):
        # Check sums with existing elements
        for y in S_list:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        s = 2 * x
        if counts.get(s, 0) >= 2:
            return False
        return True

    def add_element(x, S_list, S_set, counts):
        S_list.append(x)
        S_set.add(x)
        for y in S_list[:-1]:  # y is all previous elements
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1

    def remove_element(x, S_list, S_set, counts):
        S_list.remove(x)
        S_set.remove(x)
        for y in S_list:  # y is all remaining elements
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]

    def randomized_greedy():
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        S_list = []
        S_set = set()
        counts = {}
        
        for x in candidates:
            if can_add(x, S_list, S_set, counts):
                add_element(x, S_list, S_set, counts)
        return S_list, S_set, counts

    def local_improve(S_list, S_set, counts):
        # Try to add elements
        while time.time() < deadline:
            added = False
            # Get candidates not in S
            candidates = [x for x in range(1, 1001) if x not in S_set]
            if not candidates:
                break
            
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, S_list, S_set, counts):
                    add_element(x, S_list, S_set, counts)
                    added = True
                    break  # Restart loop to re-evaluate with new set
            
            if not added:
                # Try a swap
                if not S_list:
                    break
                
                x_to_remove = random.choice(S_list)
                remove_element(x_to_remove, S_list, S_set, counts)
                
                # Try to add a different element
                candidates = [x for x in range(1, 1001) if x not in S_set]
                random.shuffle(candidates)
                
                swapped = False
                for x in candidates:
                    if can_add(x, S_list, S_set, counts):
                        add_element(x, S_list, S_set, counts)
                        swapped = True
                        break
                
                if not swapped:
                    # Revert
                    add_element(x_to_remove, S_list, S_set, counts)
                    break
                else:
                    # We swapped. Size is same. 
                    # Continue loop to see if we can add more now.
                    pass
        
        return S_list, S_set, counts

    # Simulated annealing approach
    def simulated_annealing():
        # Start with a random valid set
        S_list, S_set, counts = randomized_greedy()
        
        if not S_list:
            return S_list
        
        current_size = len(S_list)
        best_size = current_size
        best_set = S_list[:]
        
        # Track sum counts as a numpy array for fast updates
        # sum_counts[s] = number of pairs (a,b) with a<=b, a+b=s
        max_sum = 2000
        sum_counts = np.zeros(max_sum + 1, dtype=np.int32)
        for x in S_list:
            for y in S_list:
                if x <= y:
                    sum_counts[x + y] += 1
        
        # Current excess: sum over s of max(0, sum_counts[s] - 2)
        total_excess = np.sum(np.maximum(0, sum_counts - 2))
        
        # For SA, we need a valid set. Let's start with a valid set from greedy.
        # Actually, let's restart SA from a valid set.
        # We'll use the greedy result as the starting point.
        
        T = 50.0
        T_min = 0.001
        cooling = 0.995
        
        # We'll do SA moves: add, remove, swap
        # Objective: maximize size. Constraint: valid (excess == 0)
        # If we allow invalid states, we penalize excess.
        # Let's use: cost = -size + M * excess, minimize cost.
        M = N + 1  # large enough
        
        current_cost = -current_size + M * total_excess
        best_cost = current_cost
        
        while time.time() < deadline:
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random element
                candidates = [x for x in range(1, 1001) if x not in S_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                
                # Compute delta
                delta_size = 1
                delta_excess = 0
                for y in S_list:
                    s = x + y
                    if sum_counts[s] == 1:
                        delta_excess += 0  # becomes 2, no excess
                    elif sum_counts[s] == 2:
                        delta_excess += 1  # becomes 3, excess 1
                    elif sum_counts[s] >= 3:
                        delta_excess += 1  # excess increases by 1
                s = 2 * x
                if sum_counts[s] == 1:
                    delta_excess += 0
                elif sum_counts[s] == 2:
                    delta_excess += 1
                elif sum_counts[s] >= 3:
                    delta_excess += 1
                
                delta_cost = -delta_size + M * delta_excess
                
                if delta_cost <= 0 or (T > T_min and random.random() < np.exp(-delta_cost / T)):
                    # Accept
                    add_element(x, S_list, S_set, counts)
                    for y in S_list[:-1]:
                        sum_counts[x + y] += 1
                    sum_counts[2 * x] += 1
                    current_size += 1
                    total_excess += delta_excess
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_size = current_size
                        best_set = S_list[:]
                        
            elif move_type < 0.7:
                # Remove a random element
                if not S_list:
                    continue
                x = random.choice(S_list)
                
                delta_size = -1
                delta_excess = 0
                for y in S_list:
                    if x != y:
                        s = x + y
                        if sum_counts[s] > 2:
                            delta_excess -= 1  # excess decreases by 1
                        elif sum_counts[s] == 2:
                            delta_excess -= 0  # becomes 1, no change in excess
                s = 2 * x
                if sum_counts[s] > 2:
                    delta_excess -= 1
                elif sum_counts[s] == 2:
                    delta_excess -= 0
                
                delta_cost = -delta_size + M * delta_excess
                
                if delta_cost <= 0 or (T > T_min and random.random() < np.exp(-delta_cost / T)):
                    # Accept
                    remove_element(x, S_list, S_set, counts)
                    for y in S_list:
                        sum_counts[x + y] -= 1
                    sum_counts[2 * x] -= 1
                    current_size -= 1
                    total_excess += delta_excess
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_size = current_size
                        best_set = S_list[:]
                        
            else:
                # Swap: remove one, add another
                if not S_list:
                    continue
                x_out = random.choice(S_list)
                candidates = [x for x in range(1, 1001) if x not in S_set]
                if not candidates:
                    continue
                x_in = random.choice(candidates)
                
                # First remove x_out
                temp_sum_counts = sum_counts.copy()
                for y in S_list:
                    if x_out != y:
                        temp_sum_counts[x_out + y] -= 1
                temp_sum_counts[2 * x_out] -= 1
                
                temp_excess = np.sum(np.maximum(0, temp_sum_counts - 2))
                
                # Then add x_in
                for y in S_list:
                    if x_out != y:
                        temp_sum_counts[x_in + y] += 1
                temp_sum_counts[2 * x_in] += 1
                
                final_excess = np.sum(np.maximum(0, temp_sum_counts - 2))
                
                delta_size = 0
                delta_excess = final_excess - total_excess
                delta_cost = -delta_size + M * delta_excess
                
                if delta_cost <= 0 or (T > T_min and random.random() < np.exp(-delta_cost / T)):
                    # Accept
                    remove_element(x_out, S_list, S_set, counts)
                    add_element(x_in, S_list, S_set, counts)
                    
                    # Update sum_counts
                    for y in S_list:
                        if x_out != y:
                            sum_counts[x_out + y] -= 1
                            sum_counts[x_in + y] += 1
                    sum_counts[2 * x_out] -= 1
                    sum_counts[2 * x_in] += 1
                    
                    current_size = len(S_list)
                    total_excess = final_excess
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_size = current_size
                        best_set = S_list[:]
            
            # Cooling
            if T > T_min:
                T *= cooling
                if T < T_min:
                    T = T_min
        
        # Ensure best_set is valid
        return best_set

    # Main loop: alternate between greedy+local and SA
    best_size = 0
    best_set = []
    
    while time.time() < deadline:
        # Try greedy + local improvement
        S_list, S_set, counts = randomized_greedy()
        S_list, S_set, counts = local_improve(S_list, S_set, counts)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
        
        # Try simulated annealing
        sa_result = simulated_annealing()
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set = sa_result[:]
        
        if time.time() > deadline - 2:
            break
    
    return best_set
