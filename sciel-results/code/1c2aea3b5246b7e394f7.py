import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 58.0  # Leave 2s margin for safety
    
    N = 512
    
    # Precompute 4-APs
    # An AP is (a, b, c, d) with b=a+d, c=a+2d, d=a+3d
    # Indices 0..N-1
    aps_by_element = [[] for _ in range(N)]
    
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(N - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap = (a, b, c, e)
            aps_by_element[a].append(ap)
            aps_by_element[b].append(ap)
            aps_by_element[c].append(ap)
            aps_by_element[e].append(ap)
            
    # Helper to check if x can be added to in_set (bytearray of 0/1)
    def can_add(x, in_set):
        for ap in aps_by_element[x]:
            i1, i2, i3, i4 = ap
            if i1 == x:
                if in_set[i2] and in_set[i3] and in_set[i4]:
                    return False
            elif i2 == x:
                if in_set[i1] and in_set[i3] and in_set[i4]:
                    return False
            elif i3 == x:
                if in_set[i1] and in_set[i2] and in_set[i4]:
                    return False
            else: # i4 == x
                if in_set[i1] and in_set[i2] and in_set[i3]:
                    return False
        return True

    def greedy_init():
        in_set = bytearray(N)
        elements = list(range(N))
        random.shuffle(elements)
        count = 0
        for x in elements:
            if can_add(x, in_set):
                in_set[x] = 1
                count += 1
        return in_set, count

    # Simulated Annealing
    in_set, count = greedy_init()
    best_in_set = in_set.copy()
    best_count = count
    
    # Temperature schedule
    # Start high enough to allow many removals, cool down to focus on additions
    T_start = 1.0
    T_end = 0.01
    # We will run for a fixed number of steps or time
    # Let's estimate steps per second. 
    # can_add is ~340 checks. 
    # Loop overhead is small.
    # Let's just loop until time limit.
    
    # To allow escaping, we need to accept bad moves (removals).
    # Energy = -count. 
    # Delta E for removal = +1. Accept with prob exp(-1/T).
    # Delta E for addition = -1. Always accept.
    
    # We can also use a "restart" strategy if no improvement for a long time.
    # But SA with proper cooling should handle it.
    
    # Let's use a dynamic cooling schedule based on time.
    # Or fixed steps. 
    # Let's try fixed steps with time check.
    
    steps_per_iter = 1000
    total_steps = 10_000_000 # Will be cut by time limit
    
    # We need a list of elements in set and not in set for fast random selection.
    # Maintaining these lists dynamically is costly.
    # Instead, just pick random indices and check membership.
    # For N=512, picking random index is O(1).
    
    current_count = count
    
    # Initial temperature
    T = T_start
    cooling_factor = 0.999
    
    # We will alternate between "explore" (higher T) and "exploit" (lower T)
    # Or just cool down.
    
    # Let's do a long run.
    while time.time() - start_time < time_limit:
        # Move selection
        # 60% Add, 40% Remove (to allow escaping)
        # Actually, if we are far from optimal, we want to add.
        # If we are stuck, we need to remove.
        # A good heuristic: 
        # If count < best_count * 0.9, bias towards add.
        # Else, balance.
        
        # Simple approach:
        # Try to add a random element.
        # If it fails (invalid), maybe try to remove a random element?
        # No, standard SA:
        
        # Pick a random element x
        x = random.randint(0, N - 1)
        
        if not in_set[x]:
            # Try to add
            if can_add(x, in_set):
                in_set[x] = 1
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_in_set = in_set.copy()
            # If cannot add, do nothing (or try remove? No, x is not in set)
        else:
            # Try to remove
            # Accept with probability exp(-1/T)
            if random.random() < math.exp(-1.0 / T):
                in_set[x] = 0
                current_count -= 1
        
        # Cool down
        T *= cooling_factor
        if T < T_end:
            T = T_end
            
    # Final polish: Greedy add on best solution
    in_set = best_in_set.copy()
    count = best_count
    # Try to add any remaining elements
    for x in range(N):
        if not in_set[x]:
            if can_add(x, in_set):
                in_set[x] = 1
                count += 1
                
    result = [i + 1 for i in range(N) if in_set[i]]
    
    if not result:
        return [1]
        
    return result
