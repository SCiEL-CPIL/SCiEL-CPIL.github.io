import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    mandatory = {1, 2, 5, 11}
    max_val = 284
    
    # Initial state: mandatory + greedy fill
    S = list(mandatory)
    sums = set()
    for i, x in enumerate(S):
        for y in S[i:]:
            sums.add(x + y)
    
    # Greedy fill to get a good starting point
    for x in range(1, max_val + 1):
        if x in S:
            continue
        new_sums = {x + y for y in S} | {2 * x}
        if not (new_sums & sums):
            S.append(x)
            sums |= new_sums
            if time.time() > deadline:
                return S
    S.sort()
    
    def calculate_cost(S):
        # Returns a cost to be minimized
        # Higher cost is worse
        missing = len(mandatory - set(S))
        if missing > 0:
            return 10000 + 1000 * missing - len(S)
        
        # Check for collisions
        sums = set()
        for i in range(len(S)):
            x = S[i]
            for j in range(i, len(S)):
                y = S[j]
                s = x + y
                if s in sums:
                    return 1000 - len(S) # Collision penalty
                sums.add(s)
        
        return -len(S) # Minimize -> Maximize size
    
    best_S = S[:]
    best_cost = calculate_cost(best_S)
    
    current_S = S[:]
    current_cost = best_cost
    
    T = 5.0
    cooling = 0.9995
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if T < 0.01:
            # Restart with higher T if stuck
            if no_improve_count > max_no_improve:
                T = 5.0
                no_improve_count = 0
            else:
                break
        
        # Generate neighbor
        move_type = random.random()
        new_S = current_S[:]
        
        if move_type < 0.3: # Add
            candidates = [x for x in range(1, max_val + 1) if x not in new_S]
            if candidates:
                x = random.choice(candidates)
                new_S.append(x)
        elif move_type < 0.6: # Remove
            if len(new_S) > 0:
                x = random.choice(new_S)
                new_S.remove(x)
        else: # Swap
            if len(new_S) > 0:
                x = random.choice(new_S)
                candidates = [y for y in range(1, max_val + 1) if y not in new_S]
                if candidates:
                    y = random.choice(candidates)
                    new_S.remove(x)
                    new_S.append(y)
        
        new_cost = calculate_cost(new_S)
        delta = new_cost - current_cost
        
        # Acceptance probability
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            current_S = new_S
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_S = new_S[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        T *= cooling
        
    return best_S
