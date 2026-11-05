import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    max_val = 128
    
    # 1. Precomputation
    # LCM table
    lcm_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i, max_val + 1):
            lcm_table[i][j] = i * j // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Helper to check if x can be added to a list S
    def can_add(x, S):
        # S is a list of integers
        n = len(S)
        for i in range(n):
            a = S[i]
            l_ax = lcm_table[a][x]
            for j in range(i + 1, n):
                b = S[j]
                # Check if lcm(a,b) == lcm(a,x) == lcm(b,x)
                # We already have l_ax.
                # We need lcm(a,b) == l_ax AND lcm(b,x) == l_ax
                if lcm_table[a][b] == l_ax:
                    if lcm_table[b][x] == l_ax:
                        return False
        return True

    def is_valid_set(S):
        # Full verification
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                l_ab = lcm_table[a][b]
                for k in range(j + 1, n):
                    c = S[k]
                    if l_ab == lcm_table[a][c] and l_ab == lcm_table[b][c]:
                        return False
        return True

    # 2. Greedy Construction
    def greedy_construct(order):
        S = []
        for x in order:
            if can_add(x, S):
                S.append(x)
        return S

    # Run greedy multiple times with different random orders to find a good start
    best_greedy = []
    for _ in range(20):
        random.seed(int(time.time() * 1000) + _)
        order = list(range(1, max_val + 1))
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > len(best_greedy):
            best_greedy = S
            
    # Also try standard order
    S_std = greedy_construct(range(1, max_val + 1))
    if len(S_std) > len(best_greedy):
        best_greedy = S_std
        
    current = best_greedy[:]
    best_set = current[:]
    current_size = len(current)
    
    # 3. Local Search (Simulated Annealing)
    # State: current (list), in_set (set)
    in_set = set(current)
    not_in_set = set(range(1, max_val + 1)) - in_set
    
    temperature = 1.0
    cooling_rate = 0.999
    min_temperature = 0.01
    step_count = 0
    
    # To speed up random choice from not_in_set, maintain a list
    # But not_in_set changes. Converting to list is O(N). N=128. Acceptable.
    
    while time.time() < deadline and temperature > min_temperature:
        # Choose move
        r = random.random()
        
        if r < 0.5 and not_in_set:
            # Try Add
            x = random.choice(list(not_in_set))
            if can_add(x, current):
                # Accept
                current.append(x)
                in_set.add(x)
                not_in_set.remove(x)
                if len(current) > current_size:
                    current_size = len(current)
                    best_set = current[:]
            else:
                # Reject (or accept with prob if we allowed invalid? No, must be valid)
                pass
                
        elif r < 0.7 and current:
            # Try Remove
            # Removing is always valid. 
            # We accept with probability p_remove to allow exploration
            # In SA, if we remove, size decreases by 1. Delta = -1.
            # Accept prob = exp(-(-1)/T) = exp(1/T). Since T>0, this is > 1? 
            # No, standard SA accepts worse moves with prob exp(-Delta/T).
            # If Delta is "cost", and we minimize cost, then size is negative cost.
            # Let Cost = -Size.
            # Add: Delta Cost = -1. Accept if exp(-(-1)/T) = exp(1/T) > 1 -> Always accept?
            #   Actually, if we improve, we always accept.
            # Remove: Delta Cost = +1. Accept with prob exp(-1/T).
            
            # So:
            # If we Add and valid: Accept (Improvement)
            # If we Remove: Accept with prob exp(-1/T)
            
            if random.random() < (1.0 / (1.0 + temperature)): # Heuristic acceptance for removal
                # Or simply: if random.random() < 0.1 (fixed prob)
                # Let's use exp(-1/T)
                if random.random() < pow(2.71828, -1.0 / temperature):
                    x = random.choice(current)
                    current.remove(x)
                    in_set.remove(x)
                    not_in_set.add(x)
                    
        else:
            # Try Swap
            if current and not_in_set:
                x_out = random.choice(current)
                x_in = random.choice(list(not_in_set))
                
                # Check if x_in is valid in (current - {x_out})
                # We can simulate by checking pairs in current, skipping x_out
                valid = True
                n = len(current)
                for i in range(n):
                    a = current[i]
                    if a == x_out: continue
                    l_ax = lcm_table[a][x_in]
                    for j in range(i + 1, n):
                        b = current[j]
                        if b == x_out: continue
                        if lcm_table[a][b] == l_ax:
                            if lcm_table[b][x_in] == l_ax:
                                valid = False
                                break
                    if not valid: break
                
                if valid:
                    # Swap is size-neutral. 
                    # We accept swaps to escape local optima.
                    # In SA, if size doesn't change, Delta Cost = 0.
                    # Accept with prob 1? Or always?
                    # Usually we accept neutral moves to explore.
                    # Let's accept with high probability.
                    if random.random() < 0.5:
                        # Perform swap
                        idx = current.index(x_out)
                        current[idx] = x_in
                        in_set.remove(x_out)
                        in_set.add(x_in)
                        not_in_set.remove(x_in)
                        not_in_set.add(x_out)
        
        temperature *= cooling_rate
        step_count += 1
        
        # Optional: If we have been stuck for a long time, restart?
        # Or just keep going.
        
    # Final verification
    if not is_valid_set(best_set):
        # Fallback
        best_set = greedy_construct(range(1, max_val + 1))
        
    return best_set

# EVOLVE-BLOCK-END
