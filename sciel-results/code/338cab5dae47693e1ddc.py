import random
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = 256
    # Precompute for each a, the forbidden sums: multiples of a in range (2a, 512]
    # For a, forbidden b+c values are {3a, 4a, ..., floor(512/a)*a}
    # But b+c must have b > a and b,c in set, b <= c
    
    # Start with upper tail [171..256] as initial solution
    current = set(range(171, 257))
    best = set(current)
    best_size = len(best)
    
    # Helper: check if adding element x to set S creates any violation
    # A violation: exists a in S, b in S, c in S with a < b <= c and a | (b+c)
    # When we add x, new violations involve x as a, b, or c
    def check_violation_with_x(S, x):
        # x as a: need b,c in S with x < b <= c and x | (b+c)
        # x as b: need a,c in S with a < x <= c and a | (x+c)
        # x as c: need a,b in S with a < b <= x and a | (b+x)
        S_list = sorted(S)
        S_set = S
        
        # x as smallest (a=x)
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return True
        # x as middle (b=x), a < x <= c
        for a in S_list:
            if a >= x:
                break
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return True
        # x as largest (c=x), a < b <= x
        for a in S_list:
            if a >= x:
                break
            for b in S_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return True
        return False
    
    # Greedy: try to add elements from below 171
    # Try all elements 1..170, add if no violation
    for x in range(1, 171):
        if x not in current:
            if not check_violation_with_x(current, x):
                current.add(x)
    
    if len(current) > best_size:
        best = set(current)
        best_size = len(current)
    
    # Local search: try swaps to improve
    # Try removing one element and adding another
    temp = set(current)
    
    # Simulated annealing style: try random modifications
    rng = random.Random(42)
    temperature = 10.0
    cooling = 0.999
    
    for iteration in range(200000):
        if time.time() > deadline - 5:
            break
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.3:
            # Try adding a random element not in set
            candidates = [x for x in range(1, 257) if x not in temp]
            if not candidates:
                continue
            x = rng.choice(candidates)
            if not check_violation_with_x(temp, x):
                temp.add(x)
                if len(temp) > best_size:
                    best = set(temp)
                    best_size = len(temp)
            else:
                # Violation, revert (don't add)
                pass
        elif move_type < 0.6:
            # Try removing a random element
            if temp:
                x = rng.choice(list(temp))
                temp.remove(x)
        else:
            # Try swap: remove one, add another
            if temp:
                remove_candidates = list(temp)
                add_candidates = [x for x in range(1, 257) if x not in temp]
                if add_candidates:
                    x_remove = rng.choice(remove_candidates)
                    x_add = rng.choice(add_candidates)
                    temp2 = set(temp)
                    temp2.remove(x_remove)
                    if not check_violation_with_x(temp2, x_add):
                        temp2.add(x_add)
                        temp = temp2
                        if len(temp) > best_size:
                            best = set(temp)
                            best_size = len(temp)
        
        # Cooling
        temperature *= cooling
        if temperature < 0.1:
            temperature = 0.1
    
    # Final greedy pass on best
    final = set(best)
    for x in range(1, 257):
        if x not in final:
            if not check_violation_with_x(final, x):
                final.add(x)
    
    if len(final) > len(best):
        best = final
    
    return list(best)
