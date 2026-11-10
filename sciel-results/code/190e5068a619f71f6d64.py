import random
import math

def construct():
    n = 256
    
    # Helper to check if y can be added to S
    def can_add(y, S):
        if y in S:
            return False
        for b in S:
            # Check if y forms AP with b and some a in S
            # 1. y is largest: a, b, y -> 2b - a = y -> a = 2b - y
            a = 2 * b - y
            if a in S:
                return False
            # 2. y is smallest: y, b, c -> 2b - c = y -> c = 2b - y (same as above if we iterate all b)
            # Wait, if y is smallest, the AP is y, b, c. 
            # 2b = y + c => c = 2b - y.
            # If c in S, then y, b, c is AP.
            # So checking if 2b - y in S covers both y being largest (a=2b-y) and y being smallest (c=2b-y).
            # What if y is middle?
            # a, y, c -> 2y = a + c -> c = 2y - a.
            # So we must also check if 2y - b in S? No, b is one of the endpoints.
            # If y is middle, the other two are a, c in S.
            # 2y - a = c. So for each a in S, check if 2y - a in S.
            # This is equivalent to: for each a in S, if 2y - a in S, invalid.
            
            # So we need to check:
            # 1. Exists b in S such that 2b - y in S. (y is end)
            # 2. Exists a in S such that 2y - a in S. (y is middle)
            
            # My previous check `a = 2b - y` covers case 1.
            # I need to add case 2.
            
            c = 2 * y - b
            if c in S:
                return False
        return True

    # Greedy Initialization
    S = set()
    for x in range(1, n + 1):
        if can_add(x, S):
            S.add(x)
    
    current_sum = sum(1.0 / x for x in S)
    best_S = set(S)
    best_sum = current_sum
    
    # Simulated Annealing
    # Temperature schedule
    T = 0.01
    T_min = 0.0001
    cooling = 0.999
    
    # To speed up, we precompute list of S
    S_list = list(S)
    
    for _ in range(100000): # 100k iterations
        if T < T_min:
            break
            
        # Pick a random move type
        # 1. Add
        # 2. Remove
        # 3. Swap
        
        move_type = random.randint(0, 2)
        
        if move_type == 0: # Add
            # Pick random y not in S
            y = random.randint(1, n)
            if y in S:
                continue
            if can_add(y, S):
                new_sum = current_sum + 1.0 / y
                # Always accept improvement
                S.add(y)
                current_sum = new_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_S = set(S)
        elif move_type == 1: # Remove
            if not S:
                continue
            x = random.choice(list(S))
            delta = -1.0 / x
            # Accept with probability
            if random.random() < math.exp(delta / T):
                S.remove(x)
                current_sum += delta
        else: # Swap
            if not S:
                continue
            x = random.choice(list(S))
            y = random.randint(1, n)
            if y in S:
                continue
            # Check if S \ {x} U {y} is valid
            # Temporarily remove x?
            # can_add(y, S - {x})
            # We can just check manually to avoid set creation
            valid = True
            for b in S:
                if b == x:
                    continue
                a = 2 * b - y
                if a in S and a != x:
                    valid = False
                    break
                c = 2 * y - b
                if c in S and c != x:
                    valid = False
                    break
            if valid:
                delta = (1.0 / y) - (1.0 / x)
                if delta > 0 or random.random() < math.exp(delta / T):
                    S.remove(x)
                    S.add(y)
                    current_sum += delta
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_S = set(S)
        
        T *= cooling
        
    return list(best_S)
