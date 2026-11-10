import random
import math
import time

def construct():
    n = 256
    
    # Helper to check if y can be added to set S (stored as set for O(1) lookup)
    def is_valid_addition(y, S_set):
        # Check if y creates a 3-AP with any two elements in S
        # 1. y is middle: a, y, c => a + c = 2y. a, c in S.
        # 2. y is end: a, b, y => a + y = 2b => a = 2b - y. b, a in S.
        #    OR y, b, c => y + c = 2b => c = 2b - y. b, c in S.
        
        # Case 1: y is middle
        # a + c = 2y. Since a, c in S, and S is a set, we can just check if there exists a in S such that 2y-a is in S.
        # We must ensure a != c. If a=c, then a=y, but y not in S. So a!=c is automatic.
        for a in S_set:
            c = 2 * y - a
            if c in S_set:
                return False
        
        # Case 2 & 3: y is an end term
        # There exists b in S such that 2b - y is in S.
        # Let a = 2b - y. If a in S, then (a, b, y) or (y, b, a) is an AP.
        for b in S_set:
            a = 2 * b - y
            if a in S_set:
                return False
                
        return True

    # 1. Greedy Initialization
    S = set()
    for x in range(1, n + 1):
        if is_valid_addition(x, S):
            S.add(x)
    
    best_S = set(S)
    best_val = sum(1.0 / x for x in S)
    
    # 2. Simulated Annealing
    # We use a time-based stopping condition to ensure we finish within 60s.
    # However, to be safe, we also limit iterations.
    start_time = time.time()
    max_time = 55.0 # Leave 5s buffer
    
    T = 0.05
    T_min = 1e-6
    cooling = 0.9995
    
    current_val = best_val
    current_S = set(best_S)
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.random()
        
        new_S = set(current_S)
        new_val = current_val
        
        if move_type < 0.4:
            # Try to add a random element
            candidates = [x for x in range(1, n + 1) if x not in current_S]
            if candidates:
                y = random.choice(candidates)
                if is_valid_addition(y, new_S):
                    new_S.add(y)
                    new_val += 1.0 / y
                else:
                    # Move failed, keep current
                    new_S = current_S
                    new_val = current_val
        elif move_type < 0.8:
            # Try to swap: remove x, add y
            if not current_S:
                continue
            x = random.choice(list(current_S))
            # Remove x first to check validity of y in S\{x}
            temp_S = current_S - {x}
            candidates = [y for y in range(1, n + 1) if y not in temp_S]
            if candidates:
                y = random.choice(candidates)
                if is_valid_addition(y, temp_S):
                    new_S = temp_S | {y}
                    new_val = current_val - (1.0 / x) + (1.0 / y)
                else:
                    new_S = current_S
                    new_val = current_val
        else:
            # Try to remove an element (rarely helps, but allows escape)
            if current_S:
                x = random.choice(list(current_S))
                new_S = current_S - {x}
                new_val = current_val - (1.0 / x)
        
        # Acceptance
        delta = new_val - current_val
        if delta >= 0 or random.random() < math.exp(delta / T):
            current_S = new_S
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_S = set(current_S)
        
        T *= cooling

    return list(best_S)
