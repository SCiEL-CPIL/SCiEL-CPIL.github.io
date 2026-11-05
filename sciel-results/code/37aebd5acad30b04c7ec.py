import random
import math
import time

def construct():
    n = 256
    # Precompute potential AP partners? No, dynamic check is fast enough.
    
    # Helper to check if adding x to S is valid
    def is_valid_addition(x, S):
        # Check if x forms a 3-AP with any a, b in S
        # 1. x, a, b => 2a = x + b => b = 2a - x. Need b > a and b in S.
        # 2. a, x, b => 2x = a + b => b = 2x - a. Need b > x and b in S.
        # 3. a, b, x => 2b = a + x => b = (a + x) / 2. Need a < b < x and b in S.
        
        for a in S:
            # Case 1: x is first term
            b = 2 * a - x
            if b > a and b in S:
                return False
            
            # Case 2: x is middle term
            b = 2 * x - a
            if b > x and b in S:
                return False
            
            # Case 3: x is last term
            # a, b, x => 2b = a + x => b = (a + x) / 2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if a < b < x and b in S:
                    return False
        return True

    def is_valid_swap(x, y, S):
        # Remove y, add x.
        # S_new = S - {y} U {x}
        # We need to check if x forms an AP with any a, b in S_new.
        # Note: y is not in S_new.
        S_new = S - {y}
        
        # Check APs involving x with elements in S_new
        for a in S_new:
            # 1. x, a, b => b = 2a - x. Need b > a and b in S_new.
            b = 2 * a - x
            if b > a and b in S_new:
                return False
            
            # 2. a, x, b => b = 2x - a. Need b > x and b in S_new.
            b = 2 * x - a
            if b > x and b in S_new:
                return False
            
            # 3. a, b, x => b = (a + x) / 2. Need a < b < x and b in S_new.
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if a < b < x and b in S_new:
                    return False
        return True

    # Initialize with greedy set (similar to seed but faster)
    S = set()
    for x in range(1, n + 1):
        if is_valid_addition(x, S):
            S.add(x)
    
    current_score = sum(1.0 / x for x in S)
    best_S = set(S)
    best_score = current_score

    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-5
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer

    while T > min_T and (time.time() - start_time) < time_limit:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add a random element
            candidates = [x for x in range(1, n + 1) if x not in S]
            if not candidates:
                T *= cooling_rate
                continue
            x = random.choice(candidates)
            if is_valid_addition(x, S):
                new_S = S | {x}
                new_score = current_score + 1.0 / x
                # Always accept improvement
                S = new_S
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
            else:
                # If invalid, we could try to remove something to allow it, 
                # but that's a swap. Here we just skip or reject.
                pass
                
        elif move_type < 0.6:
            # Try to remove a random element
            if S:
                y = random.choice(list(S))
                new_S = S - {y}
                new_score = current_score - 1.0 / y
                # Accept if better or with probability exp(-delta/T)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    S = new_S
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                        
        else:
            # Try to swap
            if S:
                y = random.choice(list(S))
                candidates = [x for x in range(1, n + 1) if x not in S]
                if candidates:
                    x = random.choice(candidates)
                    if is_valid_swap(x, y, S):
                        new_S = (S - {y}) | {x}
                        new_score = current_score - 1.0 / y + 1.0 / x
                        delta = new_score - current_score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            S = new_S
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_S = set(S)
        
        T *= cooling_rate

    return sorted(best_S)
