import time
import random

N = 256

def is_valid_addition(S, x):
    """
    Checks if adding x to set S results in a valid set.
    S is a set of integers.
    """
    # 1. x acts as 'a' (smallest in the triple)
    # Need b, c in S such that x < b <= c and x | (b+c)
    L = [y for y in S if y > x]
    if len(L) >= 2:
        L_set = set(L)
        # Sort L for breaking early?
        # We check if there exists b, c in L with b <= c and b+c is multiple of x.
        # Max sum 2*N.
        for k in range(3, 2 * N // x + 1):
            target = k * x
            # Check if there are b, c in L with b + c = target and b <= c
            # b < target/2
            found = False
            for b in L:
                if b > target // 2:
                    break
                c = target - b
                if c in L_set:
                    return False
    # 2. x acts as 'b' (middle)
    # Need a, c in S such that a < x <= c and a | (x+c)
    # a < x
    # c >= x
    A_less = [a for a in S if a < x]
    C_geq = [c for c in S if c >= x]
    for a in A_less:
        for c in C_geq:
            if (x + c) % a == 0:
                return False
                
    # 3. x acts as 'c' (largest)
    # Need a, b in S such that a < b <= x and a | (b+x)
    # b <= x
    # a < b
    B_leq = [b for b in S if b <= x]
    for b in B_leq:
        # a < b
        # a in S
        for a in S:
            if a < b:
                if (b + x) % a == 0:
                    return False
                    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Seed solution
    best_set = list(range(171, 257))
    best_size = len(best_set)
    
    while time.time() < deadline:
        # Random Greedy Construction
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        S = set()
        for x in candidates:
            if is_valid_addition(S, x):
                S.add(x)
                
        if len(S) > best_size:
            best_set = list(S)
            best_size = len(S)
            
            # Local Improvement: Add single elements
            improved = True
            while improved and time.time() < deadline:
                improved = False
                current_S = set(best_set)
                for x in range(1, N + 1):
                    if x not in current_S:
                        if is_valid_addition(current_S, x):
                            best_set.append(x)
                            best_size += 1
                            current_S.add(x)
                            improved = True
                            break
        
        # Local Search: Remove 1, Add 2 (via re-greedy)
        if time.time() < deadline - 2.0 and random.random() < 0.3:
            S_current = set(best_set)
            if len(S_current) > 0:
                x_remove = random.choice(list(S_current))
                S_temp = S_current - {x_remove}
                
                # Re-greedy add
                S_greedy = set(S_temp)
                candidates_to_add = [x for x in range(1, N+1) if x not in S_greedy]
                random.shuffle(candidates_to_add)
                
                for y in candidates_to_add:
                    if is_valid_addition(S_greedy, y):
                        S_greedy.add(y)
                        
                if len(S_greedy) > best_size:
                    best_set = list(S_greedy)
                    best_size = len(best_set)
                    
                    # Polish
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        current_S = set(best_set)
                        for x in range(1, N + 1):
                            if x not in current_S:
                                if is_valid_addition(current_S, x):
                                    best_set.append(x)
                                    best_size += 1
                                    current_S.add(x)
                                    improved = True
                                    break

    return best_set
