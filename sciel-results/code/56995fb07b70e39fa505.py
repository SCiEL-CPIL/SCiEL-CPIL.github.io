import time
import random
import numpy as np

def construct():
    N = 1000
    max_val = 1000
    total_deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Data structures
    S = []  # List of chosen integers
    in_S = [False] * (max_val + 1)  # in_S[i] is True if i is in S
    # counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
    # Max sum is 2000
    counts = [0] * (2 * max_val + 1)
    
    # Helper to add a number to the set and update counts
    def try_add(x):
        if in_S[x]:
            return False
        if counts[2 * x] >= 2:
            return False
        for y in S:
            if y == x:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    def try_remove(x):
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    def is_valid_swap(a, c):
        if in_S[c]:
            return False
        # Check removing a doesn't go negative (shouldn't happen)
        # Check adding c doesn't exceed 2
        if counts[2 * c] + 1 > 2:
            return False
        for y in S:
            if y == a:
                continue
            s = c + y
            if counts[s] + 1 > 2:
                return False
        return True

    def is_valid_2swap(a, b, c, d):
        if c == d or in_S[c] or in_S[d] or a == b:
            return False
        # Quick checks for the new elements
        if counts[2 * c] + 1 > 2:
            return False
        if counts[2 * d] + 1 > 2:
            return False
        if c != d and counts[c + d] + 1 > 2:
            return False
        for y in S:
            if y == a or y == b:
                continue
            if counts[c + y] + 1 > 2:
                return False
            if counts[d + y] + 1 > 2:
                return False
        return True

    def do_2swap(a, b, c, d):
        S.remove(a)
        in_S[a] = False
        S.remove(b)
        in_S[b] = False
        
        counts[2*a] -= 1
        counts[2*b] -= 1
        if a != b:
            counts[a+b] -= 1
            
        for y in S:
            if y != a and y != b:
                counts[a+y] -= 1
                counts[b+y] -= 1
                
        S.append(c)
        in_S[c] = True
        S.append(d)
        in_S[d] = True
        
        counts[2*c] += 1
        counts[2*d] += 1
        if c != d:
            counts[c+d] += 1
            
        for y in S:
            if y != c and y != d:
                counts[c+y] += 1
                counts[d+y] += 1

    def reset_state():
        S.clear()
        for i in range(max_val + 1):
            in_S[i] = False
        for i in range(2 * max_val + 1):
            counts[i] = 0

    def greedy_init(mode='random'):
        reset_state()
        candidates = list(range(1, max_val + 1))
        if mode == 'random':
            random.shuffle(candidates)
        elif mode == 'small':
            pass
        elif mode == 'large':
            candidates.reverse()
        
        for i in candidates:
            try_add(i)
        return list(S)

    # Track best solution
    best_S = []
    best_size = 0
    
    # Multi-start Local Search
    # We will run several rounds of (Init + Local Search)
    # If time runs out, return best.
    
    all_numbers = list(range(1, max_val + 1))
    not_in_S = set(all_numbers)
    
    def get_not_in_S():
        return [x for x in all_numbers if not in_S[x]]
    
    while time.time() < total_deadline:
        # 1. Initialization
        # Alternate between random and small-first to explore different landscapes
        # Small-first tends to produce denser sets in low numbers
        mode = random.choice(['random', 'small', 'random'])
        init_S = greedy_init(mode)
        
        if len(init_S) > best_size:
            best_size = len(init_S)
            best_S = list(init_S)
        
        if len(S) == 0:
            continue
            
        # 2. Local Search
        T = 1.0
        cooling_rate = 0.999
        no_improve_count = 0
        local_best_size = len(S)
        local_best_S = list(S)
        
        # Adaptive restart deadline
        base_restart_time = 4.0
        restart_deadline = time.time() + base_restart_time
        
        while time.time() < restart_deadline and time.time() < total_deadline:
            move_type = random.random()
            
            if move_type < 0.35:
                # Add: pick from candidates not in S
                candidates = get_not_in_S()
                if not candidates:
                    continue
                c = random.choice(candidates)
                if try_add(c):
                    if len(S) > local_best_size:
                        local_best_size = len(S)
                        local_best_S = list(S)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    T = max(0.1, T * cooling_rate)
                else:
                    no_improve_count += 1
            elif move_type < 0.70:
                # Swap: pick a from S, c from not in S
                if not S:
                    continue
                a = random.choice(S)
                candidates = get_not_in_S()
                if not candidates:
                    continue
                c = random.choice(candidates)
                if is_valid_swap(a, c):
                    try_remove(a)
                    try_add(c)
                    no_improve_count += 1
                    T = max(0.1, T * cooling_rate)
                else:
                    no_improve_count += 1
            elif move_type < 0.93:
                # 2-Swap
                if len(S) < 2:
                    continue
                a, b = random.sample(S, 2)
                candidates = get_not_in_S()
                if len(candidates) < 2:
                    continue
                c, d = random.sample(candidates, 2)
                if is_valid_2swap(a, b, c, d):
                    do_2swap(a, b, c, d)
                    no_improve_count += 1
                    T = max(0.1, T * cooling_rate)
                else:
                    no_improve_count += 1
            else:
                # Remove
                if S:
                    a = random.choice(S)
                    try_remove(a)
                    no_improve_count += 1
                    T = max(0.1, T * cooling_rate)
            
            # Restart local search if stuck for too long
            if no_improve_count > 400:
                break
        
        # Update global best
        if local_best_size > best_size:
            best_size = local_best_size
            best_S = local_best_S

    # Final polish: try to add more elements to best_S
    if time.time() < total_deadline - 5:
        reset_state()
        for x in sorted(best_S):
            try_add(x)
        
        # Try adding each missing element
        for x in range(1, max_val + 1):
            if time.time() > total_deadline - 2:
                break
            if not in_S[x]:
                try_add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)

    return best_S
