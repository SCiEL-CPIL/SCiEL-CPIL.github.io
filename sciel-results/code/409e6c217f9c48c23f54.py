def construct():
    import random
    import time
    from math import gcd
    from itertools import combinations

    # Precompute gcds for speed?
    # gcd(a, b) for a, b in 1..128.
    # 128x128 matrix.
    gcd_table = [[0]*129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    def is_valid_triple(a, b, c):
        # Check if gcd(a,b) == gcd(a,c) == gcd(b,c)
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        if g_ab != g_ac:
            return True # Not forbidden
        g_bc = gcd_table[b][c]
        if g_ab != g_bc:
            return True
        return False

    def can_add(x, current_set):
        # Check if adding x to current_set creates a forbidden triple
        # We need to check all pairs (a, b) in current_set
        # If gcd(a,b) == gcd(a,x) == gcd(b,x), return False
        items = list(current_set)
        for i in range(len(items)):
            a = items[i]
            g_ax = gcd_table[a][x]
            for j in range(i+1, len(items)):
                b = items[j]
                if gcd_table[a][b] == g_ax and gcd_table[b][x] == g_ax:
                    return False
        return True

    def greedy_fill():
        current_set = set()
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, current_set):
                current_set.add(x)
        return current_set

    best_set = greedy_fill()
    best_size = len(best_set)
    
    start_time = time.time()
    time_limit = 55.0
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    current_set = best_set
    current_size = best_size
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        move_type = random.random()
        
        if move_type < 0.7:
            # Try to add a random element
            remaining = [x for x in range(1, 129) if x not in current_set]
            if not remaining:
                continue
            x = random.choice(remaining)
            if can_add(x, current_set):
                new_set = current_set | {x}
                new_size = current_size + 1
                # Always accept improvement
                current_set = new_set
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set.copy()
            else:
                # Cannot add, maybe try a swap?
                pass # Handled in next branch or skipped
        
        elif move_type < 0.9:
            # Try to swap: remove y, add x
            if not current_set:
                continue
            y = random.choice(list(current_set))
            remaining = [x for x in range(1, 129) if x not in current_set]
            if not remaining:
                continue
            x = random.choice(remaining)
            
            # Check if S \ {y} U {x} is valid
            # We don't need to check full validity if we know S was valid?
            # Removing y might break validity? No, removing elements never creates new forbidden triples.
            # Adding x might create new forbidden triples with pairs in S\{y}.
            S_minus_y = current_set - {y}
            if can_add(x, S_minus_y):
                new_set = S_minus_y | {x}
                new_size = current_size # Size remains same
                # Accept neutral move to explore?
                # In SA, we accept worse/same with prob exp(-dE/T).
                # Here dE = 0. So prob = 1.
                # But we want to maximize size. Neutral moves help escape.
                current_set = new_set
                # current_size unchanged
            else:
                # Try to find a better x? No, just reject.
                pass
                
        else:
            # Try to remove a random element (shrink)
            # This is rarely useful unless we are stuck in a "hole"
            if len(current_set) > 1:
                y = random.choice(list(current_set))
                new_set = current_set - {y}
                new_size = current_size - 1
                # Accept with prob exp(1/T) ?
                # dE = -(-new_size) - (-current_size) = new_size - current_size = -1
                # Wait, Cost = -Size.
                # E_new - E_old = -new_size - (-current_size) = current_size - new_size = 1.
                # Prob = exp(-1/T).
                if random.random() < pow(2.71828, -1.0 / T):
                    current_set = new_set
                    current_size = new_size
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final local search improvement
    # Try to add any remaining
    improved = True
    while improved:
        improved = False
        remaining = set(range(1, 129)) - best_set
        for x in list(remaining):
            if can_add(x, best_set):
                best_set.add(x)
                remaining.discard(x)
                improved = True
                
    return list(best_set)
