import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    moduli = list(range(2, 129))
    
    # Precompute gcds
    gcd_val = [[0] * 129 for _ in range(129)]
    for i in range(2, 129):
        for j in range(i, 129):
            gcd_val[i][j] = math.gcd(i, j)
            gcd_val[j][i] = gcd_val[i][j]
            
    def try_add(m, current_set):
        """
        current_set: list of (r, m) tuples.
        Returns a valid residue r for m if possible, else None.
        """
        # Optimization: if current_set is empty, return 0
        if not current_set:
            return 0
            
        for r in range(m):
            valid = True
            for r_j, m_j in current_set:
                g = gcd_val[m][m_j]
                if (r - r_j) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def greedy(order):
        current_set = []
        for m in order:
            r = try_add(m, current_set)
            if r is not None:
                current_set.append((r, m))
        return current_set

    best_set = []
    best_size = 0
    
    # Heuristic orders
    heuristic_orders = [
        sorted(moduli, reverse=True),
        sorted(moduli),
        # Sort by number of divisors
        sorted(moduli, key=lambda x: -sum(1 for i in range(2, x+1) if x % i == 0)),
        sorted(moduli, key=lambda x: sum(1 for i in range(2, x+1) if x % i == 0))
    ]
    
    # Run heuristic orders first
    for order in heuristic_orders:
        if time.time() > deadline:
            break
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Random restarts
    # Estimate time per restart.
    # If it's too slow, reduce count.
    # Let's try 10000 restarts, but check time.
    
    # We can also randomize the residue choice if multiple are valid?
    # Currently try_add returns the first valid r.
    # We could pick a random valid r to diversify.
    
    def try_add_random(m, current_set, rng):
        if not current_set:
            return 0
        valid_rs = []
        for r in range(m):
            valid = True
            for r_j, m_j in current_set:
                g = gcd_val[m][m_j]
                if (r - r_j) % g == 0:
                    valid = False
                    break
            if valid:
                valid_rs.append(r)
        if valid_rs:
            return rng.choice(valid_rs)
        return None

    def greedy_random(order, rng):
        current_set = []
        for m in order:
            r = try_add_random(m, current_set, rng)
            if r is not None:
                current_set.append((r, m))
        return current_set

    num_restarts = 20000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        random_order = moduli.copy()
        random.shuffle(random_order)
        rng = random.Random(i) # Reproducible per restart? Or just use global random
        # Using global random is fine
        
        # Alternate between deterministic and random residue selection?
        # Random residue selection might find different sets.
        if i % 2 == 0:
            s = greedy(random_order)
        else:
            s = greedy_random(random_order, random)
            
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    result = [[r, m] for r, m in best_set]
    
    if not result:
        result = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return result
