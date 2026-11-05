"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def try_greedy(order, seed):
        random.seed(seed)
        current = []
        for m in order:
            valid_r = []
            for r in range(m):
                ok = True
                for r2, m2 in current:
                    g = gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    valid_r.append(r)
            if valid_r:
                r_choice = random.choice(valid_r)
                current.append((r_choice, m))
        return current

    best_sol = []
    best_len = 0
    
    candidates = list(range(2, 129))
    
    # Fallback to seed if nothing found
    seed_sol = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    best_sol = seed_sol
    best_len = len(seed_sol)
    
    # 1. Try multiples of 6
    mult6 = [m for m in candidates if m % 6 == 0]
    sol = try_greedy(mult6, 12345)
    if len(sol) > best_len:
        best_len = len(sol)
        best_sol = sol
        
    # 2. Try multiples of 2
    mult2 = [m for m in candidates if m % 2 == 0]
    sol = try_greedy(mult2, 54321)
    if len(sol) > best_len:
        best_len = len(sol)
        best_sol = sol
        
    # 3. Random restarts
    for i in range(5000):
        if time.time() > deadline:
            break
        order = candidates[:]
        random.shuffle(order)
        sol = try_greedy(order, i)
        if len(sol) > best_len:
            best_len = len(sol)
            best_sol = sol
            
    # 4. Ascending order
    order_asc = sorted(candidates)
    sol = try_greedy(order_asc, 99999)
    if len(sol) > best_len:
        best_len = len(sol)
        best_sol = sol
        
    # 5. Descending order
    order_desc = sorted(candidates, reverse=True)
    sol = try_greedy(order_desc, 88888)
    if len(sol) > best_len:
        best_len = len(sol)
        best_sol = sol

    return [[r, m] for r, m in best_sol]
# EVOLVE-BLOCK-END
