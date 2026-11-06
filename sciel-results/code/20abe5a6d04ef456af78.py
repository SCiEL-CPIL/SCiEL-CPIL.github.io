# EVOLVE-BLOCK-START
import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    moduli = list(range(2, 129))
    
    # Precompute gcds
    gcd_table = [[0]*129 for _ in range(129)]
    for i in range(2, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    best_solution = None
    best_size = 0
    
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5)+1):
            if n % i == 0:
                count += 1
                if i != n//i:
                    count += 1
        return count

    # Deterministic heuristics
    orders = [
        list(range(2, 129)),
        list(range(128, 1, -1)),
        sorted(moduli, key=lambda x: -num_divisors(x)),
        sorted(moduli, key=lambda x: x.bit_count(), reverse=True)
    ]
    
    for order in orders:
        if time.time() > deadline: break
        sol = _greedy(order, gcd_table, deterministic=True)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol

    # Randomized restarts
    while time.time() < deadline:
        random_order = moduli[:]
        random.shuffle(random_order)
        sol = _greedy(random_order, gcd_table, deterministic=False)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol

    if best_solution is None:
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return [[r, m] for r, m in best_solution]

def _greedy(order, gcd_table, deterministic):
    selected = []
    
    for m in order:
        if not selected:
            selected.append((0, m))
            continue
            
        if deterministic:
            for r in range(m):
                ok = True
                for r_k, m_k in selected:
                    g = gcd_table[m][m_k]
                    if r % g == r_k % g:
                        ok = False
                        break
                if ok:
                    selected.append((r, m))
                    break
        else:
            valid_r = []
            for r in range(m):
                ok = True
                for r_k, m_k in selected:
                    g = gcd_table[m][m_k]
                    if r % g == r_k % g:
                        ok = False
                        break
                if ok:
                    valid_r.append(r)
            
            if valid_r:
                r = random.choice(valid_r)
                selected.append((r, m))
                
    return selected
# EVOLVE-BLOCK-END
