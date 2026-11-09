import random
from math import lcm
from itertools import combinations

def construct():
    # Precompute lcms for all pairs in 1..128
    # This is 128*128/2 ~ 8k values.
    # Storing in a dictionary or 2D list.
    max_n = 128
    lcm_table = [[1]*(max_n+1) for _ in range(max_n+1)]
    for i in range(1, max_n+1):
        for j in range(i, max_n+1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    def greedy_select(order):
        selected = []
        for x in order:
            ok = True
            # Check against all pairs in selected
            # Optimization: if len(selected) is small, this is fast.
            for i in range(len(selected)):
                a = selected[i]
                for j in range(i+1, len(selected)):
                    b = selected[j]
                    if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.append(x)
        return selected

    best_set = []
    
    # 1. Standard orders
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1))
    ]
    
    # 2. Sort by number of divisors? Or prime factors?
    # Numbers with few prime factors (primes, prime powers) are "safer".
    # Let's try sorting by number of prime factors (ascending).
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
        
    orders.append(sorted(range(1, 129), key=num_prime_factors))
    orders.append(sorted(range(1, 129), key=num_prime_factors, reverse=True))
    
    # 3. Random shuffles
    random.seed(42) # Reproducible for evaluation, but allows multiple runs?
    # Actually, seed should be fixed or random? 
    # "Each evaluation must finish within 60s".
    # Using a fixed seed ensures deterministic behavior, which is good for reproducibility.
    # But we want to explore. Let's use a few fixed seeds.
    for seed_val in [1, 2, 3, 4, 5]:
        random.seed(seed_val)
        ord = list(range(1, 129))
        random.shuffle(ord)
        orders.append(ord)
        
    # Run more randoms
    for _ in range(50):
        ord = list(range(1, 129))
        random.shuffle(ord)
        orders.append(ord)

    for ord in orders:
        s = greedy_select(ord)
        if len(s) > len(best_set):
            best_set = s
            
    return best_set
