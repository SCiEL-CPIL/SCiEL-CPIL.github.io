from math import gcd
import random
import time

# Precompute GCD matrix for 1..128
N = 128
gcd_mat = [[0]*(N+1) for _ in range(N+1)]
for i in range(1, N+1):
    for j in range(i, N+1):
        g = gcd(i, j)
        gcd_mat[i][j] = g
        gcd_mat[j][i] = g

def can_add(x, S, gcd_mat):
    """
    Check if x can be added to S without creating a bad triple.
    A bad triple (a, b, x) has gcd(a,b) == gcd(a,x) == gcd(b,x).
    """
    n = len(S)
    for i in range(n):
        a = S[i]
        g_ax = gcd_mat[a][x]
        for j in range(i+1, n):
            b = S[j]
            if gcd_mat[a][b] == g_ax and gcd_mat[b][x] == g_ax:
                return False
    return True

def local_improve(S, gcd_mat, time_limit=1.0):
    S_list = list(S)
    S_set = set(S_list)
    improved = True
    iterations = 0
    max_iterations = 200
    start_time = time.time()
    
    while improved and iterations < max_iterations and (time.time() - start_time) < time_limit:
        improved = False
        iterations += 1
        
        # 1. Try to add any single element
        for x in range(1, N+1):
            if x not in S_set and can_add(x, S_list, gcd_mat):
                S_list.append(x)
                S_set.add(x)
                improved = True
                break
        
        if improved:
            continue

        # 2. Try 1-for-1 swap
        if len(S_list) > 0:
            # Try a few random swaps
            attempts = 0
            max_attempts = 20
            while attempts < max_attempts:
                attempts += 1
                x = random.choice(S_list)
                y = random.randint(1, N)
                if y in S_set:
                    continue
                S_temp_list = [u for u in S_list if u != x]
                S_temp_set = S_set - {x}
                
                if can_add(y, S_temp_list, gcd_mat):
                    S_list = S_temp_list + [y]
                    S_set = S_temp_set | {y}
                    improved = True
                    break

        if improved:
            continue

        # 3. Try 1-for-2 swap (Remove 1, Add 2) - Net gain +1
        if len(S_list) > 0:
            attempts = 0
            max_attempts = 10
            while attempts < max_attempts:
                attempts += 1
                x = random.choice(S_list)
                S_temp_list = [u for u in S_list if u != x]
                S_temp_set = S_set - {x}
                
                added = 0
                new_S_list = list(S_temp_list)
                new_S_set = set(S_temp_set)
                
                # Try to add 2 elements
                candidates = list(range(1, N+1))
                random.shuffle(candidates)
                for y in candidates:
                    if y in new_S_set:
                        continue
                    if can_add(y, new_S_list, gcd_mat):
                        new_S_list.append(y)
                        new_S_set.add(y)
                        added += 1
                        if added == 2:
                            break
                            
                if added == 2:
                    S_list = new_S_list
                    S_set = new_S_set
                    improved = True
                    break

        if improved:
            continue

    return S_list

def construct():
    best_set = []
    
    def count_divisors(n):
        count = 0
        for i in range(1, int(n**0.5)+1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    def smallest_prime_factor(n):
        if n % 2 == 0:
            return 2
        i = 3
        while i * i <= n:
            if n % i == 0:
                return i
            i += 2
        return n

    div_counts = [count_divisors(i) for i in range(N+1)]
    spf = [smallest_prime_factor(i) for i in range(N+1)]
    
    order_by_div = list(range(1, N+1))
    order_by_div.sort(key=lambda x: div_counts[x])
    
    order_by_spf = list(range(1, N+1))
    order_by_spf.sort(key=lambda x: spf[x])
    
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        order_by_div,
        list(reversed(order_by_div)),
        order_by_spf,
        list(reversed(order_by_spf)),
    ]
    
    # Add random orders
    random.seed(42)
    for _ in range(150):
        cand = list(range(1, 129))
        random.shuffle(cand)
        orders.append(cand)
        
    total_start = time.time()
    
    for idx, order in enumerate(orders):
        if time.time() - total_start > 55:
            break
            
        S = []
        for x in order:
            if can_add(x, S, gcd_mat):
                S.append(x)
                
        # Local improvement with time limit
        remaining = 55 - (time.time() - total_start)
        t_limit = min(0.5, remaining / max(1, len(orders) - idx))
        
        S = local_improve(S, gcd_mat, time_limit=t_limit)
        
        if len(S) > len(best_set):
            best_set = S
            
    return best_set
