from math import gcd
import random

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

def local_improve(S, gcd_mat):
    S_list = list(S)
    S_set = set(S_list)
    
    improved = True
    while improved:
        improved = False
        
        # 1. Try to add any single element
        for x in range(1, N+1):
            if x not in S_set and can_add(x, S_list, gcd_mat):
                S_list.append(x)
                S_set.add(x)
                improved = True
                break
        
        if improved:
            continue

        # 2. Try 1-for-1 swap to enable a future add
        # We look for x in S, y not in S such that S' = S \ {x} + {y} is valid
        # AND there exists z not in S' such that S'' = S' + {z} is valid.
        
        for x in S_list:
            S_temp_list = [u for u in S_list if u != x]
            S_temp_set = S_set - {x}
            
            for y in range(1, N+1):
                if y in S_set:
                    continue
                if can_add(y, S_temp_list, gcd_mat):
                    # This swap is valid. Check if we can add z.
                    new_S_list = S_temp_list + [y]
                    new_S_set = S_temp_set | {y}
                    
                    for z in range(1, N+1):
                        if z in new_S_set:
                            continue
                        if can_add(z, new_S_list, gcd_mat):
                            S_list = new_S_list + [z]
                            S_set = new_S_set | {z}
                            improved = True
                            break
                    
                    if improved:
                        break
            
            if improved:
                break
        
        if improved:
            continue

        # 3. Try 1-for-2 swap (Remove 1, Add 2) - Net gain +1
        for x in S_list:
            S_temp_list = [u for u in S_list if u != x]
            S_temp_set = S_set - {x}
            
            new_S_list = list(S_temp_list)
            new_S_set = set(S_temp_set)
            added = 0
            
            # Find first y
            first_added = None
            for y in range(1, N+1):
                if y in S_temp_set:
                    continue
                if can_add(y, new_S_list, gcd_mat):
                    first_added = y
                    break
            
            if first_added is None:
                continue
                
            new_S_list.append(first_added)
            new_S_set.add(first_added)
            added += 1
            
            # Find second z
            for z in range(1, N+1):
                if z in new_S_set:
                    continue
                if can_add(z, new_S_list, gcd_mat):
                    new_S_list.append(z)
                    new_S_set.add(z)
                    added += 1
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

    div_counts = [count_divisors(i) for i in range(N+1)]
    order_by_div = list(range(1, N+1))
    order_by_div.sort(key=lambda x: div_counts[x])
    
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        order_by_div,
        list(reversed(order_by_div)),
    ]
    
    # Add random orders
    for _ in range(20):
        cand = list(range(1, 129))
        random.shuffle(cand)
        orders.append(cand)
        
    for order in orders:
        S = []
        for x in order:
            if can_add(x, S, gcd_mat):
                S.append(x)
                
        # Local improvement
        S = local_improve(S, gcd_mat)
        
        if len(S) > len(best_set):
            best_set = S
            
    return best_set
