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
    n = len(S)
    for i in range(n):
        a = S[i]
        g_ax = gcd_mat[a][x]
        for j in range(i+1, n):
            b = S[j]
            if gcd_mat[a][b] == g_ax == gcd_mat[b][x]:
                return False
    return True

def local_improve(S, gcd_mat):
    S_list = list(S)
    S_set = set(S_list)
    improved = True
    iterations = 0
    # Limit iterations to ensure speed
    while improved and iterations < 10:
        improved = False
        iterations += 1
        
        # Try to add any single element
        for x in range(1, N+1):
            if x not in S_set and can_add(x, S_list, gcd_mat):
                S_list.append(x)
                S_set.add(x)
                improved = True
                break
        if improved:
            continue
            
        # Try to remove 1 and add 2 (net gain +1)
        # Sample a few removal candidates to save time
        num_removals_to_try = min(5, len(S_list))
        if num_removals_to_try > 0:
            candidates_to_remove = random.sample(S_list, num_removals_to_try)
            for x in candidates_to_remove:
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
    return S_list

def construct():
    best_set = []
    
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
    ]
    
    # Add random orders
    for _ in range(50):
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
