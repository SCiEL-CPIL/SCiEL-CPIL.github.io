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

def construct():
    best_set = []
    # Try a few deterministic orders
    # 1. Natural order
    # 2. Reverse order
    # 3. Random orders
    
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
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
        if len(S) > len(best_set):
            best_set = S
            
    return best_set
