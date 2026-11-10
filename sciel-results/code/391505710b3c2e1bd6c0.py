import random
import time
from math import gcd

N = 128
MAX_T = 85.0  # wall-clock budget for construct()

# Precompute gcd matrix and divisors/multiples lists
gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        g = gcd(i, j)
        gcd_mat[i][j] = g
        gcd_mat[j][i] = g

# For each x, precompute list of (a, d) where d = gcd(x, a)
# Also precompute for each x and each d, the set of a with gcd(x,a)=d
x_gcd_a = {}  # x -> list of (a, d)
for x in range(1, N + 1):
    lst = []
    for a in range(1, N + 1):
        if a == x:
            continue
        d = gcd_mat[x][a]
        lst.append((a, d))
    x_gcd_a[x] = lst

# Precompute for each x, group by d: x_d_groups[x][d] = list of a with gcd(x,a)=d
x_d_groups = {}
for x in range(1, N + 1):
    groups = {}
    for a, d in x_gcd_a[x]:
        if d not in groups:
            groups[d] = []
        groups[d].append(a)
    x_d_groups[x] = groups

# Precompute for each pair (a,b) with a<b, the gcd
# Already in gcd_mat

def can_add(S_set, x):
    """
    Check if x can be added to S_set without creating a violation.
    A violation exists if there are a, b in S_set such that
    gcd(x,a) == gcd(x,b) == gcd(a,b) = d.
    
    For each d, consider S_d = {a in S_set : gcd(x,a) = d}.
    We need to check if there exist a, b in S_d with gcd(a,b) = d.
    """
    groups = x_d_groups[x]
    for d, a_list in groups.items():
        # Filter to those in S_set
        S_d = [a for a in a_list if a in S_set]
        if len(S_d) < 2:
            continue
        # Check pairs in S_d
        n = len(S_d)
        for i in range(n):
            ai = S_d[i]
            for j in range(i + 1, n):
                aj = S_d[j]
                if gcd_mat[ai][aj] == d:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + MAX_T - 5.0  # safety margin
    
    best_S = set()
    best_size = 0
    
    # Multi-start greedy with different random seeds
    seed_counter = 0
    
    while time.time() < deadline:
        seed_counter += 1
        random.seed(seed_counter * 7919)  # Different seed each run
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        S = set()
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        # If we've found a large set, do local search to try to grow it
        if best_size >= 40:
            # Try to add more elements
            remaining = [x for x in range(1, N + 1) if x not in S]
            if remaining:
                random.shuffle(remaining)
                for x in remaining:
                    if can_add(S, x):
                        S.add(x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
    
    return list(best_S)
