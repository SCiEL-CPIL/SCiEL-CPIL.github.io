import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85
    n = 128
    nums = list(range(1, n + 1))
    
    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute conflict count for each element
    # conflict_count[x] = number of pairs (a,b) with a<b, a!=x, b!=x
    # such that gcd(a,b) == gcd(a,x) == gcd(b,x)
    conflict_count = [0] * (n + 1)
    for x in range(1, n + 1):
        for a in range(1, n + 1):
            if a == x:
                continue
            g_ax = gcd_table[a][x]
            for b in range(a + 1, n + 1):
                if b == x:
                    continue
                if gcd_table[a][b] == g_ax and gcd_table[b][x] == g_ax:
                    conflict_count[x] += 1
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        s_list = list(S)
        m = len(s_list)
        for i in range(m):
            a = s_list[i]
            g_ax = gcd_table[a][x]
            for j in range(i + 1, m):
                b = s_list[j]
                if gcd_table[a][b] == g_ax and gcd_table[b][x] == g_ax:
                    return False
        return True

    def greedy_build(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S, time_budget=2.0):
        """Improve set S via additions."""
        S = set(S)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_budget and time.time() < deadline - 3:
            improved = False
            for x in range(1, n + 1):
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
                        break
        return S

    best_size = 0
    best_set = set()
    
    # Strategy 1: Greedy by conflict count (fewest conflicts first)
    order_by_conflict = sorted(nums, key=lambda x: conflict_count[x])
    S = greedy_build(order_by_conflict)
    S = local_search(S, time_budget=3.0)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Strategy 2: Multiple random restarts
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        order = nums[:]
        random.shuffle(order)
        
        S = greedy_build(order)
        S = local_search(S, time_budget=0.3)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Final local search on the best set found
    best_set = local_search(best_set, time_budget=5.0)
    
    return list(best_set)
