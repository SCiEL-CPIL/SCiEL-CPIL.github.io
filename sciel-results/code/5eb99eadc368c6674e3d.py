# EVOLVE-BLOCK-START
from math import gcd
import random

def construct():
    N = 128
    random.seed(42)

    # Precompute lcm for all pairs
    lcms = [[0]*N for _ in range(N)]
    for i in range(1, N):
        for j in range(i+1, N):
            lcms[i-1][j-1] = (i*j)//gcd(i,j)
            lcms[j-1][i-1] = lcms[i-1][j-1]
    
    # Precompute conflicts
    conflicts = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(a+1, N):
            L = lcms[a-1][b-1]
            for c in range(1, N):
                if c == a or c == b:
                    continue
                if lcms[a-1][c-1] == L and lcms[b-1][c-1] == L:
                    ac, bc, cc = a-1, b-1, c-1
                    conflicts[ac].add(frozenset((bc, cc)))
                    conflicts[bc].add(frozenset((ac, cc)))
                    conflicts[cc].add(frozenset((ac, bc)))
    
    def can_add(x, S):
        for pair in conflicts[x]:
            a, b = tuple(pair)
            if a in S and b in S:
                return False
        return True

    def greedy_solve(order):
        S = set()
        for idx in order:
            if can_add(idx, S):
                S.add(idx)
        return S

    def local_search(S):
        S = set(S)
        max_iter = 200
        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1
            improved = False
            
            # Try to add any vertex
            for x in range(N):
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try to swap
            valid_swaps = []
            # Optimize: iterate over S, then U
            S_list = list(S)
            U_list = [u for u in range(N) if u not in S]
            
            for v in S_list:
                for u in U_list:
                    ok = True
                    for pair in conflicts[u]:
                        a, b = tuple(pair)
                        if a in S and b in S:
                            if a != v and b != v:
                                ok = False
                                break
                    if ok:
                        valid_swaps.append((v, u))
                        break # One swap per v is enough to diversify? 
                        # Actually, breaking here limits the options. 
                        # But it speeds up.
            
            if valid_swaps:
                v, u = random.choice(valid_swaps)
                S.remove(v)
                S.add(u)
                continue
            
            break
            
        return S

    best_set = set()
    
    # 1. Degree based order
    degree = [len(conflicts[i]) for i in range(N)]
    order_deg = sorted(range(N), key=lambda i: (degree[i], i))
    S1 = greedy_solve(order_deg)
    S1 = local_search(S1)
    if len(S1) > len(best_set):
        best_set = S1

    # 2. Random orders
    for _ in range(20):
        order_rand = list(range(N))
        random.shuffle(order_rand)
        S_r = greedy_solve(order_rand)
        S_r = local_search(S_r)
        if len(S_r) > len(best_set):
            best_set = S_r
            
    # 3. Reverse degree
    order_rev = sorted(range(N), key=lambda i: (-degree[i], i))
    S2 = greedy_solve(order_rev)
    S2 = local_search(S2)
    if len(S2) > len(best_set):
        best_set = S2

    result = sorted(i+1 for i in best_set)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
