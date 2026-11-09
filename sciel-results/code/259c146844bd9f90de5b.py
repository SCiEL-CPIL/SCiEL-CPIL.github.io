import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    # Precompute GCDs
    G = [[gcd(i, j) for j in range(1, N + 1)] for i in range(1, N + 1)]
    
    # Precompute conflicts
    # conflicts[i][j] is a set of k such that {i, j, k} is a bad triple
    conflicts = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            d = G[i][j]
            for k in range(1, N + 1):
                if k == i or k == j:
                    continue
                if G[i][k] == d and G[j][k] == d:
                    conflicts[i][j].add(k)
                    conflicts[j][i].add(k)

    def greedy_init(order):
        S = set()
        for x in order:
            valid = True
            for i in S:
                if S.intersection(conflicts[i][x]):
                    valid = False
                    break
            if valid:
                S.add(x)
        return S

    def local_search(S):
        S = set(S)
        no_improve_count = 0
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add an element
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                can_add = True
                for i in S:
                    if S.intersection(conflicts[i][x]):
                        can_add = False
                        break
                if can_add:
                    S.add(x)
                    improved = True
                    no_improve_count = 0
                    break # Restart to try adding more
            
            if improved:
                continue
                
            # 2. Try a swap
            if no_improve_count > 50:
                S_list = list(S)
                if S_list:
                    # Try a few random swaps
                    for _ in range(20):
                        i = random.choice(S_list)
                        # Pick j not in S
                        # To be efficient, we can just pick random j and check
                        j = random.randint(1, N)
                        if j in S:
                            continue
                        
                        new_S = S - {i}
                        # Check if j conflicts with any pair in new_S
                        valid_swap = True
                        for a in new_S:
                            if new_S.intersection(conflicts[a][j]):
                                valid_swap = False
                                break
                        if valid_swap:
                            S = new_S
                            S.add(j)
                            improved = True
                            no_improve_count = 0
                            break
                    if improved:
                        continue

            no_improve_count += 1
            if no_improve_count > 2000:
                break

        return S

    best_set = set()
    
    # Run multiple random restarts
    while time.time() < deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S0 = greedy_init(order)
        S1 = local_search(S0)
        
        if len(S1) > len(best_set):
            best_set = S1
            
    return list(best_set)
