import time
import random
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute conflicts
    # conflicts[x] is a list of pairs (a, b) such that (a, b, x) is a bad triple.
    # We store a set of frozensets for faster lookup?
    # Or just list of pairs.
    conflicts = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x: continue
            ga_x = gcds[a][x]
            for b in range(a + 1, N + 1):
                if b == x: continue
                if gcds[a][b] == ga_x and gcds[b][x] == ga_x:
                    conflicts[x].add((a, b))
                    
    def is_valid_addition(S_set, x):
        for a, b in conflicts[x]:
            if a in S_set and b in S_set:
                return False
        return True

    start_time = time.time()
    deadline = start_time + 55
    random.seed(42)
    
    best_set = set()
    best_size = 0
    
    def greedy_init():
        S = set()
        elems = elements[:]
        random.shuffle(elems)
        for x in elems:
            if is_valid_addition(S, x):
                S.add(x)
        return S

    def tabu_search(S, max_iter=5000):
        S = set(S)
        current_size = len(S)
        tabu_list = {} # element -> time when it can be added again
        tabu_tenure = 10
        no_improve = 0
        local_best_size = current_size
        local_best_set = set(S)
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to add an element
            not_in_S = [x for x in elements if x not in S]
            if not_in_S:
                random.shuffle(not_in_S)
                added = False
                for x in not_in_S:
                    if is_valid_addition(S, x):
                        # Check tabu
                        if tabu_list.get(x, 0) <= i:
                            S.add(x)
                            current_size += 1
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_set = set(S)
                            no_improve = 0
                            added = True
                            break
                        else:
                            # Aspiration: if it improves global best?
                            # We don't track global best here, but we can check if it's a new local best
                            # For now, skip if tabu
                            pass
                
                if added:
                    continue
            
            # If no add, try a swap (remove u, add v)
            # Or just remove a random element to escape
            if no_improve > 20:
                # Remove a random element
                if S:
                    u = random.choice(list(S))
                    S.remove(u)
                    current_size -= 1
                    tabu_list[u] = i + tabu_tenure
                    no_improve = 0
            else:
                no_improve += 1
                
        return local_best_set

    # Run multiple restarts
    for run in range(200):
        if time.time() > deadline - 5:
            break
            
        S = greedy_init()
        S = tabu_search(S)
        
        if len(S) > len(best_set):
            best_set = set(S)
            
    # Final polish
    not_in_best = [x for x in elements if x not in best_set]
    random.shuffle(not_in_best)
    for x in not_in_best:
        if is_valid_addition(best_set, x):
            best_set.add(x)

    return list(best_set)
