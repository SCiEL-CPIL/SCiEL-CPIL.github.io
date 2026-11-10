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
            
    # Precompute conflicts: for each x, store pairs (a,b) where (a,b,x) is a bad triple
    conflicts = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x: continue
            ga_x = gcds[a][x]
            for b in range(a + 1, N + 1):
                if b == x: continue
                if gcds[a][b] == ga_x and gcds[b][x] == ga_x:
                    conflicts[x].add((a, b))
                    
    # For faster addition check: for each x, store the conflict pairs as a set of frozensets
    # and also precompute for each element u, which x's conflicts involve u
    # Actually, let's precompute a different structure:
    # For each x, we need to check if any pair (a,b) in conflicts[x] is fully in S.
    # Alternative: for each x, store conflicts as list of (a,b) and check membership.
    
    def is_valid_addition(S_set, x):
        """Check if x can be added to S_set without creating a bad triple."""
        for a, b in conflicts[x]:
            if a in S_set and b in S_set:
                return False
        return True
    
    def count_conflicts_for_addition(S_set, x):
        """Count how many conflict pairs of x are fully in S_set."""
        count = 0
        for a, b in conflicts[x]:
            if a in S_set and b in S_set:
                count += 1
        return count

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

    def local_search(S, max_iter=20000):
        """Improved local search with systematic swap attempts."""
        S = set(S)
        current_size = len(S)
        local_best_size = current_size
        local_best_set = set(S)
        no_improve = 0
        
        # Precompute for each x not in S, the number of conflicts with S
        # This helps prioritize which elements to try adding
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to add an element
            not_in_S = [x for x in elements if x not in S]
            if not_in_S:
                # Sort by number of conflicts (fewer conflicts first)
                not_in_S.sort(key=lambda x: count_conflicts_for_addition(S, x))
                added = False
                for x in not_in_S:
                    if is_valid_addition(S, x):
                        S.add(x)
                        current_size += 1
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_set = set(S)
                        no_improve = 0
                        added = True
                        break
                
                if added:
                    continue
            
            # If no add possible, try swaps: remove u, add v
            # Systematically try each u in S and each v not in S
            swaps_tried = 0
            max_swaps_per_iter = 50
            for u in list(S):
                if swaps_tried >= max_swaps_per_iter:
                    break
                S.remove(u)
                current_size -= 1
                
                # Try to add some v
                not_in_S = [x for x in elements if x not in S]
                # Sort by conflicts
                not_in_S.sort(key=lambda x: count_conflicts_for_addition(S, x))
                added_v = None
                for v in not_in_S:
                    if is_valid_addition(S, v):
                        S.add(v)
                        current_size += 1
                        added_v = v
                        break
                
                if added_v is not None and current_size > local_best_size:
                    local_best_size = current_size
                    local_best_set = set(S)
                    no_improve = 0
                else:
                    # Undo the swap
                    if added_v is not None:
                        S.remove(added_v)
                        current_size -= 1
                    S.add(u)
                    current_size += 1
                    no_improve += 1
                swaps_tried += 1
            
            # If many swaps failed, try random removal
            if no_improve > 50 and S:
                u = random.choice(list(S))
                S.remove(u)
                current_size -= 1
                no_improve = 0
                
        return local_best_set

    # Run multiple restarts
    for run in range(50):
        if time.time() > deadline - 5:
            break
            
        S = greedy_init()
        S = local_search(S)
        
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
            
    # Final polish: try to add any remaining elements
    not_in_best = [x for x in elements if x not in best_set]
    random.shuffle(not_in_best)
    for x in not_in_best:
        if is_valid_addition(best_set, x):
            best_set.add(x)
    
    return list(best_set)
