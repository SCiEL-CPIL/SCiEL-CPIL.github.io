import time
import random

def construct():
    deadline = time.time() + 110
    mandatory = {1, 2, 5, 11}
    best_set = set(mandatory)
    best_size = 4
    
    def get_diffs(S):
        D = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                D.add(abs(S_list[i] - S_list[j]))
        return D

    def can_add(S, D, x):
        if x in S:
            return False
        for y in S:
            if abs(x - y) in D:
                return False
        return True

    def add_diffs(D, S, x):
        for y in S:
            D.add(abs(x - y))

    def remove_diffs(D, S, y):
        for z in S:
            if z != y:
                D.discard(abs(y - z))

    def can_swap(S, D, y, z):
        if z in S:
            return False
        
        # Diffs remaining in S without y
        D_rem = set(D)
        for s in S:
            if s != y:
                D_rem.discard(abs(y - s))
        
        # Check z against S\{y} using D_rem
        for u in S:
            if u == y:
                continue
            if abs(z - u) in D_rem:
                return False
        return True

    def swap_diffs(D, S, y, z):
        # Remove diffs involving y
        for s in S:
            if s != y:
                D.discard(abs(y - s))
        # Add diffs involving z
        for s in S:
            if s != y:
                D.add(abs(z - s))

    while time.time() < deadline:
        # Start from best_set, perturb if size > 4
        if len(best_set) > 4:
            S = set(best_set)
            D = get_diffs(S)
            # Perturb: remove 1 or 2 random non-mandatory elements
            candidates = [x for x in S if x not in mandatory]
            if candidates:
                # Remove 1 random
                y = random.choice(candidates)
                S.remove(y)
                remove_diffs(D, S, y)
        else:
            S = set(mandatory)
            D = get_diffs(S)
            
        # Local search
        no_progress = 0
        while time.time() < deadline and no_progress < 1000:
            improved = False
            
            # Try to add
            for _ in range(100):
                x = random.randint(1, 284)
                if can_add(S, D, x):
                    S.add(x)
                    add_diffs(D, S, x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = set(S)
                        if best_size >= 20:
                            return list(best_set)
                    break
            
            if not improved:
                # Try to swap
                candidates = [x for x in S if x not in mandatory]
                if candidates:
                    y = random.choice(candidates)
                    swapped = False
                    for _ in range(100):
                        z = random.randint(1, 284)
                        if can_swap(S, D, y, z):
                            S.remove(y)
                            S.add(z)
                            swap_diffs(D, S, y, z)
                            swapped = True
                            break
                    if not swapped:
                        # Remove y
                        S.remove(y)
                        remove_diffs(D, S, y)
                
                no_progress += 1
            else:
                no_progress = 0
                
    return list(best_set)
