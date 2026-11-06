import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    best_set = set()
    best_score = 0.0
    
    def calc_score(S):
        return sum(1.0/x for x in S)
    
    def is_compatible(z, S):
        for a in S:
            # Case 1: a, z, b => b = 2z - a
            b = 2*z - a
            if b != a and b in S:
                return False
            # Case 2: a, b, z => b = (a+z)/2
            if (a+z) % 2 == 0:
                b = (a+z)//2
                if b != a and b in S:
                    return False
            # Case 3: z, a, b => b = 2a - z
            b = 2*a - z
            if b != a and b in S:
                return False
        return True

    def greedy_init():
        S = set()
        for x in range(1, N+1):
            if is_compatible(x, S):
                S.add(x)
        return S

    def local_search(S):
        S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any compatible element not in S
            # Iterate z from 1 to N, add if compatible
            # This ensures we always pick the smallest available z (highest reciprocal)
            for z in range(1, N+1):
                if z not in S and is_compatible(z, S):
                    S.add(z)
                    improved = True
                    break
            
            # 2. Try swaps
            if not improved:
                S_list = list(S)
                for y in S_list:
                    # Try to replace y with z < y
                    for z in range(1, y):
                        if z in S:
                            continue
                        S_temp = S - {y}
                        if is_compatible(z, S_temp):
                            S.remove(y)
                            S.add(z)
                            improved = True
                            break
                    if improved:
                        break
                        
        return S

    # Initial run
    S = greedy_init()
    S = local_search(S)
    best_score = calc_score(S)
    best_set = S

    # Restart loop with perturbation
    while time.time() < deadline:
        # Perturb best_set
        S_perturbed = set(best_set)
        if not S_perturbed:
            continue
            
        # Remove 1-2 random elements
        num_removals = random.randint(1, 2)
        elements_to_remove = random.sample(list(S_perturbed), num_removals)
        for e in elements_to_remove:
            S_perturbed.remove(e)
        
        # Re-optimize
        S_perturbed = local_search(S_perturbed)
        
        score = calc_score(S_perturbed)
        if score > best_score:
            best_score = score
            best_set = S_perturbed

    return sorted(best_set)
