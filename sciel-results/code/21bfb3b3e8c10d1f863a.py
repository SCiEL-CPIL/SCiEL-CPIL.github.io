import time
import random
from itertools import combinations

def construct():
    # Generate all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    all_sets.append(mask)
    
    # Helper to check if a set C can be added to family F (list of masks)
    # Sunflower condition: No three distinct sets A, B, C have A&B = A&C = B&C.
    # For fixed C and A in F, let K = A & C.
    # A sunflower with C and A requires a B in F such that:
    # 1. B & A = K
    # 2. B & C = K
    # This is equivalent to:
    # B contains K (B & K == K)
    # B is disjoint from (A \ K) and (C \ K)
    # Let R = (A | C) ^ K. Then B & R == 0.
    
    def can_add(C, F):
        for A in F:
            K = A & C
            R = (A | C) ^ K
            for B in F:
                if (B & K) == K and (B & R) == 0:
                    return False
        return True

    def add_set(C, F):
        F.append(C)

    def remove_set(C, F):
        F.remove(C)

    deadline = time.time() + 55.0  # Safety margin
    best_family = []
    best_size = 0
    random.seed(42)

    # Multiple random restarts
    num_restarts = 10
    restarts_done = 0
    
    while time.time() < deadline and restarts_done < num_restarts:
        F = []
        F_set = set()
        
        # Greedy phase with random ordering
        candidates = all_sets[:]
        random.shuffle(candidates)
        
        for C in candidates:
            if C in F_set:
                continue
            if can_add(C, F):
                add_set(C, F)
                F_set.add(C)
        
        current_size = len(F)
        if current_size > best_size:
            best_size = current_size
            best_family = F[:]
        
        # Local Search: 1-swap
        # We try to improve the current family by swapping sets
        # We limit the number of steps to avoid spending too much time per restart
        steps = 0
        max_steps = 500
        
        while steps < max_steps and time.time() < deadline - 5.0:
            steps += 1
            # Pick a random set to remove
            if not F:
                break
            A_remove = random.choice(F)
            remove_set(A_remove, F)
            F_set.remove(A_remove)
            
            # Try to add a random set
            improved = False
            for _ in range(20):
                C = random.choice(all_sets)
                if C in F_set:
                    continue
                if can_add(C, F):
                    add_set(C, F)
                    F_set.add(C)
                    improved = True
                    if len(F) > best_size:
                        best_size = len(F)
                        best_family = F[:]
                    break
            
            if not improved:
                # Revert
                add_set(A_remove, F)
                F_set.add(A_remove)
            
            # Early stop if we hit a local maximum and can't add anything easily?
            # For now, just keep swapping.

        restarts_done += 1

    # Convert best_family to result format
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
