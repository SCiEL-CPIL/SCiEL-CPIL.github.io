# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free subset of Z/205Z.
    
    Uses Simulated Annealing to maximize the size of the independent set
    in the conflict graph defined by forbidden square differences.
    """
    m = 205
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)
    
    # Precompute conflict matrix
    # conflict[i][j] = True if i and j cannot be in the same set
    conflict = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            d = (i - j) % m
            if d in forbidden:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Helper to check if node u is compatible with all nodes in the set S
    # S is a set of integers
    def is_compatible(u, S):
        for v in S:
            if conflict[u][v]:
                return False
        return True

    # Helper to check if u is compatible with all nodes in S except v
    def is_compatible_except(u, S, v):
        for w in S:
            if w == v:
                continue
            if conflict[u][w]:
                return False
        return True

    # Initial state: Greedy construction
    def greedy_initial():
        S = set()
        candidates = list(range(m))
        random.shuffle(candidates)
        for u in candidates:
            if is_compatible(u, S):
                S.add(u)
        return S

    # Simulated Annealing
    # Objective: Maximize |S|
    # Energy E = -|S|
    
    current_S = greedy_initial()
    best_S = set(current_S)
    best_size = len(current_S)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    max_steps = 200000
    start_time = time.time()
    
    # Lists for fast random selection
    # We maintain a list of elements in S and a list of elements not in S
    # Updating these lists on every add/remove/swap is O(N), which is fine for N=205.
    
    in_S = list(current_S)
    not_in_S = [i for i in range(m) if i not in current_S]
    
    for step in range(max_steps):
        if time.time() - start_time > 550: # Safety timeout
            break
            
        if T < T_min:
            break
            
        # Choose a move type
        # 1. Add (if possible)
        # 2. Remove
        # 3. Swap
        
        # Bias towards Add if possible, but allow exploration
        # Let's pick a random node to try to add
        if not_in_S:
            u = random.choice(not_in_S)
            
            # Check if u can be added
            if is_compatible(u, current_S):
                # Accept Add
                current_S.add(u)
                in_S.append(u)
                not_in_S.remove(u)
                
                if len(current_S) > best_size:
                    best_S = set(current_S)
                    best_size = len(current_S)
                continue
            
            # If cannot add directly, try a Swap
            # Find a neighbor v in S that conflicts with u
            # Or just pick a random v in S and check if u is compatible with S\{v}
            if in_S:
                v = random.choice(in_S)
                if is_compatible_except(u, current_S, v):
                    # Accept Swap
                    current_S.remove(v)
                    current_S.add(u)
                    in_S.remove(v)
                    in_S.append(u)
                    not_in_S.remove(u)
                    not_in_S.append(v)
                    
                    # Size unchanged, but structure changed
                    # Update best if needed? No, size is same.
                    # But we might have escaped a local minimum.
                    continue
                else:
                    # Swap failed, try a Remove move? Or just reject?
                    # Let's try a Remove move to allow future Adds
                    # Pick random w in S to remove
                    # Actually, let's just do a standard SA step:
                    # If Add fails and Swap fails, we might want to Remove something
                    # to reduce density, hoping to add more later?
                    # Or just reject this move.
                    # Let's try a Remove move with probability
                    if random.random() < 0.5:
                         # Do a Remove
                         w = random.choice(in_S)
                         # Delta E = +1
                         if math.exp(-1.0 / T) > random.random():
                             current_S.remove(w)
                             in_S.remove(w)
                             not_in_S.append(w)
                         # If rejected, do nothing
                    # If not doing remove, do nothing (reject move)
            else:
                # S is empty, add u (should have been caught by is_compatible)
                pass
        else:
            # Not_in_S is empty, we have all nodes? Impossible.
            break

        T *= cooling

    # Final local improvement: Greedy add
    for u in range(m):
        if u not in best_S and is_compatible(u, best_S):
            best_S.add(u)
            
    return np.array(sorted(list(best_S)), dtype=np.int64)

# EVOLVE-BLOCK-END
