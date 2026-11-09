# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Symmetrize: if d is forbidden, -d is also effectively forbidden
    forbidden_sym = set()
    for d in forbidden:
        forbidden_sym.add(d)
        forbidden_sym.add((-d) % m)
    forbidden_sym.discard(0)
    
    # Precompute conflict bitsets:
    # conflict_mask[x] is a bitmask where bit y is set if x and y conflict
    conflict_mask = [0] * m
    for x in range(m):
        mask = 0
        for y in range(m):
            if x == y:
                continue
            d = (x - y) % m
            if d in forbidden_sym:
                mask |= (1 << y)
        conflict_mask[x] = mask
    
    def is_valid_addition(S_bitset, x):
        """Check if x can be added to set represented by S_bitset"""
        return (conflict_mask[x] & S_bitset) == 0
    
    def greedy_construct(order):
        """Construct a set greedily based on the given order of candidates"""
        S_bitset = 0
        S_list = []
        for x in order:
            if is_valid_addition(S_bitset, x):
                S_bitset |= (1 << x)
                S_list.append(x)
        return S_list, S_bitset
    
    def local_search(S_list, S_bitset):
        """
        Local search with Add and Remove+Add moves using bitsets.
        """
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Try to add any possible element
            for x in range(m):
                if not (S_bitset >> x) & 1:
                    if is_valid_addition(S_bitset, x):
                        S_bitset |= (1 << x)
                        S_list.append(x)
                        improved = True
            
            # Phase 2: If no additions, try removing one element and refilling
            if not improved:
                # Try removing each element
                for u in list(S_list):
                    # Remove u
                    S_bitset_copy = S_bitset & ~(1 << u)
                    S_list_copy = [s for s in S_list if s != u]
                    
                    # Refill greedily: try to add any x not in S_set_copy
                    for x in range(m):
                        if not (S_bitset_copy >> x) & 1:
                            if is_valid_addition(S_bitset_copy, x):
                                S_bitset_copy |= (1 << x)
                                S_list_copy.append(x)
                    
                    # If the new set is strictly larger, accept it
                    if len(S_list_copy) > len(S_list):
                        S_list = S_list_copy
                        S_bitset = S_bitset_copy
                        improved = True
                        
        return S_list, S_bitset

    best_set = []
    best_bitset = 0
    
    # Deadline guard: 630s limit, keep safety margin
    deadline = time.time() + 610.0
    
    # Randomized Greedy + Local Search with restarts
    for i in range(50000):  # Upper bound on restarts, but limited by time
        if time.time() > deadline:
            break
        
        # Create a random order for greedy construction
        order = list(range(m))
        random.shuffle(order)
        
        # 1. Greedy construction
        S, S_bitset = greedy_construct(order)
        
        # 2. Local search to improve
        S, S_bitset = local_search(S, S_bitset)
        
        # 3. Update best if better
        if len(S) > len(best_set):
            best_set = S
            best_bitset = S_bitset
            
    # Fallback: if no set found (should not happen), return [0]
    if not best_set:
        best_set = [0]
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
