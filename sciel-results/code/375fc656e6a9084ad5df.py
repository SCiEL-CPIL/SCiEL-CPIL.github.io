import math
import time
import random

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # All possible moduli from 2 to 128
    all_moduli = list(range(2, 129))
    
    # Greedy approach with multiple strategies
    def greedy_build(moduli_list, order_strategy='size_desc'):
        """
        Greedily assign residues to moduli.
        Returns list of [residue, modulus] pairs.
        """
        if not moduli_list:
            return []
        
        # Order the moduli
        if order_strategy == 'size_desc':
            ordered = sorted(moduli_list, reverse=True)
        elif order_strategy == 'size_asc':
            ordered = sorted(moduli_list)
        elif order_strategy == 'random':
            ordered = moduli_list[:]
            random.shuffle(ordered)
        else:
            ordered = moduli_list[:]
        
        assigned = []  # list of (r, m)
        residues = {}
        
        for m in ordered:
            found = False
            # Try residues 0 to m-1
            for r in range(m):
                valid = True
                for (r2, m2) in assigned:
                    if not _disjoint(r, m, r2, m2):
                        valid = False
                        break
                if valid:
                    assigned.append((r, m))
                    residues[m] = r
                    found = True
                    break
            
            if not found:
                # This modulus cannot be added, skip it
                continue
        
        return [[r, m] for (r, m) in assigned]
    
    best_result = []
    
    # Try multiple strategies
    strategies = [
        ('size_desc', None),
        ('size_asc', None),
        ('random', 1),
        ('random', 2),
        ('random', 3),
    ]
    
    for strategy, seed in strategies:
        if time.time() > deadline:
            break
        
        if seed is not None:
            random.seed(seed)
        
        result = greedy_build(all_moduli, strategy)
        
        if len(result) > len(best_result):
            best_result = result
    
    # If we still have time, try more random restarts
    if time.time() < deadline - 5:
        for i in range(20):
            if time.time() > deadline:
                break
            random.seed(100 + i)
            result = greedy_build(all_moduli, 'random')
            if len(result) > len(best_result):
                best_result = result
    
    # Fallback if nothing found
    if not best_result:
        # Return a simple valid set
        best_result = [[1, 2], [0, 3], [2, 4]]
    
    return best_result
