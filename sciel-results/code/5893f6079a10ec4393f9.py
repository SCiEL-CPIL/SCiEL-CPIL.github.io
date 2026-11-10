"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Precompute prime factors for all numbers 1..511
    # factors[i] is a set of prime factors of i
    factors = [set() for _ in range(512)]
    for i in range(2, 512):
        if not factors[i]:  # i is prime
            for j in range(i, 512, i):
                factors[j].add(i)
    
    # Calculate score for a set of numbers
    def get_score(selected):
        total = 0.0
        for a in selected:
            total += 1.0 / (512 - a)
        return total
    
    # Check if candidate is coprime to all in selected
    def is_coprime_to_all(candidate, selected):
        cand_factors = factors[candidate]
        for s in selected:
            if cand_factors & factors[s]:
                return False
        return True
    
    # 1. Greedy Initialization
    selected = []
    for a in range(511, 0, -1):
        if is_coprime_to_all(a, selected):
            selected.append(a)
    
    current_score = get_score(selected)
    
    # 2. Local Search (Hill Climbing)
    # We allow time for multiple passes
    while time.time() < deadline:
        improved = False
        
        # Try to add any new element
        best_add = None
        best_add_val = 0
        for a in range(1, 512):
            if a not in selected:
                if is_coprime_to_all(a, selected):
                    val = 1.0 / (512 - a)
                    if val > best_add_val:
                        best_add_val = val
                        best_add = a
        
        if best_add is not None:
            selected.append(best_add)
            current_score += best_add_val
            improved = True
            continue # Restart search as set changed
        
        # Try to swap: remove u, add v
        # To optimize, we only consider v that are coprime to selected \ {u}
        # We want to maximize (val_v - val_u)
        
        # Precompute values
        values = {a: 1.0 / (512 - a) for a in range(1, 512)}
        
        for u in list(selected):
            # Temporarily remove u
            remaining = [x for x in selected if x != u]
            
            best_v = None
            best_gain = 0
            
            for v in range(1, 512):
                if v == u or v in remaining:
                    continue
                
                # Check if v is coprime to remaining
                if is_coprime_to_all(v, remaining):
                    gain = values[v] - values[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
            
            if best_v is not None and best_gain > 1e-9:
                # Perform swap
                selected.remove(u)
                selected.append(best_v)
                current_score += best_gain
                improved = True
                break # Restart search as set changed
        
        if not improved:
            break
            
    return selected
# EVOLVE-BLOCK-END
