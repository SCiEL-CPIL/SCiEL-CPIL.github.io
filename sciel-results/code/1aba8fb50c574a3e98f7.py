"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    base = 51975
    
    # Generate all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size descending
    divisors_desc = sorted(divisors, reverse=True)
    
    covered_count = np.zeros(base, dtype=np.int32)
    classes = []
    
    # Greedy Phase: Large moduli first
    for m in divisors_desc:
        best_r = 0
        best_gain = -1
        
        for r in range(m):
            indices = np.arange(r, base, m)
            gain = np.sum(covered_count[indices] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        classes.append([m, best_r])
        indices = np.arange(best_r, base, m)
        covered_count[indices] += 1
        
    # Local Search Phase
    max_iterations = 20
    for _ in range(max_iterations):
        improved = False
        
        for i, (m, r) in enumerate(classes):
            current_indices = np.arange(r, base, m)
            L = np.sum(covered_count[current_indices] == 1)
            
            best_r_new = -1
            best_G = -1
            
            for r_new in range(m):
                if r_new == r:
                    continue
                new_indices = np.arange(r_new, base, m)
                G = np.sum(covered_count[new_indices] == 0)
                if G > best_G:
                    best_G = G
                    best_r_new = r_new
            
            if best_G > L:
                covered_count[current_indices] -= 1
                new_indices = np.arange(best_r_new, base, m)
                covered_count[new_indices] += 1
                classes[i] = [m, best_r_new]
                improved = True
        
        if not improved:
            break
            
    return classes
# EVOLVE-BLOCK-END
