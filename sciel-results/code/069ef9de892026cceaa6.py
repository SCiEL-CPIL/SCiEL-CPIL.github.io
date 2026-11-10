import random
import time
import math

def construct():
    start_time = time.time()
    max_time = 55.0  # Safety margin
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    
    best_solution = []
    best_size = 0
    
    # Initial greedy with fixed order to get a baseline
    # We try to add moduli in increasing order
    current_set = []
    for m in moduli:
        valid_residues = []
        for r in range(m):
            compatible = True
            for r2, m2 in current_set:
                g = math.gcd(m, m2)
                if (r - r2) % g == 0:
                    compatible = False
                    break
            if compatible:
                valid_residues.append(r)
        
        if valid_residues:
            # Pick the first valid residue
            current_set.append([valid_residues[0], m])
            
    best_solution = current_set
    best_size = len(current_set)
    
    # Randomized local search / greedy with restarts
    restarts = 0
    while time.time() - start_time < max_time:
        restarts += 1
        random.shuffle(moduli)
        current_set = []
        
        for m in moduli:
            valid_residues = []
            for r in range(m):
                compatible = True
                for r2, m2 in current_set:
                    g = math.gcd(m, m2)
                    if (r - r2) % g == 0:
                        compatible = False
                        break
                if compatible:
                    valid_residues.append(r)
            
            if valid_residues:
                # Pick a random valid residue to diversify
                current_set.append([random.choice(valid_residues), m])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_solution = [pair[:] for pair in current_set]
            
        # Optional: Local search improvement on the best solution found
        # Try to swap or add/remove to improve
        # Given time constraints, simple restarts might be sufficient
        
    return best_solution
