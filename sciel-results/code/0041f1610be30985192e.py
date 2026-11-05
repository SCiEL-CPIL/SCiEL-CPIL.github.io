import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin for 90s limit
    
    best_count = 0
    best_solution = []
    
    # Moduli from 2 to 128
    moduli = list(range(2, 129))
    
    # Precompute gcd matrix?
    # gcd(m, n) is fast.
    
    restarts = 0
    while time.time() < deadline:
        random.shuffle(moduli)
        current_solution = []
        
        for m in moduli:
            # Try to find a residue for m
            # We want to maximize the chance of finding one.
            # The number of valid residues is m - sum(forbidden).
            # Just try random order.
            
            residues = list(range(m))
            random.shuffle(residues)
            
            added = False
            for a in residues:
                valid = True
                for b, n in current_solution:
                    g = math.gcd(m, n)
                    if a % g == b % g:
                        valid = False
                        break
                if valid:
                    current_solution.append([a, m])
                    added = True
                    break
        
        # Validate
        if is_valid(current_solution):
            if len(current_solution) > best_count:
                best_count = len(current_solution)
                best_solution = [pair[:] for pair in current_solution]
                
        restarts += 1
        
    # Fallback: If no solution found (should not happen), return a single valid class
    if not best_solution:
        best_solution = [[0, 2]]
        
    return best_solution

def is_valid(solution):
    n = len(solution)
    for i in range(n):
        for j in range(i + 1, n):
            a1, m1 = solution[i]
            a2, m2 = solution[j]
            g = math.gcd(m1, m2)
            if a1 % g == a2 % g:
                return False
    return True
