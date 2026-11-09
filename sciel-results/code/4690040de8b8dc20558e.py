import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0
    
    N = 51975
    
    # Get divisors
    divisors = []
    for d in range(2, N + 1):
        if N % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # State
    counts = np.zeros(N, dtype=np.int32)
    uncovered = N
    residues = {m: 0 for m in divisors} # m -> r
    
    def update_add(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        dec = np.sum(counts[idx] == 0)
        counts[idx] += 1
        uncovered -= dec
        
    def update_remove(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        inc = np.sum(counts[idx] == 1)
        counts[idx] -= 1
        uncovered += inc
        
    # Initialize counts with random residues
    for m in divisors:
        r = random.randint(0, m-1)
        residues[m] = r
        update_add(m, r)
        
    best_uncovered = uncovered
    best_residues = residues.copy()
    
    # Simulated Annealing
    # T starts high, decreases over time
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    prev_uncovered = uncovered
    
    while time.time() < deadline - 5 and T > T_min:
        # Pick a random modulus
        m = random.choice(divisors)
        current_r = residues[m]
        
        # Pick a random new residue
        new_r = random.randint(0, m-1)
        if new_r == current_r:
            T *= cooling_rate
            continue
            
        # Calculate move
        update_remove(m, current_r)
        update_add(m, new_r)
        
        delta = uncovered - prev_uncovered
        
        # Acceptance probability
        if delta < 0:
            accept = True
        else:
            if T == 0:
                accept = False
            else:
                # Avoid overflow/underflow in exp
                if delta / T > 50: 
                    accept = False
                else:
                    accept = random.random() < np.exp(-delta / T)
                
        if accept:
            residues[m] = new_r
            prev_uncovered = uncovered
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_residues = residues.copy()
        else:
            # Revert
            update_remove(m, new_r)
            update_add(m, current_r)
            
        T *= cooling_rate
        
    # Reconstruct best classes
    classes = []
    for m in divisors:
        classes.append([m, best_residues[m]])
        
    return classes
