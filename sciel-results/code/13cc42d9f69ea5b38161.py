import time
import random
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    n = len(A)
    sums = set()
    prods = set()
    # Optimization: Use lists and set conversion, or direct addition.
    # Direct addition is usually fine for n=20.
    for i in range(n):
        ai = A[i]
        for j in range(n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    rng = random.Random(42)
    
    # Initial best: Try a few strong structural candidates first
    # 1. Geometric progression (minimizes product set size significantly)
    # 2. Arithmetic progression (minimizes sum set size significantly)
    # 3. Hybrid: Geometric with small perturbations
    
    candidates = []
    
    # Geometric: 1, 2, 4, ..., 2^19 is too big (2^19 = 524288 > 10000)
    # Max geometric with ratio 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 (13 terms)
    # Need 20 terms. So pure geometric ratio 2 won't work within 10000.
    # Ratio 1.5? 1, 1.5, 2.25... integers only.
    
    # Let's try a "dense" geometric-like set for small numbers and fill up.
    # Or simply start with a random set and let SA do the work.
    
    # A good initial guess: A set that is a mix of small numbers (to keep products small) 
    # and structured to keep sums small.
    # Actually, the product set size is the dominant constraint for small numbers.
    # If A = {1, 2, 4, ..., 2^k}, |A*A| is small.
    # But we need 20 distinct integers.
    
    # Strategy: Start with a set that is a geometric progression as much as possible,
    # then use SA to adjust.
    
    # Initial Candidate 1: 1 to 20 (Arithmetic)
    A1 = list(range(1, 21))
    
    # Initial Candidate 2: Powers of 2, extended
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
    # Next: 8192 (13 terms).
    # Need 7 more. Add 5000, 6000, 7000, 8000, 9000, 9500, 9900?
    A2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # Fill remaining 6 with random large numbers to stay distinct and <= 10000
    remaining = rng.sample([x for x in range(5000, 10000) if x not in A2], 6)
    A2.extend(remaining)
    
    # Initial Candidate 3: Random
    A3 = rng.sample(range(1, 10001), 20)
    
    best_A = A1
    best_score = _score(A1)
    
    for A in [A2, A3]:
        s = _score(A)
        if s < best_score:
            best_score = s
            best_A = sorted(A)
            
    # Simulated Annealing
    current_A = best_A[:]
    current_score = best_score
    
    # Temperature parameters
    T_start = 1000
    T_end = 1
    cooling_rate = 0.999
    
    T = T_start
    step = 0
    max_steps = 50000 # Limit steps to ensure time safety
    
    while time.time() < deadline and T > T_end and step < max_steps:
        step += 1
        
        # Pick a random index to modify
        i = rng.randint(0, 19)
        old_val = current_A[i]
        
        # Generate a new value
        # Bias towards keeping the structure: 
        # If the value is small, change it slightly. If large, change it more.
        # Or just random within 1..10000.
        # To avoid duplicates, we can retry a few times or just check validity.
        
        new_val = rng.randint(1, 10000)
        
        # Ensure distinctness
        if new_val in current_A and new_val != old_val:
            continue
            
        # Create new state
        new_A = current_A[:]
        new_A[i] = new_val
        new_A.sort() # Keep sorted for consistency, though not strictly necessary for score
        
        # Check validity (distinct, positive, <= 10000)
        # Since we picked from 1..10000 and checked distinctness against current_A, 
        # and current_A is valid, new_A should be valid unless new_val == old_val (which is fine)
        # Wait, if new_val == old_val, score is same. 
        # If new_val is in current_A, we skipped. 
        # So new_A is always valid.
        
        new_score = _score(new_A)
        
        # Metropolis criterion
        delta = new_score - current_score
        if delta < 0:
            # Accept
            current_A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            # Accept with probability exp(-delta / T)
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                current_A = new_A
                current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
    # Final local search refinement on the best found
    # Simple hill climbing on the best_A
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(20):
            for delta in [-2, -1, 1, 2]:
                new_val = best_A[i] + delta
                if 1 <= new_val <= 10000 and new_val not in best_A:
                    new_A = best_A[:]
                    new_A[i] = new_val
                    new_A.sort()
                    s = _score(new_A)
                    if s < best_score:
                        best_score = s
                        best_A = new_A
                        improved = True
                        break
            if improved:
                break
                
    return sorted(best_A)
