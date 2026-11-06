import random
import time
import math

def construct():
    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)  # 168
    MAX_X = 30000  # Increase search horizon
    
    def get_coverage(residues):
        cov = [0] * (MAX_X + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            x = start
            while x <= MAX_X:
                cov[x] += 1
                x += p
        return cov

    def find_L(cov):
        L = 0
        for x in range(1, MAX_X + 1):
            if cov[x] < 2:
                break
            L = x
        return L

    def solve_sa(init_residues, time_limit):
        residues = init_residues[:]
        cov = get_coverage(residues)
        L = find_L(cov)
        
        best_L = L
        best_res = residues[:]
        
        start_time = time.time()
        T = 100.0 # Initial temperature
        T_min = 0.01
        cooling = 0.999
        
        # Precompute indices for faster access
        while time.time() - start_time < time_limit:
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Generate a move
            # 1. Single change
            if random.random() < 0.5:
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                # Apply move
                if old_r == 0: start = p
                else: start = old_r
                y = start
                while y <= MAX_X:
                    cov[y] -= 1
                    y += p
                
                if new_r == 0: start = p
                else: start = new_r
                y = start
                while y <= MAX_X:
                    cov[y] += 1
                    y += p
                    
                residues[i] = new_r
                new_L = find_L(cov)
                
                # Check acceptance
                delta = new_L - L
                if delta > 0 or random.random() < math.exp(delta / T):
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_res = residues[:]
                else:
                    # Revert
                    if new_r == 0: start = p
                    else: start = new_r
                    y = start
                    while y <= MAX_X:
                        cov[y] -= 1
                        y += p
                    
                    if old_r == 0: start = p
                    else: start = old_r
                    y = start
                    while y <= MAX_X:
                        cov[y] += 1
                        y += p
                        
                    residues[i] = old_r
            else:
                # 2. Two-swap
                i = random.randint(0, n_primes - 1)
                j = random.randint(0, n_primes - 1)
                if i == j: continue
                
                p_i = primes[i]
                p_j = primes[j]
                old_r_i = residues[i]
                old_r_j = residues[j]
                
                new_r_i = random.randint(0, p_i - 1)
                new_r_j = random.randint(0, p_j - 1)
                
                if new_r_i == old_r_i and new_r_j == old_r_j:
                    continue
                    
                # Apply changes for i
                if old_r_i == 0: start = p_i
                else: start = old_r_i
                y = start
                while y <= MAX_X:
                    cov[y] -= 1
                    y += p_i
                
                if new_r_i == 0: start = p_i
                else: start = new_r_i
                y = start
                while y <= MAX_X:
                    cov[y] += 1
                    y += p_i
                    
                # Apply changes for j
                if old_r_j == 0: start = p_j
                else: start = old_r_j
                y = start
                while y <= MAX_X:
                    cov[y] -= 1
                    y += p_j
                
                if new_r_j == 0: start = p_j
                else: start = new_r_j
                y = start
                while y <= MAX_X:
                    cov[y] += 1
                    y += p_j
                    
                residues[i] = new_r_i
                residues[j] = new_r_j
                
                new_L = find_L(cov)
                
                delta = new_L - L
                if delta > 0 or random.random() < math.exp(delta / T):
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_res = residues[:]
                else:
                    # Revert i
                    if new_r_i == 0: start = p_i
                    else: start = new_r_i
                    y = start
                    while y <= MAX_X:
                        cov[y] -= 1
                        y += p_i
                    
                    if old_r_i == 0: start = p_i
                    else: start = old_r_i
                    y = start
                    while y <= MAX_X:
                        cov[y] += 1
                        y += p_i
                        
                    # Revert j
                    if new_r_j == 0: start = p_j
                    else: start = new_r_j
                    y = start
                    while y <= MAX_X:
                        cov[y] -= 1
                        y += p_j
                    
                    if old_r_j == 0: start = p_j
                    else: start = old_r_j
                    y = start
                    while y <= MAX_X:
                        cov[y] += 1
                        y += p_j
                        
                    residues[i] = old_r_i
                    residues[j] = old_r_j
                    
        return best_res, best_L

    best_L = 0
    best_res = None
    start_time = time.time()
    time_budget = 115
    
    # Use a few different initializations
    inits = [
        [0] * n_primes,
        [random.randint(0, p - 1) for p in primes]
    ]
    
    for init in inits:
        if time.time() - start_time > time_budget:
            break
        res, L = solve_sa(init, time_budget - (time.time() - start_time))
        if L > best_L:
            best_L = L
            best_res = res[:]
            
    # If we have time left, do another restart with best_res as start? 
    # SA usually converges, so maybe just run one long SA is better.
    # But multiple restarts help escape global minima.
    
    if best_res is None:
        return [0] * 168
        
    return best_res
