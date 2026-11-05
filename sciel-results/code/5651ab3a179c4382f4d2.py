import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Use primes >= 13 as candidates (sum of reciprocals ~0.92, leaves room)
    candidate_primes = [p for p in primes if p >= 13]
    
    L_max = 2000
    covered = np.zeros(L_max + 1, dtype=bool)
    indices = np.arange(1, L_max + 1)
    
    selected_classes = []
    used_primes = set()
    current_sum = Fraction(0, 1)
    
    # Phase 1: Sequential greedy - for each x from 1, if uncovered, pick best prime
    for x in range(1, L_max + 1):
        if time.time() > deadline - 30:
            break
            
        if covered[x]:
            continue
            
        best_p = None
        best_r = None
        best_gain = -1
        best_cost = None
        
        for p in candidate_primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if current_sum + cost > Fraction(1, 1):
                continue
            
            r = x % p
            
            # Calculate gain: how many uncovered integers in [1, L_max] are covered by [p, r]?
            mask = (indices % p) == r
            unc_mask = ~covered[1:L_max+1]
            gain = int(np.sum(unc_mask & mask))
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
                best_r = r
                best_cost = cost
        
        if best_p is None:
            break
            
        used_primes.add(best_p)
        current_sum += best_cost
        selected_classes.append([best_p, best_r])
        
        # Update covered
        mask = (indices % best_p) == best_r
        covered[1:L_max+1][mask] = True
    
    # Calculate current L
    def compute_L(covered_arr, max_check):
        L = 0
        for i in range(1, min(max_check, L_max) + 1):
            if covered_arr[i]:
                L = i
            else:
                break
        return L
    
    current_L = compute_L(covered, L_max)
    
    # Phase 2: Local search - try to improve by swapping primes or adjusting residues
    unused_primes = [p for p in primes if p not in used_primes]
    
    improved = True
    max_iterations = 20
    iteration = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 10:
        improved = False
        iteration += 1
        
        # Try swapping a used prime with an unused prime
        for i, (p_used, r_used) in enumerate(selected_classes[:]):
            if time.time() > deadline - 10:
                break
            
            for p_new in unused_primes:
                if time.time() > deadline - 10:
                    break
                
                cost_old = Fraction(1, p_used)
                cost_new = Fraction(1, p_new)
                
                # Check if swap is budget-feasible
                if current_sum - cost_old + cost_new > Fraction(1, 1):
                    continue
                
                # Try all residues for p_new
                best_L_new = 0
                best_r_new = 0
                
                for r_new in range(min(p_new, L_max + 1)):
                    # Compute coverage with old prime removed and new prime added
                    temp_covered = np.zeros(L_max + 1, dtype=bool)
                    
                    for j, (p_j, r_j) in enumerate(selected_classes):
                        if j == i:
                            continue
                        start = r_j if r_j != 0 else p_j
                        x = start
                        while x <= L_max:
                            temp_covered[x] = True
                            x += p_j
                    
                    # Add new prime
                    start = r_new if r_new != 0 else p_new
                    x = start
                    while x <= L_max:
                        temp_covered[x] = True
                        x += p_new
                    
                    L_temp = compute_L(temp_covered, L_max)
                    
                    if L_temp > best_L_new:
                        best_L_new = L_temp
                        best_r_new = r_new
                
                if best_L_new > current_L:
                    # Perform the swap
                    selected_classes[i] = [p_new, best_r_new]
                    used_primes.remove(p_used)
                    used_primes.add(p_new)
                    unused_primes.remove(p_new)
                    unused_primes.append(p_used)
                    current_sum = current_sum - cost_old + cost_new
                    
                    # Update covered
                    covered = np.zeros(L_max + 1, dtype=bool)
                    for p_j, r_j in selected_classes:
                        start = r_j if r_j != 0 else p_j
                        x = start
                        while x <= L_max:
                            covered[x] = True
                            x += p_j
                    
                    current_L = best_L_new
                    improved = True
                    break
            
            if improved:
                break
        
        # Try adjusting residues of existing primes
        for i, (p_i, r_i) in enumerate(selected_classes[:]):
            if time.time() > deadline - 10:
                break
            
            best_L_adj = 0
            best_r_adj = r_i
            
            for r_new in range(min(p_i, L_max + 1)):
                if r_new == r_i:
                    continue
                
                temp_covered = np.zeros(L_max + 1, dtype=bool)
                
                for j, (p_j, r_j) in enumerate(selected_classes):
                    if j == i:
                        r_use = r_new
                    else:
                        r_use = r_j
                    start = r_use if r_use != 0 else p_j
                    x = start
                    while x <= L_max:
                        temp_covered[x] = True
                        x += p_j
                
                L_temp = compute_L(temp_covered, L_max)
                
                if L_temp > best_L_adj:
                    best_L_adj = L_temp
                    best_r_adj = r_new
            
            if best_L_adj > current_L:
                selected_classes[i][1] = best_r_adj
                current_L = best_L_adj
                improved = True
                
                # Update covered
                covered = np.zeros(L_max + 1, dtype=bool)
                for p_j, r_j in selected_classes:
                    start = r_j if r_j != 0 else p_j
                    x = start
                    while x <= L_max:
                        covered[x] = True
                        x += p_j
    
    # Calculate final L
    L = compute_L(covered, L_max)
    
    return selected_classes
