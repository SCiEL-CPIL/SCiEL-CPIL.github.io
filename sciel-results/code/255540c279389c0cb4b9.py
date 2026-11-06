import numpy as np
import random
from fractions import Fraction

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Select primes to satisfy budget 1
    # Sort by 1/p descending (small primes first) to pack budget?
    # Or just use primes >= 13 as in seed, which is a known good set.
    # Let's try to optimize the prime set selection too.
    # But changing prime set is harder. Let's stick to a good fixed set first.
    # The seed uses p >= 13. Sum is ~0.94.
    # We can maybe add small primes and remove large ones?
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 = 1.26 > 1.
    # 1/2 + 1/3 + 1/5 + 1/7 = 1.17 > 1.
    # 1/2 + 1/3 + 1/5 = 1.03 > 1.
    # 1/2 + 1/3 = 0.83. We can add 1/7 (0.14) -> 0.97.
    # So {2, 3, 7} is a valid set with sum 0.97.
    # Does {2, 3, 7} cover more than {13, 17, ...}?
    # Small primes cover denser sets.
    # However, we need to cover 1..L.
    # If we use 2, we cover all even or all odd.
    # If we use 3, we cover 1/3 of numbers.
    # The "Erdos-Ruzsa" conjecture usually involves large primes?
    # No, it's about covering systems.
    # Let's try a mixed set.
    
    # Let's stick to the seed's prime set for now to ensure we don't regress, 
    # but optimize the residues much better.
    P = [p for p in primes if p >= 13]
    
    # Verify budget
    sum_recip = sum(1.0/p for p in P)
    if sum_recip > 1.0:
        # Remove largest primes
        P.sort(reverse=True)
        while sum(1.0/p for p in P) > 1.0 and len(P) > 1:
            P.pop()
            
    # Residues
    residues = {p: random.randint(0, p-1) for p in P}
    
    L_max = 5000 # Check prefix up to 5000 for optimization
    # Use numpy for coverage
    # covered[n] is True if n is covered
    
    def get_coverage_count(res_map, P_list, L_limit):
        covered = np.zeros(L_limit + 1, dtype=bool)
        for p in P_list:
            r = res_map[p]
            start = r if r > 0 else p
            if start <= L_limit:
                covered[start:L_limit+1:p] = True
        return np.sum(covered[1:L_limit+1]) # Count covered in 1..L_limit

    def get_prefix_L(res_map, P_list, L_limit):
        covered = np.zeros(L_limit + 1, dtype=bool)
        for p in P_list:
            r = res_map[p]
            start = r if r > 0 else p
            if start <= L_limit:
                covered[start:L_limit+1:p] = True
        # Find first False
        for n in range(1, L_limit + 1):
            if not covered[n]:
                return n - 1
        return L_limit

    # Simulated Annealing
    current_res = residues.copy()
    current_score = get_coverage_count(current_res, P, L_max)
    
    T = 100.0
    T_min = 0.01
    alpha = 0.999
    
    for _ in range(5000): # Iterations
        if T < T_min: break
        
        # Pick a random prime and change its residue
        p = random.choice(P)
        old_r = current_res[p]
        new_r = random.randint(0, p-1)
        if new_r == old_r: continue
        
        current_res[p] = new_r
        new_score = get_coverage_count(current_res, P, L_max)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
        else:
            current_res[p] = old_r
            
        T *= alpha
        
    # Final local search to refine
    improved = True
    while improved:
        improved = False
        # Find smallest uncovered in 1..L_max
        covered = np.zeros(L_max + 1, dtype=bool)
        for p in P:
            r = current_res[p]
            start = r if r > 0 else p
            if start <= L_max:
                covered[start:L_max+1:p] = True
        
        # Find first hole
        hole = -1
        for n in range(1, L_max + 1):
            if not covered[n]:
                hole = n
                break
        
        if hole == -1:
            break # Covered all L_max
            
        # Try to fix hole
        for p in P:
            new_r = hole % p
            if current_res[p] == new_r: continue
            
            old_r = current_res[p]
            current_res[p] = new_r
            
            # Check if 1..hole-1 still covered
            covered_temp = np.zeros(hole, dtype=bool) # 0..hole-1
            for pp in P:
                rr = current_res[pp]
                start = rr if rr > 0 else pp
                if start < hole:
                    covered_temp[start:hole:pp] = True
            
            if np.all(covered_temp[1:hole]): # 1..hole-1 covered
                improved = True
                break
            else:
                current_res[p] = old_r
                
    # Calculate final L
    final_L = get_prefix_L(current_res, P, 100000) # Check up to 100k? 
    # The task says capped at 10^5.
    # But calculating up to 100k every time is slow.
    # We can check up to 5000 during SA, and final check up to 100k.
    
    return [[p, current_res[p]] for p in P]
