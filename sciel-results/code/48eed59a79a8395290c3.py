import time
import random
import math
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin before 150s kill

    # Generate primes up to 1000
    def _primes_upto(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    all_primes = _primes_upto(1000)
    # Start with primes >= 13. Sum of 1/p for p>=13 is roughly 0.98, well under 1.
    # Including 11, 7, 5, 3, 2 might exceed 1 or be too expensive.
    # Sum 1/p for p in {2,3,5,7,11,13,...}
    # 1/2+1/3+1/5+1/7+1/11 = 0.5+0.333+0.2+0.142+0.0909 = 1.26 > 1.
    # So we must exclude small primes.
    # 1/3+1/5+1/7+1/11+1/13... = 0.333+0.2+0.142+0.0909+0.0769...
    # Sum of 1/p for p>=3 is approx 1.5 - 0.5 = 1.0? No, sum 1/p diverges slowly.
    # Sum 1/p for p in [3, 1000] is approx 1.55 - 0.5 = 1.05?
    # Let's check: sum 1/p for p<=1000 is approx log(log(1000)) + M = log(6.9) + 0.26 = 1.93 + 0.26 = 2.19.
    # Sum 1/p for p>=13:
    # 1/13+1/17+1/19+...
    # Let's compute it to be safe.
    
    selected_primes = []
    cost = Fraction(0, 1)
    for p in all_primes:
        if p < 13:
            continue
        c = Fraction(1, p)
        if cost + c <= Fraction(1, 1):
            selected_primes.append(p)
            cost += c
        else:
            break
            
    N = len(selected_primes)
    
    # We want to maximize L.
    # MAX_L for search space. We know L=999 is the target.
    # Using MAX_L = 20000 allows us to find L > 999 if possible, but makes get_L slower.
    # However, get_L using np.argmax on a boolean array is fast.
    # Let's use MAX_L = 50000 to be safe, but check performance.
    # Actually, if we cover 1..999, L is at least 999.
    # If we can cover more, great.
    # The counts array size is MAX_L+1.
    # Let's stick to 10000 or 20000 for speed, as 999 is the main goal.
    # But wait, if we find a gap at 1000, L=999.
    # If we find a gap at 1500, L=1499.
    # Let's use MAX_L = 20000.
    
    MAX_L = 20000
    
    # Initialize residues randomly
    residues = [random.randint(0, p - 1) for p in selected_primes]
    
    # Initialize counts using NumPy
    # dtype int16 is enough (max N ~ 160)
    counts = np.zeros(MAX_L + 1, dtype=np.int16)
    
    for i in range(N):
        p = selected_primes[i]
        r = residues[i]
        counts[r::p] += 1

    def get_L():
        # Find first index i >= 1 where counts[i] == 0
        # If all covered up to MAX_L, return MAX_L
        # counts[1:] == 0 is boolean array
        # np.argmax returns first True index. If no True, returns len.
        idx = np.argmax(counts[1:] == 0)
        # If idx == len(counts[1:]), it means no zero found, so all 1..MAX_L covered.
        # In that case, L = MAX_L.
        # Otherwise, L = idx.
        if idx == len(counts[1:]):
            return MAX_L
        return idx

    L = get_L()
    best_L = L
    best_residues = residues[:]
    
    # Simulated Annealing with restarts
    # Use a restart strategy: if stuck for a while, restart with new random init or reheat.
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Pick random prime to change
        i = random.randint(0, N - 1)
        p = selected_primes[i]
        old_r = residues[i]
        
        # Pick random new residue
        new_r = random.randint(0, p - 1)
        
        if old_r == new_r:
            continue
            
        # Apply change
        counts[old_r::p] -= 1
        counts[new_r::p] += 1
        
        new_L = get_L()
        
        # Acceptance criterion
        if new_L > L:
            L = new_L
            residues[i] = new_r
            if L > best_L:
                best_L = L
                best_residues = residues[:]
                no_improve_count = 0
        else:
            delta = L - new_L
            if delta == 0:
                if random.random() < 0.5:
                    L = new_L
                    residues[i] = new_r
                else:
                    counts[old_r::p] += 1
                    counts[new_r::p] -= 1
            else:
                # Worsening move
                # Use a more aggressive acceptance for large T
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    L = new_L
                    residues[i] = new_r
                else:
                    counts[old_r::p] += 1
                    counts[new_r::p] -= 1
            
            if L <= best_L:
                no_improve_count += 1
            else:
                no_improve_count = 0
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Reheat
            T = 5.0
            no_improve_count = 0
            # Optional: perturb current solution significantly
            # Or restart from scratch? Restarting from scratch might be better.
            # Let's try a "soft restart": keep current residues but reheat.
            # If that fails, we might need a hard restart.
            # For now, just reheat.
            
    # Final check: ensure we return the best found
    # Re-verify best_residues just in case
    # Actually, best_residues is a copy.
    return [[selected_primes[i], best_residues[i]] for i in range(N)]
