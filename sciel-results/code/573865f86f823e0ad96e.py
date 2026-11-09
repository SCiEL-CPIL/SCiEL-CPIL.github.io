import random
import time
import math

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    best_result = []
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    # Also consider the full set as a fallback, but prime-restricted sets are better candidates
    # We will try each prime's multiples
    for p in primes:
        if time.time() - start_time > timeout:
            break
            
        candidates = [m for m in range(2, 129) if m % p == 0]
        if len(candidates) < len(best_result):
            continue
            
        # Run randomized greedy on this subset
        # We do a few restarts for each prime
        max_restarts = 50
        for _ in range(max_restarts):
            if time.time() - start_time > timeout:
                break
                
            random.shuffle(candidates)
            current_list = []
            
            for m in candidates:
                found = False
                residues = list(range(m))
                random.shuffle(residues)
                
                for r in residues:
                    compatible = True
                    for r_i, m_i in current_list:
                        g = math.gcd(m, m_i)
                        if (r - r_i) % g == 0:
                            compatible = False
                            break
                    if compatible:
                        current_list.append([r, m])
                        found = True
                        break
                
            if len(current_list) > len(best_result):
                best_result = current_list
                
    # Also try the full set with the original logic just in case, but it's likely worse
    # Actually, the full set logic might find a mix if the prime restriction is too strict?
    # But coprime moduli are impossible. So any valid set must be a subset of non-coprime numbers.
    # The largest non-coprime sets are multiples of small primes.
    
    return best_result
