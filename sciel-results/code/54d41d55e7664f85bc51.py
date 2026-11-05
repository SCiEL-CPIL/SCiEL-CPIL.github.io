import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_count = 0
    best_result = None
    
    moduli = list(range(2, 129))
    n_moduli = len(moduli)
    
    gcds = [[0]*129 for _ in range(129)]
    for i in range(2, 129):
        for j in range(2, 129):
            gcds[i][j] = math.gcd(i, j)
            
    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    moduli_sorted = sorted(moduli, key=lambda m: (-count_prime_factors(m), m))
    
    restarts = 0
    while time.time() < deadline:
        current_order = moduli_sorted[:]
        random.shuffle(current_order)
        
        chosen = []
        
        for idx, m in enumerate(current_order):
            # Prune: if we can't beat best_count, break
            remaining = n_moduli - idx
            if len(chosen) + remaining <= best_count:
                break
                
            valid_residues = []
            for r in range(m):
                is_valid = True
                for r_i, m_i in chosen:
                    g = gcds[m][m_i]
                    if r % g == r_i % g:
                        is_valid = False
                        break
                if is_valid:
                    valid_residues.append(r)
            
            if valid_residues:
                r = random.choice(valid_residues)
                chosen.append((r, m))
        
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_result = [[r, m] for r, m in chosen]
            
        restarts += 1
        
    if best_result is None:
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_result
