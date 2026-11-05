import math
import random
import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    def gcd(a, b):
        return math.gcd(a, b)

    max_mod = 128
    gcds = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for i in range(2, max_mod + 1):
        for j in range(2, max_mod + 1):
            gcds[i][j] = gcd(i, j)
            
    primes = []
    for i in range(2, max_mod + 1):
        is_prime = True
        for p in range(2, int(math.isqrt(i)) + 1):
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    composites = [m for m in range(2, max_mod + 1) if m not in primes]
    
    def find_residue(m, classes):
        # classes is list of [r, m]
        # Check all r in [0, m)
        for r in range(m):
            ok = True
            for r_i, m_i in classes:
                if (r - r_i) % gcds[m][m_i] == 0:
                    ok = False
                    break
            if ok:
                return r
        return -1

    best_classes = []
    best_count = 0
    
    # Initial greedy with primes
    base_classes = [[0, p] for p in primes]
    best_classes = base_classes[:]
    best_count = len(base_classes)
    
    for restart in range(100):
        if time.time() > deadline - 10:
            break
        random.seed(restart)
        current = base_classes[:]
        used = set(primes)
        
        comps = composites[:]
        random.shuffle(comps)
        
        for m in comps:
            r = find_residue(m, current)
            if r != -1:
                current.append([r, m])
                used.add(m)
                
        if len(current) > best_count:
            best_count = len(current)
            best_classes = current[:]
            
        # Local search: try to add remaining
        unused = [m for m in composites if m not in used]
        if not unused:
            continue
            
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            random.shuffle(unused)
            added = False
            for m in unused:
                r = find_residue(m, current)
                if r != -1:
                    current.append([r, m])
                    used.add(m)
                    unused.remove(m)
                    added = True
                    break
            
            if not added:
                # Try a swap
                if current:
                    idx = random.randint(0, len(current) - 1)
                    r_old, m_old = current[idx]
                    # Try changing residue of m_old
                    for r_new in range(m_old):
                        if r_new == r_old:
                            continue
                        # Check if r_new is valid for m_old against others
                        # Remove current[idx]
                        other_classes = current[:idx] + current[idx+1:]
                        valid = True
                        for r_i, m_i in other_classes:
                            if (r_new - r_i) % gcds[m_old][m_i] == 0:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # If valid, try to add an unused m
                        current[idx][0] = r_new
                        for m in unused:
                            r_add = find_residue(m, current)
                            if r_add != -1:
                                current.append([r_add, m])
                                used.add(m)
                                unused.remove(m)
                                added = True
                                break
                        if not added:
                            current[idx][0] = r_old
                        else:
                            break
                    if not added:
                        current[idx][0] = r_old

        if len(current) > best_count:
            best_count = len(current)
            best_classes = current[:]

    return best_classes
