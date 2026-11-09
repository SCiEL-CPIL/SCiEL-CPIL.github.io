import math
import time
import random

MAX_MOD = 128

def gcd(a, b):
    return math.gcd(a, b)

def is_disjoint(r1, m1, r2, m2):
    g = gcd(m1, m2)
    if g == 1:
        return True
    return ((r2 - r1) % g) != 0

def verify_solution(pairs):
    n = len(pairs)
    for i in range(n):
        r1, m1 = pairs[i]
        if m1 < 2 or m1 > MAX_MOD or r1 < 0 or r1 >= m1:
            return False
        for j in range(i + 1, n):
            r2, m2 = pairs[j]
            if m1 == m2:
                return False
            if not is_disjoint(r1, m1, r2, m2):
                return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s kill
    
    # Candidate moduli: All odd moduli in [3, 127] and two even powers of 2
    odd_moduli = [m for m in range(3, MAX_MOD + 1, 2)]
    even_moduli = [2, 4]
    
    all_moduli = odd_moduli + even_moduli
    
    # Sort by modulus size (ascending) to handle small domains first
    sorted_moduli = sorted(all_moduli)
    
    def backtrack(idx, current_pairs):
        if time.time() > deadline:
            return None
        
        if idx == len(sorted_moduli):
            return current_pairs[:]
        
        m = sorted_moduli[idx]
        
        # Try residues in random order
        residues = list(range(m))
        random.shuffle(residues)
        
        for r in residues:
            ok = True
            for (r2, m2) in current_pairs:
                if not is_disjoint(r, m, r2, m2):
                    ok = False
                    break
            
            if ok:
                result = backtrack(idx + 1, current_pairs + [[r, m]])
                if result is not None:
                    return result
        
        return None

    # Try deterministic order first
    solution = backtrack(0, [])
    
    if solution is None:
        # If it fails, try a few random orders
        for _ in range(50):
            if time.time() > deadline:
                break
            random.shuffle(sorted_moduli)
            solution = backtrack(0, [])
            if solution is not None:
                break
    
    if solution is not None and verify_solution(solution):
        return solution
    
    # Fallback: Return the 2 even moduli
    return [[0, 2], [1, 4]]
