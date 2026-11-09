import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def _is_pairwise_disjoint(moduli, residues):
    """Check if the list of (m, r) pairs is pairwise disjoint."""
    n = len(moduli)
    for i in range(n):
        m1, r1 = moduli[i], residues[i]
        for j in range(i + 1, n):
            m2, r2 = moduli[j], residues[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                return False
    return True

def _score(moduli):
    """Compute the sum of 1/m for the given moduli."""
    return sum(1.0 / m for m in moduli)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Candidate moduli: 20 to 200
    candidates = list(range(20, 201))
    # Sort by 1/m descending (i.e., m ascending) to prioritize high-value moduli
    candidates.sort(key=lambda m: -1.0/m)
    
    best_moduli = []
    best_residues = []
    best_score = 0.0
    
    # Greedy construction with backtracking limited to a small depth for refinement
    # Strategy: 
    # 1. Fast greedy pass to get a good baseline.
    # 2. Try local improvements by swapping or adding/removing moduli.
    
    def greedy_construct(order):
        moduli = []
        residues = []
        for m in order:
            # Find a residue r in [0, m) such that (m, r) is disjoint from all existing
            found = False
            for r in range(m):
                ok = True
                for i in range(len(moduli)):
                    m2 = moduli[i]
                    r2 = residues[i]
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    moduli.append(m)
                    residues.append(r)
                    found = True
                    break
            if not found:
                continue
        return moduli, residues

    # Initial greedy pass
    moduli, residues = greedy_construct(candidates)
    best_moduli = moduli[:]
    best_residues = residues[:]
    best_score = _score(moduli)
    
    # Check time
    if time.time() > deadline:
        return [[m, r] for m, r in zip(best_moduli, best_residues)]
    
    # Local search / Hill Climbing
    # We can try to improve by:
    # 1. Removing a low-value modulus to free up space for a higher-value one? No, we already pick highest value first.
    # 2. The greedy choice of residue might block future moduli. We can try different residues for existing moduli.
    
    # Let's try a simple local search:
    # For each modulus in the current set, try changing its residue to see if it allows adding more moduli or improves score?
    # Actually, the score is fixed by the set of moduli. So we want to maximize the set of moduli.
    # The greedy algorithm picks moduli in order of 1/m. It might skip a modulus m because no residue works.
    # But maybe if we had chosen a different residue for an earlier modulus, m would have fit?
    
    # This is a constraint satisfaction problem. 
    # Let's try a randomized restart approach with limited time.
    
    import random
    rng = random.Random(42)
    
    def randomized_greedy():
        # Shuffle candidates slightly? No, we must keep high value ones.
        # Instead, randomize the order of residue selection? 
        # Or randomize the order of moduli among those with similar 1/m?
        
        # Let's try: 
        # 1. Start with empty set.
        # 2. At each step, pick the best available modulus (highest 1/m) that can be placed.
        # 3. When placing, pick a random valid residue instead of the first one.
        
        moduli = []
        residues = []
        remaining = candidates[:]
        
        while remaining:
            # Find best modulus that can be placed
            best_m = None
            best_r = None
            # Sort remaining by 1/m descending
            remaining.sort(key=lambda m: -1.0/m)
            
            for m in remaining:
                # Find a random valid residue
                valid_residues = []
                for r in range(m):
                    ok = True
                    for i in range(len(moduli)):
                        m2 = moduli[i]
                        r2 = residues[i]
                        g = math.gcd(m, m2)
                        if r % g == r2 % g:
                            ok = False
                            break
                    if ok:
                        valid_residues.append(r)
                
                if valid_residues:
                    best_m = m
                    best_r = rng.choice(valid_residues)
                    break
            
            if best_m is None:
                break
            
            moduli.append(best_m)
            residues.append(best_r)
            remaining.remove(best_m)
            
        return moduli, residues

    # Try a few randomized restarts
    for _ in range(100):
        if time.time() > deadline:
            break
        
        m_list, r_list = randomized_greedy()
        s = _score(m_list)
        if s > best_score:
            best_score = s
            best_moduli = m_list
            best_residues = r_list
    
    # Another strategy: 
    # The key insight from the prompt: "Restricting to moduli sharing a common prime factor (especially p=2) allows including many small moduli"
    # Let's try a strategy where we focus on even moduli.
    
    def even_greedy():
        moduli = []
        residues = []
        # Even moduli from 20 to 200
        even_candidates = [m for m in range(20, 201, 2)]
        even_candidates.sort(key=lambda m: -1.0/m)
        
        for m in even_candidates:
            found = False
            for r in range(m):
                ok = True
                for i in range(len(moduli)):
                    m2 = moduli[i]
                    r2 = residues[i]
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    moduli.append(m)
                    residues.append(r)
                    found = True
                    break
            if not found:
                continue
        return moduli, residues

    m_list, r_list = even_greedy()
    s = _score(m_list)
    if s > best_score:
        best_score = s
        best_moduli = m_list
        best_residues = r_list

    # Try multiples of 4
    def mult4_greedy():
        moduli = []
        residues = []
        candidates4 = [m for m in range(20, 201, 4)]
        candidates4.sort(key=lambda m: -1.0/m)
        
        for m in candidates4:
            found = False
            for r in range(m):
                ok = True
                for i in range(len(moduli)):
                    m2 = moduli[i]
                    r2 = residues[i]
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    moduli.append(m)
                    residues.append(r)
                    found = True
                    break
            if not found:
                continue
        return moduli, residues

    m_list, r_list = mult4_greedy()
    s = _score(m_list)
    if s > best_score:
        best_score = s
        best_moduli = m_list
        best_residues = r_list

    # Try multiples of 10
    def mult10_greedy():
        moduli = []
        residues = []
        candidates10 = [m for m in range(20, 201, 10)]
        candidates10.sort(key=lambda m: -1.0/m)
        
        for m in candidates10:
            found = False
            for r in range(m):
                ok = True
                for i in range(len(moduli)):
                    m2 = moduli[i]
                    r2 = residues[i]
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    moduli.append(m)
                    residues.append(r)
                    found = True
                    break
            if not found:
                continue
        return moduli, residues

    m_list, r_list = mult10_greedy()
    s = _score(m_list)
    if s > best_score:
        best_score = s
        best_moduli = m_list
        best_residues = r_list

    # Combine strategies: 
    # Try to fill gaps in the even_greedy solution with odd moduli?
    # Or try a hybrid: start with even moduli, then add odd ones if possible.
    
    def hybrid_greedy():
        moduli = []
        residues = []
        # First, place even moduli
        even_candidates = [m for m in range(20, 201, 2)]
        even_candidates.sort(key=lambda m: -1.0/m)
        
        for m in even_candidates:
            found = False
            for r in range(m):
                ok = True
                for i in range(len(moduli)):
                    m2 = moduli[i]
                    r2 = residues[i]
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    moduli.append(m)
                    residues.append(r)
                    found = True
                    break
            if not found:
                continue
        
        # Then, try to add odd moduli
        odd_candidates = [m for m in range(21, 201, 2)]
        odd_candidates.sort(key=lambda m: -1.0/m)
        
        for m in odd_candidates:
            found = False
            for r in range(m):
                ok = True
                for i in range(len(moduli)):
                    m2 = moduli[i]
                    r2 = residues[i]
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    moduli.append(m)
                    residues.append(r)
                    found = True
                    break
            if not found:
                continue
        
        return moduli, residues

    m_list, r_list = hybrid_greedy()
    s = _score(m_list)
    if s > best_score:
        best_score = s
        best_moduli = m_list
        best_residues = r_list

    # Final verification
    if not _is_pairwise_disjoint(best_moduli, best_residues):
        # Fallback to the simple seed solution
        return [[20 * j, j - 1] for j in range(1, 11)]

    return [[m, r] for m, r in zip(best_moduli, best_residues)]

# EVOLVE-BLOCK-END
