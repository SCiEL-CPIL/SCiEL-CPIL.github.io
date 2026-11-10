import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Candidate moduli: All even numbers in [20, 200]
    # This ensures gcd(m, n) >= 2 for all pairs, avoiding the gcd=1 impossibility.
    moduli = list(range(20, 201, 2))
    
    # We will try to assign residues to this set.
    # If it fails, we will remove elements (starting from largest m) until it succeeds.
    
    def is_valid(moduli, residues):
        n = len(moduli)
        for i in range(n):
            m_i = moduli[i]
            r_i = residues[i]
            for j in range(i + 1, n):
                m_j = moduli[j]
                r_j = residues[j]
                g = gcd(m_i, m_j)
                if r_i % g == r_j % g:
                    return False
        return True

    def solve(moduli):
        n = len(moduli)
        if n == 0:
            return []
        
        # Initialize residues randomly
        residues = [random.randint(0, m - 1) for m in moduli]
        
        # Local search
        # We track the number of conflicts
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Find a conflict
            conflict_found = False
            # Randomly sample pairs to speed up, or check all?
            # N is small (~90), so checking all is fine (8100 checks)
            # But we can just pick a random pair and check?
            # Better: find one conflict, fix it.
            
            # To be efficient, we can maintain a list of conflicts, but for N=90, full scan is fast enough in Python?
            # 8000 gcd calls might be slow if repeated 5000 times.
            # Let's optimize: only check a subset of pairs per iteration?
            # Or just pick a random pair. If it's a conflict, fix. If not, continue.
            # This is a standard heuristic.
            
            i, j = random.sample(range(n), 2)
            m_i = moduli[i]
            m_j = moduli[j]
            r_i = residues[i]
            r_j = residues[j]
            g = gcd(m_i, m_j)
            
            if r_i % g == r_j % g:
                # Conflict found. Change one of them.
                if random.random() < 0.5:
                    residues[i] = random.randint(0, m_i - 1)
                else:
                    residues[j] = random.randint(0, m_j - 1)
                conflict_found = True
        
        # Final check
        if is_valid(moduli, residues):
            return residues
        return None

    # Try to solve for the full set
    current_moduli = moduli[:]
    best_residues = None
    
    # We want to keep as many moduli as possible, prioritizing small m (high weight)
    # So we sort by m ascending. If we have to drop, drop the largest m.
    # The list is already sorted ascending.
    
    # Try full set
    while current_moduli:
        if time.time() > deadline:
            break
            
        residues = solve(current_moduli)
        if residues is not None:
            best_residues = residues
            break
        
        # If failed, remove the last element (largest m)
        current_moduli.pop()
        
    # If we still didn't find a solution (e.g. time out during solve), 
    # we might have a partial best_residues or None.
    # If None, we need a fallback.
    # The seed is always valid.
    if best_residues is None:
        # Fallback to seed
        return [[20 * j, j - 1] for j in range(1, 11)]
    
    # Construct result
    result = []
    for m, r in zip(current_moduli, best_residues):
        result.append([m, r])
        
    return result
