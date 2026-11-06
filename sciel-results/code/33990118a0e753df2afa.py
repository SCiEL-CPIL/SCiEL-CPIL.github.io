import time
import math
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    min_m = 20
    max_m = 200
    moduli = list(range(min_m, max_m + 1))
    
    # Precompute GCDs for speed
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
            
    def check_valid(m, r, state):
        """Check if (m, r) is valid given state {m2: r2}"""
        for m2, r2 in state.items():
            if m2 == m:
                continue
            g = gcd_table.get((m, m2), math.gcd(m, m2))
            if r % g == r2 % g:
                return False
        return True

    def greedy_construct():
        """
        Greedy construction with AC-3 style pruning.
        Processes moduli in order of 1/m (descending) to maximize score.
        Uses forward checking to prune residues that are inconsistent with assigned moduli.
        """
        state = {}
        # Sort moduli by 1/m descending (i.e., m ascending)
        for m in sorted(moduli, key=lambda x: 1.0/x, reverse=True):
            if time.time() > deadline:
                break
                
            # Try to find a residue for m that is consistent with all assigned moduli
            found = False
            # Try residues in order 0 to m-1
            for r in range(m):
                if check_valid(m, r, state):
                    state[m] = r
                    found = True
                    break
            
            if not found:
                # Try local search to find a valid residue
                # This is a simplified version; in a full CSP, we'd backtrack
                # For now, skip this modulus
                pass
                
        return state

    def local_search(state, max_iter=200):
        """
        Local search to improve the solution.
        Tries to swap or reassign residues to resolve conflicts and add more moduli.
        """
        current_state = dict(state)
        current_score = sum(1.0/m for m in current_state)
        best_state = dict(current_state)
        best_score = current_score
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Pick a random modulus to reassign
            if not current_state:
                break
                
            m_var = random.choice(list(current_state.keys()))
            old_r = current_state[m_var]
            
            # Try a new random residue
            new_r = random.randint(0, m_var - 1)
            if new_r == old_r:
                continue
                
            # Check if new assignment is valid
            if check_valid(m_var, new_r, current_state):
                current_state[m_var] = new_r
            else:
                # Try to find a better residue
                best_r = old_r
                min_conf = 0
                # Count conflicts for current assignment
                for m2, r2 in current_state.items():
                    if m2 == m_var:
                        continue
                    g = gcd_table.get((m_var, m2), math.gcd(m_var, m2))
                    if old_r % g == r2 % g:
                        min_conf += 1
                        
                # Try a few random residues
                for _ in range(10):
                    r_try = random.randint(0, m_var - 1)
                    if r_try == old_r:
                        continue
                    conf = 0
                    for m2, r2 in current_state.items():
                        if m2 == m_var:
                            continue
                        g = gcd_table.get((m_var, m2), math.gcd(m_var, m2))
                        if r_try % g == r2 % g:
                            conf += 1
                    if conf < min_conf:
                        min_conf = conf
                        best_r = r_try
                        
                if best_r != old_r:
                    current_state[m_var] = best_r
            
            # Try to add a new modulus
            unassigned = [m for m in moduli if m not in current_state]
            if unassigned:
                m_new = random.choice(unassigned)
                # Try to find a valid residue for m_new
                for r in range(m_new):
                    if check_valid(m_new, r, current_state):
                        current_state[m_new] = r
                        break
            
            # Calculate new score
            new_score = sum(1.0/m for m in current_state)
            if new_score > best_score:
                best_score = new_score
                best_state = dict(current_state)
                
        return best_state

    # Start with greedy construction
    current_state = greedy_construct()
    
    # Improve with local search
    best_state = local_search(current_state)
    
    # Convert to list of lists
    result = [[m, r] for m, r in best_state.items()]
    return result
