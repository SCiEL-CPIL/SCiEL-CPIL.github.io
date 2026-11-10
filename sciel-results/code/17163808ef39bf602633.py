"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    def check_deadline():
        return time.time() > deadline
    
    moduli = list(range(20, 201))
    n = len(moduli)
    
    # Precompute gcd table
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_compatible(selected, m_idx, r):
        """Check if residue r for moduli[m_idx] is compatible with all selected classes."""
        for (s_idx, s_r) in selected:
            d = gcd_table[m_idx][s_idx]
            if d > 1 and r % d == s_r % d:
                return False
        return True
    
    def greedy(order, residue_order_func=None):
        """
        Greedy: process moduli in given order, pick best compatible residue.
        residue_order_func: if provided, returns a list of residues to try in order.
        """
        selected = []
        score = 0.0
        
        for m_idx in order:
            if check_deadline():
                break
            m = moduli[m_idx]
            
            # Determine residue order
            if residue_order_func:
                residues = residue_order_func(m_idx, m, selected)
            else:
                residues = list(range(m))
            
            for r in residues:
                if is_compatible(selected, m_idx, r):
                    selected.append((m_idx, r))
                    score += 1.0 / m
                    break
        
        return selected, score
    
    # Strategy 1: Simple greedy, smallest m first, smallest residue first
    order1 = list(range(n))
    result1, score1 = greedy(order1)
    
    if check_deadline():
        return [[moduli[m_idx], r] for (m_idx, r) in result1]
    
    # Strategy 2: Smallest m first, but try residues that are "most flexible"
    # A residue r for m is flexible if for many future moduli m', gcd(m,m')=1 or r is compatible
    def flexible_residue_order(m_idx, m, selected):
        """Order residues by how few conflicts they cause with unprocessed moduli."""
        unprocessed = [i for i in range(n) if i not in [s[0] for s in selected]]
        residues = list(range(m))
        
        def conflict_count(r):
            count = 0
            for j in unprocessed:
                d = gcd_table[m_idx][j]
                if d > 1:
                    # This residue r might conflict with some residue for moduli[j]
                    # But we don't know what residue will be picked, so just count potential conflicts
                    count += 1  # conservative
            return count
        
        # Actually, better: try residues that are coprime-friendly
        # Residues that are 0 mod d for many d are bad
        # Let's just try all and pick the one with fewest immediate conflicts
        # For now, use a heuristic: prefer residues that are small (they tend to be compatible)
        return residues
    
    result2, score2 = greedy(order1, flexible_residue_order)
    
    if check_deadline():
        best = result1 if score1 >= score2 else result2
        return [[moduli[m_idx], r] for (m_idx, r) in best]
    
    # Strategy 3: Process moduli in order of decreasing "constraint" 
    # Moduli with many non-coprime partners should be processed first
    def constraint_score(m_idx):
        """Higher = more constrained (fewer coprime partners)."""
        coprime_count = sum(1 for j in range(n) if j != m_idx and gcd_table[m_idx][j] == 1)
        return -coprime_count  # negative so sorted ascending = most constrained first
    
    order3 = sorted(range(n), key=lambda i: (-1.0 / moduli[i], constraint_score(i)))
    result3, score3 = greedy(order3)
    
    if check_deadline():
        best = result1
        best_score = score1
        if score2 > best_score:
            best, best_score = result2, score2
        if score3 > best_score:
            best, best_score = result3, score3
        return [[moduli[m_idx], r] for (m_idx, r) in best]
    
    # Strategy 4: Local search - start from best greedy, try swaps
    import random
    random.seed(123)
    
    best_result = result1
    best_score = score1
    if score2 > best_score:
        best_result, best_score = result2, score2
    if score3 > best_score:
        best_result, best_score = result3, score3
    
    # Convert to dict for easy manipulation
    best_dict = {m_idx: r for (m_idx, r) in best_result}
    
    for iteration in range(200):
        if check_deadline():
            break
        
        # Try to add a new modulus
        used_indices = set(best_dict.keys())
        candidates = [i for i in range(n) if i not in used_indices]
        
        if not candidates:
            break
        
        # Pick a random candidate
        m_idx = random.choice(candidates)
        m = moduli[m_idx]
        
        # Find a compatible residue
        found = False
        for r in range(m):
            if is_compatible(list(best_dict.items()), m_idx, r):
                best_dict[m_idx] = r
                best_score += 1.0 / m
                found = True
                break
        
        if not found:
            # Try removing a conflicting modulus and re-adding
            # Find which moduli conflict with all residues of m
            for r in range(m):
                conflicts = []
                for (s_idx, s_r) in list(best_dict.items()):
                    d = gcd_table[m_idx][s_idx]
                    if d > 1 and r % d == s_r % d:
                        conflicts.append(s_idx)
                
                if len(conflicts) < 3:  # Only if few conflicts
                    # Try removing one conflict and re-adding m
                    for c_idx in conflicts:
                        removed_score = 1.0 / moduli[c_idx]
                        new_score = best_score - removed_score + 1.0 / m
                        if new_score > best_score:
                            del best_dict[c_idx]
                            best_dict[m_idx] = r
                            best_score = new_score
                            break
                    break
    
    return [[moduli[m_idx], r] for (m_idx, r) in best_dict.items()]
# EVOLVE-BLOCK-END

