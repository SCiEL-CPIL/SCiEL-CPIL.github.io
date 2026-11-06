"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    # Use a safer deadline to ensure we return before the 120s/150s kill switch
    deadline = time.time() + 100
    
    # Precompute gcd table for all pairs in [20, 200]
    gcd_table = {}
    for a in range(20, 201):
        for b in range(20, 201):
            gcd_table[(a, b)] = gcd(a, b)
    
    def is_valid_residue(m, r, selected):
        """Check if residue r for modulus m is compatible with all selected (m_i, r_i)."""
        for (m_i, r_i) in selected:
            g = gcd_table[(m, m_i)]
            if r % g == r_i % g:
                return False
        return True
    
    def find_valid_residue(m, selected):
        """Find the smallest valid residue for modulus m, or None."""
        for r in range(m):
            if is_valid_residue(m, r, selected):
                return r
        return None
    
    def compute_score(selected):
        """Compute sum of 1/m as a float."""
        return sum(1.0 / m for m, r in selected)
    
    # Phase 1: Greedy construction
    # Strategy: process moduli in ascending order (smaller m = higher 1/m weight)
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        if time.time() > deadline - 5:
            break
        r = find_valid_residue(m, selected)
        if r is not None:
            selected.append((m, r))
    
    best_score = compute_score(selected)
    best_selected = list(selected)
    
    # Phase 2: Local search
    # We limit the number of iterations and add strict time checks in inner loops.
    if time.time() < deadline - 5:
        # Move 1: Aggressively try to add missing moduli
        # This is the most direct way to increase score.
        while time.time() < deadline - 5:
            current_mods = set(m for m, r in selected)
            added_any = False
            # Sort missing moduli by 1/m (descending) to prioritize high-value additions
            missing = [m for m in range(20, 201) if m not in current_mods]
            missing.sort(key=lambda x: -1.0/x)
            
            for m in missing:
                if time.time() > deadline - 5:
                    break
                r = find_valid_residue(m, selected)
                if r is not None:
                    selected.append((m, r))
                    added_any = True
                    # Update best if score improved (it always does if we add)
                    cur_score = compute_score(selected)
                    if cur_score > best_score:
                        best_score = cur_score
                        best_selected = list(selected)
                    break # Restart loop to re-evaluate with new set
            
            if not added_any:
                break
        
        # Move 2: Limited Swaps to enable further additions
        # We only do a few passes to avoid O(N^3) blowup
        max_swap_passes = 10
        for _ in range(max_swap_passes):
            if time.time() > deadline - 2:
                break
            
            improved = False
            n = len(selected)
            
            # Limit the scope of swaps to reduce complexity
            # Only try swaps for the first half of the list against the second half?
            # Or just iterate with early exit.
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                m_i, r_i = selected[i]
                for j in range(i + 1, n):
                    if time.time() > deadline - 2:
                        break
                    m_j, r_j = selected[j]
                    
                    # Quick check: if gcd is 1, any swap is valid locally? No, need to check others.
                    # But if gcd(m_i, m_j) == 1, the constraint between i and j is always satisfied.
                    # We still need to check against k.
                    
                    # Check if swap is valid against all others
                    valid = True
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        m_k, r_k = selected[k]
                        
                        # Check m_i with new residue r_j
                        g_ik = gcd_table[(m_i, m_k)]
                        if r_j % g_ik == r_k % g_ik:
                            valid = False
                            break
                        
                        # Check m_j with new residue r_i
                        g_jk = gcd_table[(m_j, m_k)]
                        if r_i % g_jk == r_k % g_jk:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        selected[i] = (m_i, r_j)
                        selected[j] = (m_j, r_i)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
            
            # After a swap, try to add more moduli
            if time.time() < deadline - 2:
                current_mods = set(m for m, r in selected)
                missing = [m for m in range(20, 201) if m not in current_mods]
                missing.sort(key=lambda x: -1.0/x)
                
                for m in missing:
                    if time.time() > deadline - 2:
                        break
                    r = find_valid_residue(m, selected)
                    if r is not None:
                        selected.append((m, r))
                        cur_score = compute_score(selected)
                        if cur_score > best_score:
                            best_score = cur_score
                            best_selected = list(selected)
                        break
    
    return [[m, r] for m, r in best_selected]
# EVOLVE-BLOCK-END
