import math
import time
import random

def _is_compatible(m, r, selected):
    """Check if (m, r) is compatible with all in selected."""
    for (m_i, r_i) in selected:
        g = math.gcd(m, m_i)
        if r % g == r_i % g:
            return False
    return True

def _find_valid_residue(m, selected):
    """Find a valid residue for modulus m compatible with selected."""
    for r in range(m):
        if _is_compatible(m, r, selected):
            return r
    return None

def _score(selected):
    """Compute score as sum of 1/m."""
    return sum(1.0 / m for m, r in selected)

def _local_search(selected, deadline):
    """Improve selected via local search: try changing residues, swapping moduli."""
    best = list(selected)
    best_score = _score(best)
    n = len(best)
    
    # Precompute gcd matrix for speed
    moduli = [m for m, r in best]
    gcds = [[math.gcd(moduli[i], moduli[j]) for j in range(n)] for i in range(n)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 1: change residue of existing modulus
        for i in range(n):
            if time.time() > deadline:
                return best, best_score
            m_i = moduli[i]
            old_r = best[i][1]
            # Try other residues
            for r in range(m_i):
                if r == old_r:
                    continue
                # Check compatibility with all other moduli
                compatible = True
                for j in range(n):
                    if j == i:
                        continue
                    g = gcds[i][j]
                    if r % g == best[j][1] % g:
                        compatible = False
                        break
                if compatible:
                    # Score doesn't change (same modulus), but we might enable future improvements
                    # For now, just track if we can change residue without breaking
                    pass
        
        # Try 2: replace a modulus with a different one
        for i in range(n):
            if time.time() > deadline:
                return best, best_score
            m_old, r_old = best[i]
            # Try replacing with moduli not currently in best
            current_moduli_set = set(moduli)
            for m_new in range(20, 201):
                if m_new in current_moduli_set:
                    continue
                if m_new == m_old:
                    continue
                # Check if we can find a valid residue for m_new
                r_new = None
                for r in range(m_new):
                    compatible = True
                    for j in range(n):
                        if j == i:
                            continue
                        g = math.gcd(m_new, moduli[j])
                        if r % g == best[j][1] % g:
                            compatible = False
                            break
                    if compatible:
                        r_new = r
                        break
                if r_new is not None:
                    # Compute new score
                    new_score = best_score - 1.0/m_old + 1.0/m_new
                    if new_score > best_score + 1e-12:
                        # Accept the change
                        best[i] = (m_new, r_new)
                        moduli[i] = m_new
                        # Update gcds
                        for j in range(n):
                            if j != i:
                                gcds[i][j] = math.gcd(m_new, moduli[j])
                                gcds[j][i] = gcds[i][j]
                        best_score = new_score
                        improved = True
                        break
        
        # Try 3: add a new modulus
        if time.time() < deadline:
            current_moduli_set = set(moduli)
            for m_new in range(20, 201):
                if m_new in current_moduli_set:
                    continue
                r_new = None
                for r in range(m_new):
                    compatible = True
                    for j in range(n):
                        g = math.gcd(m_new, moduli[j])
                        if r % g == best[j][1] % g:
                            compatible = False
                            break
                    if compatible:
                        r_new = r
                        break
                if r_new is not None:
                    new_score = best_score + 1.0/m_new
                    if new_score > best_score + 1e-12:
                        best.append((m_new, r_new))
                        moduli.append(m_new)
                        n += 1
                        # Update gcds
                        for j in range(n-1):
                            gcds[i][j] = math.gcd(m_new, moduli[j]) if i < n-1 else math.gcd(m_new, moduli[j])
                            if i < n-1:
                                gcds[j][i] = gcds[i][j]
                        best_score = new_score
                        improved = True
                        break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    selected = []  # List of (m, r)
    
    # Phase 1: Greedy construction
    for m in range(20, 201):
        if time.time() > deadline - 60:  # Reserve 60s for local search
            break
        
        # Try to find a residue r for m that is compatible with all selected
        r = _find_valid_residue(m, selected)
        if r is not None:
            selected.append((m, r))
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 5:
        selected, score = _local_search(selected, deadline - 5)
    
    return [[m, r] for (m, r) in selected]
# EVOLVE-BLOCK-END
