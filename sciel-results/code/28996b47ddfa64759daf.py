import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Helper: Check if (m, r) is valid with respect to set S
    def is_valid(m, r, S):
        for m_i, r_i in S:
            g = math.gcd(m, m_i)
            if g > 1:
                if r % g == r_i % g:
                    return False
        return True
    
    # Helper: Find a valid residue for m given S
    def find_valid_residue(m, S):
        for r in range(m):
            if is_valid(m, r, S):
                return r
        return None

    # 1. Greedy Construction
    # Select moduli in ascending order (highest weight first)
    current_set = []
    for m in range(20, 201):
        r = find_valid_residue(m, current_set)
        if r is not None:
            current_set.append([m, r])
    
    best_score = sum(1.0/m for m, r in current_set)
    best_set = [list(x) for x in current_set]
    
    # 2. Local Search
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # Phase 1: Try to add any missing modulus
        # Iterate from smallest m to largest m (highest weight first)
        for m in range(20, 201):
            # Check if m is already in set
            # Optimization: keep a set of moduli for O(1) lookup
            # But we are modifying list, so let's just check
            in_set = any(m_i == m for m_i, _ in current_set)
            if in_set:
                continue
            
            r = find_valid_residue(m, current_set)
            if r is not None:
                current_set.append([m, r])
                improved = True
                # Update best if better (it always is, since we added positive weight)
                new_score = sum(1.0/m_i for m_i, _ in current_set)
                if new_score > best_score:
                    best_score = new_score
                    best_set = [list(x) for x in current_set]
                break # Restart loop to check if more adds are possible
        
        if improved:
            continue
            
        # Phase 2: Try Swaps
        # We want to replace a large m (low weight) with a small m (high weight)
        # Sort current moduli descending to try removing the "worst" ones first
        moduli_in_set = sorted([m for m, _ in current_set], reverse=True)
        
        for m_out in moduli_in_set:
            if (time.time() - start_time) > deadline:
                break
            
            # Find index of m_out
            idx_out = -1
            for i, (m_i, r_i) in enumerate(current_set):
                if m_i == m_out:
                    idx_out = i
                    break
            
            if idx_out == -1:
                continue
            
            # Save the element to restore if no swap happens
            r_out = current_set[idx_out][1]
            
            # Temporarily remove
            current_set.pop(idx_out)
            
            swap_made = False
            # Try to add any m_in < m_out
            for m_in in range(20, m_out):
                in_set = any(m_i == m_in for m_i, _ in current_set)
                if in_set:
                    continue
                
                r_in = find_valid_residue(m_in, current_set)
                if r_in is not None:
                    current_set.append([m_in, r_in])
                    swap_made = True
                    improved = True
                    new_score = sum(1.0/m_i for m_i, _ in current_set)
                    if new_score > best_score:
                        best_score = new_score
                        best_set = [list(x) for x in current_set]
                    break
            
            if not swap_made:
                # Restore m_out
                current_set.insert(idx_out, [m_out, r_out])
            
            if swap_made:
                break # Restart main loop

    return best_set
