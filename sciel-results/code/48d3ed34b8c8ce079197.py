import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)
    
    # Map modulus value to index for quick lookup if needed, but we work with indices
    # Precompute GCDs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        gcd_tab[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g
            
    # Precompute forbidden masks
    # forbidden[i][j][r] is a bitmask of residues for j that are forbidden if i takes residue r
    # We only need this for i < j in the ordering? 
    # Let's use a fixed order: 0 to n-1.
    # Or we can sort moduli? 
    # If we sort moduli, the indices change. 
    # Let's stick to indices 0..n-1 corresponding to moduli 2..128.
    # But "Most Constrained First" is better.
    # Let's calculate a "degree" or "constraint weight" for each modulus.
    # Weight = sum of gcd(m_i, m_j) for j != i? 
    # Or just use the modulus value itself? Smaller moduli have fewer residues, so they are more constrained in domain size, but larger moduli have more potential conflicts.
    # Actually, small moduli (like 2) have very few choices. 
    # Let's order by ascending modulus. This is a common heuristic.
    
    # However, the precomputation `forbidden[i][j][r]` depends on the order if we only compute forward.
    # Let's compute `forbidden_masks[i][j]` as a list of length `moduli[i]`? No, that's huge.
    # `forbidden[i][j]` is a list of length `moduli[i]`, where each element is a bitmask for `moduli[j]`.
    # Total memory: sum_{i} moduli[i] * n. 
    # sum_{i=2}^{128} i * 127 approx 127 * (128*129/2) approx 127 * 8256 approx 1 million integers.
    # Each integer is a Python int (big int). This is fine.
    
    forbidden_masks = [[None] * n for _ in range(n)]
    for i in range(n):
        m_i = moduli[i]
        for j in range(n):
            if i == j:
                continue
            m_j = moduli[j]
            g = gcd_tab[i][j]
            masks = [0] * m_i
            for r in range(m_i):
                # Forbidden residues for j are r_j such that r_j % g == r % g
                start = r % g
                mask = 0
                for k in range(start, m_j, g):
                    mask |= (1 << k)
                masks[r] = mask
            forbidden_masks[i][j] = masks

    # Backtracking
    domains = [(1 << moduli[i]) - 1 for i in range(n)]
    assigned = [-1] * n
    
    # Order: Ascending modulus
    order = list(range(n))
    # order.sort(key=lambda i: moduli[i]) # Already sorted
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
            
        i = order[idx]
        m_i = moduli[i]
        mask = domains[i]
        
        if mask == 0:
            return False
            
        # Try residues
        r = 0
        while mask:
            if mask & 1:
                # Try residue r
                assigned[i] = r
                
                # Update domains for future variables
                old_domains = [0] * n
                changed = []
                success = True
                
                for next_idx in range(idx + 1, n):
                    j = order[next_idx]
                    # Apply constraint from i to j
                    forb_mask = forbidden_masks[i][j][r]
                    old_domains[j] = domains[j]
                    new_domain = domains[j] & ~forb_mask
                    if new_domain == 0:
                        success = False
                        break
                    domains[j] = new_domain
                    changed.append(j)
                    
                if success:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                for j in changed:
                    domains[j] = old_domains[j]
                    
                assigned[i] = -1
                
            mask >>= 1
            r += 1
            
        return False

    # Run backtracking
    if backtrack(0):
        return [[assigned[i], moduli[i]] for i in range(n)]
        
    # Fallback: Randomized Greedy
    best_solution = []
    for _ in range(1000):
        if time.time() > deadline:
            break
        current_solution = []
        order_rand = list(range(n))
        random.shuffle(order_rand)
        
        valid = True
        temp_assigned = {}
        for i in order_rand:
            m = moduli[i]
            # Find valid residue
            valid_res = None
            for r in range(m):
                ok = True
                for j, r_j in temp_assigned.items():
                    g = gcd_tab[i][j]
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                if ok:
                    valid_res = r
                    break
            if valid_res is None:
                valid = False
                break
            temp_assigned[i] = valid_res
            
        if valid:
            if len(temp_assigned) > len(best_solution):
                best_solution = temp_assigned
                
    if best_solution:
        return [[r, moduli[i]] for i, r in best_solution.items()]
        
    return [[0, 2]]
