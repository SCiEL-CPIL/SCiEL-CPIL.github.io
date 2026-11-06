import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    base = 51975
    
    # Find all divisors of base
    divisors = []
    for i in range(1, int(base**0.5) + 1):
        if base % i == 0:
            divisors.append(i)
            if i != base // i:
                divisors.append(base // i)
    
    # Filter: m > 1, odd
    divisors = [m for m in divisors if m > 1 and m % 2 == 1]
    divisors.sort(reverse=True)
    
    # Precompute indices for each modulus and residue
    indices_by_m = {}
    for m in divisors:
        res_indices = []
        for r in range(m):
            res_indices.append(np.arange(r, base, m))
        indices_by_m[m] = res_indices

    covered = np.zeros(base, dtype=np.uint8)
    available_moduli = set(divisors)
    classes = []
    
    best_uncovered = base
    best_classes = []
    
    # Phase 1: Greedy set cover
    while available_moduli:
        if time.time() > deadline - 30:  # Leave time for local search
            break
            
        best_gain = -1
        best_m = None
        best_r = None
        best_indices = None
        
        for m in available_moduli:
            res_indices_list = indices_by_m[m]
            for r in range(m):
                idx = res_indices_list[r]
                gain = np.sum(covered[idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    best_indices = idx
                    
        if best_gain <= 0:
            break
            
        classes.append([best_m, best_r])
        covered[best_indices] = 1
        available_moduli.remove(best_m)
        
        current_uncovered = np.sum(covered == 0)
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_classes = [c[:] for c in classes]
            
        if best_uncovered == 0:
            break
            
    # Phase 2: Local search - try swapping classes to improve coverage
    if time.time() < deadline - 5 and best_uncovered > 0:
        # Start from best_classes found so far
        local_classes = [c[:] for c in best_classes]
        local_covered = np.zeros(base, dtype=np.uint8)
        for m, r in local_classes:
            local_covered[indices_by_m[m][r]] = 1
        local_uncovered = np.sum(local_covered == 0)
        used_moduli = set(m for m, r in local_classes)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try swapping each class with an unused modulus
            for i, (m_old, r_old) in enumerate(local_classes):
                if time.time() > deadline - 5:
                    break
                # Temporarily remove this class
                local_covered[indices_by_m[m_old][r_old]] = 0
                uncovered_after_remove = np.sum(local_covered == 0)
                
                # Try replacing with each unused modulus
                for m_new in divisors:
                    if m_new in used_moduli:
                        continue
                    res_list = indices_by_m[m_new]
                    for r_new in range(m_new):
                        idx_new = res_list[r_new]
                        # Count how many new residues this would cover
                        gain = np.sum(local_covered[idx_new] == 0)
                        # Net change: we lose coverage from old, gain from new
                        # But some of the new coverage might overlap with what we just removed
                        # Actually, we need to be more careful
                        # Let's compute: uncovered_after_remove - gain_from_new
                        # But gain_from_new is based on current state (without old)
                        # So net uncovered = uncovered_after_remove - gain
                        new_uncovered = uncovered_after_remove - gain
                        if new_uncovered < local_uncovered:
                            # Commit this swap
                            local_covered[idx_new] = 1
                            local_classes[i] = [m_new, r_new]
                            used_moduli.remove(m_old)
                            used_moduli.add(m_new)
                            local_uncovered = new_uncovered
                            improved = True
                            break  # Move to next class
                    if improved:
                        break
                if improved:
                    break
                    
        if local_uncovered < best_uncovered:
            best_uncovered = local_uncovered
            best_classes = local_classes
    
    return best_classes
