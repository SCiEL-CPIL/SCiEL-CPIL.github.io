import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
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
    divisors.sort()
    
    # Precompute indices for each modulus and residue
    indices_by_m = {}
    for m in divisors:
        res_indices = []
        for r in range(m):
            idx = np.arange(r, base, m)
            res_indices.append(idx)
        indices_by_m[m] = res_indices
    
    best_uncovered = base
    best_classes = []
    
    def greedy_build(covered, available_moduli, noise_scale=0.0):
        """Greedy set cover with optional noise."""
        classes = []
        covered = covered.copy()
        available_moduli = set(available_moduli)
        
        while available_moduli:
            if time.time() > deadline - 8:
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
                    if noise_scale > 0:
                        gain += random.uniform(0, noise_scale)
                    
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
            
        return classes, covered
    
    def local_search(classes, covered, max_iter=200):
        """Local search: swap residues, add/remove classes."""
        if time.time() > deadline - 5:
            return classes, covered
        
        classes = [c[:] for c in classes]
        covered = covered.copy()
        current_uncovered = int(np.sum(covered == 0))
        improved = True
        
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            if time.time() > deadline - 5:
                break
            
            used_moduli = set(m for m, r in classes)
            
            # Try swapping residue for existing classes
            for i in range(len(classes)):
                if time.time() > deadline - 5:
                    break
                
                m_i, r_i = classes[i]
                old_idx = indices_by_m[m_i][r_i]
                
                # Try all residues for this modulus
                best_r_new = None
                best_uncovered = current_uncovered
                
                for r_new in range(m_i):
                    if r_new == r_i:
                        continue
                    
                    new_idx = indices_by_m[m_i][r_new]
                    
                    # Compute uncovered after swap
                    temp_covered = covered.copy()
                    temp_covered[old_idx] = 0
                    temp_covered[new_idx] = 1
                    uncovered_after = int(np.sum(temp_covered == 0))
                    
                    if uncovered_after < best_uncovered:
                        best_uncovered = uncovered_after
                        best_r_new = r_new
                
                if best_r_new is not None:
                    classes[i] = [m_i, best_r_new]
                    covered = covered.copy()
                    covered[old_idx] = 0
                    covered[indices_by_m[m_i][best_r_new]] = 1
                    current_uncovered = best_uncovered
                    improved = True
                    break
            
            # Try adding a new class
            if not improved:
                for m in divisors:
                    if m in used_moduli:
                        continue
                    
                    best_r = None
                    best_gain = -1
                    for r in range(m):
                        idx = indices_by_m[m][r]
                        gain = int(np.sum(covered[idx] == 0))
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    
                    if best_gain > 0:
                        classes.append([m, best_r])
                        covered[indices_by_m[m][best_r]] = 1
                        current_uncovered = int(np.sum(covered == 0))
                        improved = True
                        break
            
            # Try removing a class that doesn't help
            if not improved:
                for i in range(len(classes)):
                    m_i, r_i = classes[i]
                    idx = indices_by_m[m_i][r_i]
                    
                    temp_covered = covered.copy()
                    temp_covered[idx] = 0
                    uncovered_after = int(np.sum(temp_covered == 0))
                    
                    if uncovered_after <= current_uncovered:
                        del classes[i]
                        covered = temp_covered
                        current_uncovered = uncovered_after
                        improved = True
                        break
        
        return classes, covered
    
    # Run multiple greedy strategies with different noise levels
    num_runs = 15
    
    for run in range(num_runs):
        if time.time() > deadline - 10:
            break
            
        covered = np.zeros(base, dtype=np.uint8)
        available_moduli = set(divisors)
        
        # Vary noise scale
        noise_scale = 0.0 if run == 0 else 0.05 + 0.01 * (run % 5)
        
        classes, covered = greedy_build(covered, available_moduli, noise_scale)
        
        current_uncovered = int(np.sum(covered == 0))
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_classes = [c[:] for c in classes]
            
            if best_uncovered == 0:
                return best_classes
        
        # Local search refinement
        if classes and time.time() < deadline - 5:
            refined_classes, refined_covered = local_search(classes, covered)
            refined_uncovered = int(np.sum(refined_covered == 0))
            if refined_uncovered < best_uncovered:
                best_uncovered = refined_uncovered
                best_classes = [c[:] for c in refined_classes]
                
                if best_uncovered == 0:
                    return best_classes
    
    # Final polish: try to improve the best solution further
    if best_classes and time.time() < deadline - 3:
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in best_classes:
            covered[indices_by_m[m][r]] = 1
        
        polished_classes, polished_covered = local_search(best_classes, covered, max_iter=100)
        polished_uncovered = int(np.sum(polished_covered == 0))
        if polished_uncovered < best_uncovered:
            best_uncovered = polished_uncovered
            best_classes = [c[:] for c in polished_classes]
    
    return best_classes
