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
    
    # Precompute indices
    indices_by_m = {}
    for m in divisors:
        res_indices = []
        for r in range(m):
            idx = np.arange(r, base, m)
            res_indices.append(idx)
        indices_by_m[m] = res_indices

    best_uncovered = base
    best_classes = []
    
    def local_search(classes, covered, max_iter=500):
        """Try swap operations to improve coverage."""
        if time.time() > deadline - 5:
            return classes, covered
        
        classes = [c[:] for c in classes]
        covered = covered.copy()
        current_uncovered = np.sum(covered == 0)
        improved = True
        
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            if time.time() > deadline - 5:
                break
            
            used_moduli = set(m for m, r in classes)
            
            for i, (m_i, r_i) in enumerate(classes):
                if time.time() > deadline - 5:
                    break
                
                # Try swapping class i with a different residue for same modulus
                for r_new in range(m_i):
                    if r_new == r_i:
                        continue
                    
                    # Compute gain: how many new residues covered minus how many lost
                    old_idx = indices_by_m[m_i][r_i]
                    new_idx = indices_by_m[m_i][r_new]
                    
                    # Residues that would be lost (covered only by this class)
                    # and residues that would be gained
                    # Approximate: count uncovered in new_idx and covered-only-in-old in new_idx
                    # More precisely: 
                    # gain = |new_idx ∩ uncovered| - |old_idx ∩ (covered only by this class) ∩ new_idx|
                    # But we can approximate:
                    new_uncovered = np.sum(covered[new_idx] == 0)
                    old_only = np.sum((covered[old_idx] == 1) & (covered[new_idx] == 0))
                    # Actually, we need to be more careful
                    # Let's compute the net change
                    # Remove old class, see what becomes uncovered, then add new class
                    temp_covered = covered.copy()
                    temp_covered[old_idx] = 0
                    uncovered_after_remove = np.sum(temp_covered == 0)
                    temp_covered[new_idx] = 1
                    uncovered_after_add = np.sum(temp_covered == 0)
                    
                    net_change = uncovered_after_add - current_uncovered
                    if net_change < 0:
                        # Improvement
                        classes[i] = [m_i, r_new]
                        covered = temp_covered
                        current_uncovered = uncovered_after_add
                        improved = True
                        break
                
                if improved:
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
                        gain = np.sum(covered[idx] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    
                    if best_gain > 0:
                        classes.append([m, best_r])
                        covered[indices_by_m[m][best_r]] = 1
                        current_uncovered = np.sum(covered == 0)
                        improved = True
                        break
            
            # Try removing a class that doesn't help
            if not improved:
                for i in range(len(classes)):
                    m_i, r_i = classes[i]
                    idx = indices_by_m[m_i][r_i]
                    
                    # Check if removing this class increases uncovered count
                    temp_covered = covered.copy()
                    temp_covered[idx] = 0
                    uncovered_after_remove = np.sum(temp_covered == 0)
                    
                    if uncovered_after_remove == current_uncovered:
                        # Removing doesn't hurt
                        del classes[i]
                        covered = temp_covered
                        improved = True
                        break
        
        return classes, covered

    # Run multiple greedy strategies
    num_runs = 10
    
    for run in range(num_runs):
        if time.time() > deadline - 10:
            break
            
        covered = np.zeros(base, dtype=np.uint8)
        available_moduli = set(divisors)
        classes = []
        
        while available_moduli:
            if time.time() > deadline - 8:
                break
                
            best_gain = -1
            best_m = None
            best_r = None
            best_indices = None
            
            # Add noise to gain to randomize selection among similar options
            noise_scale = 0.1 if run > 0 else 0.0
            
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
                
            # Commit
            classes.append([best_m, best_r])
            covered[best_indices] = 1
            available_moduli.remove(best_m)
            
            current_uncovered = np.sum(covered == 0)
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in classes]
                
            if best_uncovered == 0:
                return best_classes
        
        # Local search refinement
        if classes and time.time() < deadline - 5:
            refined_classes, refined_covered = local_search(classes, covered)
            refined_uncovered = np.sum(refined_covered == 0)
            if refined_uncovered < best_uncovered:
                best_uncovered = refined_uncovered
                best_classes = [c[:] for c in refined_classes]
                if best_uncovered == 0:
                    return best_classes

    return best_classes
