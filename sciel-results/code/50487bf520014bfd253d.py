"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    base = 51975
    
    # Generate all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size descending
    divisors_desc = sorted(divisors, reverse=True)
    
    covered_count = np.zeros(base, dtype=np.int32)
    classes = []
    
    def get_class_stats(arr, m):
        """
        Returns zeros_count and ones_count arrays of length m.
        zeros_count[r] = count of i in [0, base) s.t. i%m==r and arr[i]==0
        ones_count[r] = count of i in [0, base) s.t. i%m==r and arr[i]==1
        """
        full_blocks = base // m
        remainder = base % m
        
        zeros_count = np.zeros(m, dtype=np.int32)
        ones_count = np.zeros(m, dtype=np.int32)
        
        if full_blocks > 0:
            block = arr[:full_blocks * m].reshape(full_blocks, m)
            zeros_count += (block == 0).sum(axis=0)
            ones_count += (block == 1).sum(axis=0)
            
        if remainder > 0:
            rem_indices = np.arange(full_blocks * m, base)
            rem_residues = rem_indices % m
            rem_vals = arr[rem_indices]
            
            zeros_count += np.bincount(rem_residues, weights=(rem_vals == 0).astype(np.int32), minlength=m)[:m]
            ones_count += np.bincount(rem_residues, weights=(rem_vals == 1).astype(np.int32), minlength=m)[:m]
            
        return zeros_count, ones_count

    # Greedy Phase: Large moduli first
    for m in divisors_desc:
        zeros_count, _ = get_class_stats(covered_count, m)
        best_r = np.argmax(zeros_count)
        
        classes.append([m, best_r])
        covered_count[best_r::m] += 1
        
    # Local Search Phase
    max_iterations = 50
    current_uncovered = np.sum(covered_count == 0)
    
    for _ in range(max_iterations):
        improved = False
        
        for i, (m, r) in enumerate(classes):
            # Calculate newly_uncovered if we remove r
            newly_uncovered = np.sum(covered_count[r::m] == 1)
            
            # Temporarily remove r
            covered_count[r::m] -= 1
            
            # Calculate stats for temp state
            zeros_count, _ = get_class_stats(covered_count, m)
            
            # Find best r_new (maximizes newly_covered = zeros in temp state)
            best_r_new = np.argmax(zeros_count)
            newly_covered = zeros_count[best_r_new]
            
            # Check improvement
            if newly_covered > newly_uncovered:
                if best_r_new != r:
                    covered_count[best_r_new::m] += 1
                    classes[i] = [m, best_r_new]
                    current_uncovered = current_uncovered + newly_uncovered - newly_covered
                    improved = True
                else:
                    # best_r_new is r, just add it back
                    covered_count[r::m] += 1
            else:
                # No improvement, revert removal
                covered_count[r::m] += 1
                
        if not improved:
            break
            
    return classes
# EVOLVE-BLOCK-END
