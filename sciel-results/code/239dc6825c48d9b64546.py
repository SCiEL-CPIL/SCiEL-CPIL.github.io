"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np

    base = 51975
    N = base

    # Generate all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size descending
    divisors_desc = sorted(divisors, reverse=True)
    
    counts = np.zeros(N, dtype=np.int32)
    classes = []
    
    # Greedy Phase
    for m in divisors_desc:
        unc_indices = np.where(counts == 0)[0]
        if len(unc_indices) == 0:
            break
        
        # Count uncovered residues in each class mod m
        counts_res = np.bincount(unc_indices % m, minlength=m)
        best_r = np.argmax(counts_res)
        
        if counts_res[best_r] > 0:
            classes.append([m, best_r])
            indices = np.arange(best_r, N, m)
            counts[indices] += 1
    
    # Local Search Phase (Coordinate Descent)
    current_uncovered = np.sum(counts == 0)
    
    for _ in range(1000):
        if current_uncovered == 0:
            break
            
        improved = False
        indices_to_process = list(range(len(classes)))
        np.random.shuffle(indices_to_process)
        
        for i in indices_to_process:
            m, r = classes[i]
            old_indices = np.arange(r, N, m)
            
            # Remove old class
            counts[old_indices] -= 1
            
            # Find best new residue
            unc_indices = np.where(counts == 0)[0]
            if len(unc_indices) == 0:
                # Covered everything
                counts[old_indices] += 1
                continue
            
            counts_res = np.bincount(unc_indices % m, minlength=m)
            best_r = np.argmax(counts_res)
            
            new_indices = np.arange(best_r, N, m)
            
            # Apply new class
            counts[new_indices] += 1
            
            new_uncovered = np.sum(counts == 0)
            
            if new_uncovered < current_uncovered:
                current_uncovered = new_uncovered
                classes[i] = [m, best_r]
                improved = True
            else:
                # Revert
                counts[new_indices] -= 1
                counts[old_indices] += 1
                
        if not improved:
            break
            
    return classes
# EVOLVE-BLOCK-END
