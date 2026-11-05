"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 135  # safety margin before 150s limit
    
    # All odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Precompute index arrays for fast coverage updates
    # For each modulus m, store a list of arrays: indices[r] = np.arange(r, base, m)
    index_cache = {}
    for m in divisors:
        idx_list = []
        for r in range(m):
            idx_list.append(np.arange(r, base, m))
        index_cache[m] = idx_list
    
    # Greedy: process moduli in increasing order
    # Smaller moduli cover more residues per class, so they should be placed first
    # to maximize the "uncovered" pool that later (larger) moduli can exploit
    divisors.sort()
    
    covered = np.zeros(base, dtype=np.uint8)
    classes = []  # list of [m, r]
    
    for m in divisors:
        if time.time() > deadline - 30:
            break
        # Find residue r that covers most uncovered residues
        idx_list = index_cache[m]
        best_r = 0
        best_uncovered = -1
        for r in range(m):
            positions = idx_list[r]
            already_covered = covered[positions].sum()
            uncovered = len(positions) - already_covered
            if uncovered > best_uncovered:
                best_uncovered = uncovered
                best_r = r
        classes.append([m, best_r])
        covered[idx_list[best_r]] = 1
    
    # Local search: try reassigning residues for each class
    # Use efficient incremental coverage update
    def compute_uncovered(classes_list):
        cov = np.zeros(base, dtype=np.uint8)
        for m, r in classes_list:
            cov[index_cache[m][r]] = 1
        return base - cov.sum()
    
    current_score = compute_uncovered(classes)
    
    # Iterative improvement passes
    improved = True
    pass_count = 0
    while improved and time.time() < deadline - 20:
        improved = False
        pass_count += 1
        
        for i in range(len(classes)):
            if time.time() > deadline - 20:
                break
            m, r = classes[i]
            idx_list = index_cache[m]
            
            # Compute coverage without this class
            # We can do this by tracking which positions are covered by class i vs others
            # Simpler: temporarily set to 0, find best, restore
            old_positions = idx_list[r]
            old_vals = covered[old_positions].copy()
            covered[old_positions] = 0
            
            base_uncovered = (covered == 0).sum()
            
            best_r = r
            best_score = base_uncovered
            
            for new_r in range(m):
                if new_r == r:
                    continue
                positions = idx_list[new_r]
                would_cover = (covered[positions] == 0).sum()
                new_uncovered = base_uncovered - would_cover
                if new_uncovered < best_score:
                    best_score = new_uncovered
                    best_r = new_r
            
            # Restore
            covered[old_positions] = old_vals
            
            if best_r != r:
                classes[i][1] = best_r
                covered[old_positions] = 0
                covered[idx_list[best_r]] = 1
                current_score = best_score
                improved = True
    
    # Try removing classes that don't help
    if time.time() < deadline - 10:
        i = 0
        while i < len(classes) and time.time() < deadline - 10:
            m, r = classes[i]
            saved = classes.pop(i)
            score_without = compute_uncovered(classes)
            if score_without <= current_score:
                current_score = score_without
            else:
                classes.insert(i, saved)
                i += 1
    
    # Final polish: one more local search pass with tighter time budget
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(classes)):
                if time.time() > deadline - 3:
                    break
                m, r = classes[i]
                idx_list = index_cache[m]
                
                old_positions = idx_list[r]
                old_vals = covered[old_positions].copy()
                covered[old_positions] = 0
                
                base_uncovered = (covered == 0).sum()
                
                best_r = r
                best_score = base_uncovered
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    positions = idx_list[new_r]
                    would_cover = (covered[positions] == 0).sum()
                    new_uncovered = base_uncovered - would_cover
                    if new_uncovered < best_score:
                        best_score = new_uncovered
                        best_r = new_r
                
                # Restore
                covered[old_positions] = old_vals
                
                if best_r != r:
                    classes[i][1] = best_r
                    covered[old_positions] = 0
                    covered[idx_list[best_r]] = 1
                    current_score = best_score
                    improved = True
    
    return classes
# EVOLVE-BLOCK-END
