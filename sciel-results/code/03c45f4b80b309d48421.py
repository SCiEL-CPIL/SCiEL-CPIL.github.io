"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n_max = 512
    # Numbers 1..512
    nums = np.arange(1, n_max + 1)
    
    # Precompute LCM compatibility matrix
    # lcm(a,b) = a*b // gcd(a,b)
    # We need lcm(a,b) <= 512 for all pairs
    
    # Build compatibility: compat[i,j] = True if lcm(nums[i], nums[j]) <= 512
    # Use vectorized approach
    size = n_max
    # Create matrix of all pairs
    # For memory efficiency, we can compute row by row or use a smart approach
    
    # Actually, let's compute the conflict count for each element
    # conflict[i] = number of j such that lcm(nums[i], nums[j]) > 512
    
    # Vectorized: for each i, compute lcm with all j
    # lcm(a,b) = a*b//gcd(a,b)
    # We can compute gcd matrix using numpy
    
    # gcd matrix
    a = nums[:, None]  # (512, 1)
    b = nums[None, :]  # (1, 512)
    
    # Compute gcd
    # numpy doesn't have gcd directly in all versions, use np.gcd
    gcd_mat = np.gcd(a, b)  # (512, 512)
    
    # lcm = a * b // gcd
    # But a*b might overflow? No, max is 512*512 = 262144, fine for int64
    lcm_mat = (a * b) // gcd_mat  # (512, 512)
    
    # Compatibility: lcm <= 512
    compat = lcm_mat <= n_max  # (512, 512) boolean
    
    # For each element, count how many others it conflicts with
    # conflict_count[i] = number of j where not compat[i,j]
    conflict_count = (~compat).sum(axis=1)
    
    # Iterative removal: remove element with highest conflict count
    # Keep a set of active elements
    active = np.ones(size, dtype=bool)
    best_set = set(range(1, size + 1))
    best_size = size
    
    # Greedy removal
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        # Among active elements, find one with max conflicts (with other active elements)
        if not active.any():
            break
            
        # Recompute conflicts among active elements
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
        
        # Conflict count within active set
        sub_compat = compat[np.ix_(active_indices, active_indices)]
        sub_conflicts = (~sub_compat).sum(axis=1)
        
        # Find element with max conflicts
        max_idx_local = np.argmax(sub_conflicts)
        max_idx = active_indices[max_idx_local]
        
        if sub_conflicts[max_idx_local] == 0:
            # No more conflicts, we're done
            break
        
        # Remove this element
        active[max_idx] = False
        best_set.discard(max_idx + 1)
        best_size = len(best_set)
    
    # Now try to add back removed elements
    removed = [i for i in range(size) if not active[i]]
    # Sort removed by how few conflicts they have with current active set
    for r in removed:
        elapsed = time.time() - start_time
        if elapsed > deadline - 3:
            break
        
        # Check if r is compatible with all active elements
        if active[r]:
            continue
        # Check compatibility of r with all active
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            # Can add any
            active[r] = True
            best_set.add(r + 1)
            continue
        
        # Check if r is compatible with all active
        r_compat = compat[r, active_indices]
        if r_compat.all():
            active[r] = True
            best_set.add(r + 1)
    
    # Try more passes of add-back
    for _ in range(10):
        elapsed = time.time() - start_time
        if elapsed > deadline - 3:
            break
        
        added_any = False
        removed = [i for i in range(size) if not active[i]]
        # Sort by conflict count with active set (ascending)
        if len(removed) == 0:
            break
        
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            # Add all
            for r in removed:
                active[r] = True
                best_set.add(r + 1)
            break
        
        # For each removed, count conflicts with active
        sub_compat = compat[np.ix_(removed, active_indices)]
        sub_conflicts = (~sub_compat).sum(axis=1)
        
        # Try adding those with 0 conflicts first
        zero_conflict = np.where(sub_conflicts == 0)[0]
        for zc in zero_conflict:
            r = removed[zc]
            active[r] = True
            best_set.add(r + 1)
            added_any = True
            active_indices = np.where(active)[0]
        
        if not added_any:
            break
    
    # Final local search: try swapping
    for _ in range(50):
        elapsed = time.time() - start_time
        if elapsed > deadline - 2:
            break
        
        improved = False
        active_indices = np.where(active)[0]
        removed_indices = np.where(~active)[0]
        
        if len(removed_indices) == 0:
            break
        
        # For each removed element, check if it's compatible with all active
        for r in removed_indices:
            elapsed = time.time() - start_time
            if elapsed > deadline - 2:
                break
            
            if active[r]:
                continue
            
            if len(active_indices) == 0:
                active[r] = True
                best_set.add(r + 1)
                improved = True
                active_indices = np.where(active)[0]
                continue
            
            r_compat = compat[r, active_indices]
            if r_compat.all():
                active[r] = True
                best_set.add(r + 1)
                improved = True
                active_indices = np.where(active)[0]
    
    result = sorted(best_set)
    return result
# EVOLVE-BLOCK-END
