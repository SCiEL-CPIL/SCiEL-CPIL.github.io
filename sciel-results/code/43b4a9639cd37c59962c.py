import time
import random
from itertools import combinations
import numpy as np

def construct():
    start = time.time()
    deadline = start + 55  # safety margin
    
    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}
    
    # Precompute: for each pair (i, j), the intersection A & B
    # and for each candidate C, check if (A&C) == (A&B) and (B&C) == (A&B)
    # Vectorize: for family indices F, compute all pairwise intersections
    # and check against candidate C
    
    # Precompute mask array for numpy
    masks_arr = np.array(all_masks, dtype=np.int32)
    
    def can_add(C, fam_indices):
        """Check if mask C can be added to family given by indices."""
        if len(fam_indices) < 2:
            return True
        fam_masks = masks_arr[fam_indices]
        n = len(fam_masks)
        # Compute all pairwise intersections
        # For each pair (i,j), I = fam_masks[i] & fam_masks[j]
        # Check if (fam_masks[i] & C) == I and (fam_masks[j] & C) == I
        # Vectorize: for each i, compute (fam_masks[i] & C) for all i
        C_int = C
        # Compute intersections of C with each family member
        inter_C = fam_masks & C_int  # shape (n,)
        # For each pair (i,j), check if inter_C[i] == (fam_masks[i] & fam_masks[j]) 
        # and inter_C[j] == (fam_masks[i] & fam_masks[j])
        # i.e., inter_C[i] == inter_C[j] == (fam_masks[i] & fam_masks[j])
        # This is equivalent to: for all i != j, inter_C[i] == (fam_masks[i] & fam_masks[j])
        # and inter_C[j] == (fam_masks[i] & fam_masks[j])
        # So inter_C[i] == inter_C[j] and inter_C[i] == (fam_masks[i] & fam_masks[j])
        
        # Compute pairwise intersections matrix
        # inter_matrix[i,j] = fam_masks[i] & fam_masks[j]
        inter_matrix = np.bitwise_and(fam_masks[:, np.newaxis], fam_masks[np.newaxis, :])
        # For each pair (i,j), check:
        # inter_C[i] == inter_matrix[i,j] and inter_C[j] == inter_matrix[i,j]
        # i.e., inter_C[i] == inter_matrix[i,j] and inter_C[j] == inter_matrix[i,j]
        # This means inter_C[i] == inter_C[j] == inter_matrix[i,j]
        
        # Check condition: inter_C[i] == inter_matrix[i,j] for all i,j
        # and inter_C[j] == inter_matrix[i,j] for all i,j
        # Combined: inter_C[i] == inter_matrix[i,j] and inter_C[j] == inter_matrix[i,j]
        # => inter_C[i] == inter_C[j] and inter_C[i] == inter_matrix[i,j]
        
        # For each i, check if inter_C[i] == inter_matrix[i,:] for all j != i
        # and inter_C[j] == inter_matrix[i,j]
        
        # Simpler: for each pair (i,j) with i<j, check if inter_C[i] == inter_matrix[i,j] 
        # and inter_C[j] == inter_matrix[i,j]
        
        # Create upper triangle mask
        tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        # For each (i,j) in upper triangle, check condition
        # cond[i,j] = (inter_C[i] == inter_matrix[i,j]) & (inter_C[j] == inter_matrix[i,j])
        cond_i = inter_C[np.newaxis, :] == inter_matrix  # cond_i[i,j] = (inter_C[j] == inter_matrix[i,j])
        cond_j = inter_C[:, np.newaxis] == inter_matrix  # cond_j[i,j] = (inter_C[i] == inter_matrix[i,j])
        # Wait, let me re-derive:
        # We need: for pair (i,j), inter_C[i] == inter_matrix[i,j] AND inter_C[j] == inter_matrix[i,j]
        # inter_C[i] is the i-th element of inter_C vector
        # So: inter_C[i] == inter_matrix[i,j] and inter_C[j] == inter_matrix[i,j]
        # Let's build:
        # cond1[i,j] = (inter_C[i] == inter_matrix[i,j])  -> inter_C[i] broadcast
        # cond2[i,j] = (inter_C[j] == inter_matrix[i,j])  -> inter_C[j] broadcast
        
        # inter_C is shape (n,), inter_matrix is shape (n,n)
        # cond1: for each i, inter_C[i] compared with inter_matrix[i, :]
        cond1 = inter_matrix == inter_C[:, np.newaxis]  # cond1[i,j] = (inter_matrix[i,j] == inter_C[i])
        cond2 = inter_matrix == inter_C[np.newaxis, :]   # cond2[i,j] = (inter_matrix[i,j] == inter_C[j])
        valid = cond1 & cond2  # valid[i,j] = (inter_C[i] == inter_matrix[i,j]) & (inter_C[j] == inter_matrix[i,j])
        # For a sunflower, we need valid[i,j] to be True for some i != j
        # Check if any valid[i,j] is True for i != j
        valid_upper = valid & tri_mask
        if np.any(valid_upper):
            return False
        return True
    
    # Greedy initialization with random ordering
    best_family = []
    best_size = 0
    
    for trial in range(15):
        if time.time() > deadline - 15:
            break
        order = list(range(n_total))
        random.shuffle(order)
        family_indices = []
        for idx in order:
            if time.time() > deadline - 15:
                break
            if can_add(all_masks[idx], family_indices):
                family_indices.append(idx)
        
        if len(family_indices) > best_size:
            best_size = len(family_indices)
            best_family = family_indices[:]
    
    # Local search: try swaps
    family_indices = best_family[:]
    family_set = set(family_indices)
    
    if time.time() < deadline - 5:
        # Try to add any set
        added = True
        while added and time.time() < deadline - 5:
            added = False
            for idx in range(n_total):
                if idx not in family_set:
                    if can_add(all_masks[idx], family_indices):
                        family_indices.append(idx)
                        family_set.add(idx)
                        added = True
        
        # Swap-based local search
        T = 1.0
        cooling = 0.9995
        while time.time() < deadline - 2 and T > 0.01:
            if family_indices:
                idx_out = random.randint(0, len(family_indices)-1)
                mask_out = family_indices[idx_out]
                
                # Find a random mask not in family
                idx_in = random.randint(0, n_total - 1)
                attempts = 0
                while idx_in in family_set and attempts < 50:
                    idx_in = random.randint(0, n_total - 1)
                    attempts += 1
                if idx_in in family_set:
                    continue
                
                # Check if idx_in can be added to family without mask_out
                fam_temp = [x for k, x in enumerate(family_indices) if k != idx_out]
                if can_add(all_masks[idx_in], fam_temp):
                    new_family = fam_temp + [idx_in]
                    family_indices = new_family
                    family_set.remove(mask_out)
                    family_set.add(idx_in)
                
                # Try pure addition
                if idx_in not in family_set:
                    if can_add(all_masks[idx_in], family_indices):
                        family_indices.append(idx_in)
                        family_set.add(idx_in)
            
            T *= cooling
        
        # Final attempt to add
        added = True
        while added and time.time() < deadline - 1:
            added = False
            for idx in range(n_total):
                if idx not in family_set:
                    if can_add(all_masks[idx], family_indices):
                        family_indices.append(idx)
                        family_set.add(idx)
                        added = True
    
    # Convert back to sorted lists
    result = []
    for idx in family_indices:
        mask = all_masks[idx]
        elements = [i+1 for i in range(16) if mask & (1 << i)]
        result.append(elements)
    
    return result
