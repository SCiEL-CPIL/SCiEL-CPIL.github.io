import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # 1. Generate all masks
    all_masks = []
    mask_to_idx = {}
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
        mask_to_idx[mask] = len(all_masks) - 1
    
    N = len(all_masks) # 1820
    
    # 2. Precompute conflicts
    # conf_dict[(i, j)] = list of k such that i, j, k form sunflower
    conf_dict = {}
    
    # Helper to get conflict list for pair i, j
    def get_conflicts(i, j):
        key = (i, j) if i < j else (j, i)
        if key in conf_dict:
            return conf_dict[key]
        
        A = all_masks[i]
        B = all_masks[j]
        I = A & B
        k = bin(I).count("1")
        
        # Elements in A or B but not in I
        D = (A | B) & ~I
        # Elements available for X: not in D, must contain I
        # X = I union Y, where Y is subset of R of size 4-k
        # R = [16] \ (D union I) = [16] \ (A union B)
        # Wait.
        # X \cap A = I and X \cap B = I.
        # This implies X \cap (A \cup B) = I.
        # So X \cap ((A \cup B) \setminus I) = empty.
        # Let D = (A \cup B) \setminus I.
        # X must be subset of [16] \ D.
        # And X must contain I.
        # Let R = ([16] \ D) \setminus I.
        # R is exactly [16] \ (A \cup B).
        # Because A \cup B = I \cup D.
        # So R is the complement of A \cup B in [16].
        
        R_mask = ((1 << 16) - 1) & ~(A | B)
        # Number of bits in R_mask
        # We need to choose 4-k elements from R_mask.
        
        # Iterate combinations of indices in R_mask
        r_indices = [idx for idx in range(16) if (R_mask >> idx) & 1]
        
        conflicts = []
        need = 4 - k
        if need < 0 or need > len(r_indices):
            return []
        
        for comb in combinations(r_indices, need):
            # Construct mask
            X = I
            for idx in comb:
                X |= (1 << idx)
            if X in mask_to_idx:
                k_idx = mask_to_idx[X]
                if k_idx != i and k_idx != j:
                    conflicts.append(k_idx)
        
        conf_dict[key] = conflicts
        return conflicts

    # Precompute all pairs?
    # 1.65M pairs.
    # Each call to get_conflicts does some work.
    # Let's precompute to speed up the greedy loop.
    if time.time() < deadline - 10:
        for i in range(N):
            for j in range(i + 1, N):
                get_conflicts(i, j)
    
    # 3. Greedy Construction
    best_family = []
    best_size = 0
    
    def greedy_run(seed):
        random.seed(seed)
        F = []
        Forbidden = set()
        # We will iterate through candidates.
        # To make it efficient, we keep a list of available candidates.
        available = list(range(N))
        random.shuffle(available)
        
        while available:
            if time.time() > deadline:
                return F
            
            # Find a valid candidate
            # Heuristic: Pick one that adds fewest new conflicts
            # For speed, we might just pick the first valid one, 
            # or evaluate a small sample.
            
            valid_candidates = []
            # Check first 50 available? Or all?
            # If we check all, it's O(N * |F| * conflict_size)
            # Let's try to pick the best from a sample of 20.
            
            sample_size = min(20, len(available))
            sample_indices = random.sample(range(len(available)), sample_size)
            
            best_cand_idx_in_sample = -1
            min_new_conflicts = float('inf')
            best_cand_val = None
            
            for idx_in_sample in sample_indices:
                cand_idx = available[idx_in_sample]
                if cand_idx in Forbidden:
                    continue
                
                # Calculate new conflicts
                new_conf = set()
                for a in F:
                    key = (a, cand_idx) if a < cand_idx else (cand_idx, a)
                    if key in conf_dict:
                        new_conf.update(conf_dict[key])
                
                # Subtract already forbidden
                truly_new = new_conf - Forbidden
                count = len(truly_new)
                
                if count < min_new_conflicts:
                    min_new_conflicts = count
                    best_cand_val = cand_idx
                    best_cand_idx_in_sample = idx_in_sample
            
            if best_cand_val is None:
                # No valid candidate in sample. 
                # Check if any exist in remaining?
                # For now, break or scan rest?
                # Scanning rest is expensive. Let's break.
                # Or try to find any valid one.
                found = False
                for idx in range(len(available)):
                    cand_idx = available[idx]
                    if cand_idx in Forbidden:
                        continue
                    # Found one
                    # Add it
                    F.append(cand_idx)
                    # Update Forbidden
                    for a in F[:-1]:
                        key = (a, cand_idx) if a < cand_idx else (cand_idx, a)
                        if key in conf_dict:
                            Forbidden.update(conf_dict[key])
                    available.pop(idx)
                    found = True
                    break
                
                if not found:
                    break
            else:
                # Add best candidate
                cand_idx = best_cand_val
                F.append(cand_idx)
                # Update Forbidden
                for a in F[:-1]:
                    key = (a, cand_idx) if a < cand_idx else (cand_idx, a)
                    if key in conf_dict:
                        Forbidden.update(conf_dict[key])
                
                # Remove from available
                # Efficient removal?
                # available is a list. pop by index.
                available.pop(best_cand_idx_in_sample)
        
        return F

    # Run multiple seeds
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        family = greedy_run(seed)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
    
    # Convert back to lists
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
