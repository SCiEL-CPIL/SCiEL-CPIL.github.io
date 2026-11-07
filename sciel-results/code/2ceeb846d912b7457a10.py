# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Deadline guard: 90s limit, use 85s for safety
    start_time = time.time()
    max_time = 85.0
    
    # Precompute all masks as a numpy array
    all_masks_np = np.arange(total_masks, dtype=np.int64)
    
    # Helper to check comparability using numpy broadcasting
    # Returns a boolean matrix where comp[i, j] is True if masks[i] and masks[j] are comparable
    def get_comparability_matrix(masks_list):
        masks_np = np.array(masks_list, dtype=np.int64).reshape(-1, 1)
        inter = masks_np & masks_np.T
        sub1 = (inter == masks_np)
        sub2 = (inter == masks_np.T)
        comp = sub1 | sub2
        # Diagonal is always True (set is comparable to itself), but we care about distinct sets.
        # For counting pairs, we will use the upper triangle excluding diagonal.
        return comp

    def count_pairs(comp_matrix):
        # Sum upper triangle excluding diagonal
        # np.triu_indices(m, k=1)
        # But doing this every time is slow.
        # Total sum of comp matrix is sum of degrees.
        # Sum of degrees = 2 * (number of edges) + m (diagonal)
        # We want number of edges (unordered pairs of distinct comparable sets).
        # Let S = sum(comp_matrix).
        # Edges = (S - m) // 2
        # This works because comp is symmetric.
        
        # Wait, is comp symmetric?
        # (A & B) == A or (A & B) == B.
        # If A subset B, then (A&B)==A is true.
        # If B subset A, then (A&B)==B is true.
        # Yes, it is symmetric.
        
        # However, we must be careful. 
        # If we just sum the boolean matrix, we get the count of True values.
        # Diagonal is always True (m values).
        # Off-diagonal True values come in pairs (i,j) and (j,i).
        # So Total True = m + 2 * (Comparable Pairs).
        # Comparable Pairs = (Total True - m) // 2.
        
        return int((np.sum(comp_matrix) - m) // 2)

    def get_initial_selection():
        """
        Greedily construct a good initial set of masks.
        Strategy: Pick masks that have high potential to be comparable to many others.
        A simple heuristic: Start with a long chain, then add sets that are comparable to the most existing sets.
        """
        selected = []
        selected_set = set()
        
        # Start with a random long chain to ensure a strong base of comparable pairs
        perm = list(range(n))
        random.shuffle(perm)
        cur = 0
        chain = [0]
        for p in perm:
            cur |= (1 << p)
            chain.append(cur)
            
        # Add chain elements first
        for mask in chain:
            if mask not in selected_set:
                selected.append(mask)
                selected_set.add(mask)
        
        # Fill the rest greedily
        while len(selected) < m:
            best_mask = -1
            best_score = -1
            
            # Sample a subset of candidates to speed up initialization
            # Instead of scanning all 4096, we can be smarter.
            # But 4096 is small. Let's just scan all for the first few, then sample.
            
            if len(selected) < 20:
                candidates = [m for m in range(total_masks) if m not in selected_set]
            else:
                # Sample 200 random candidates
                candidates = random.sample(range(total_masks), 200)
                candidates = [c for c in candidates if c not in selected_set]
            
            for mask in candidates:
                score = 0
                for s in selected:
                    if (mask & s) == mask or (mask & s) == s:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = mask
            
            if best_mask != -1:
                selected.append(best_mask)
                selected_set.add(best_mask)
            else:
                # Fallback
                available = [x for x in range(total_masks) if x not in selected_set]
                if available:
                    mask = random.choice(available)
                    selected.append(mask)
                    selected_set.add(mask)
                else:
                    break
                
        return selected

    def optimize(masks_list):
        """
        Greedy refinement using NumPy vectorization for candidate evaluation.
        """
        current_masks = list(masks_list)
        comp_matrix = get_comparability_matrix(current_masks)
        current_score = count_pairs(comp_matrix)
        
        # Precompute degrees for each mask in the current set
        # degree[i] = number of other masks comparable to current_masks[i]
        # From comp_matrix, row i sum includes self (1). So degree[i] = sum(row i) - 1.
        degrees = np.sum(comp_matrix, axis=1) - 1
        
        max_passes = 100
        improved = True
        pass_count = 0
        
        while improved and pass_count < max_passes:
            improved = False
            pass_count += 1
            
            if time.time() - start_time > max_time - 2.0:
                break
            
            for i in range(m):
                old_mask = current_masks[i]
                old_degree = degrees[i]
                
                # Find best replacement
                # We want to maximize: new_degree - old_degree
                # new_degree = sum(1 for k != i if comparable(new_mask, current_masks[k]))
                
                # Vectorized calculation:
                # Create a mask of "others" (all indices except i)
                # Actually, we can just calculate comparability of ALL 4096 masks against the current set,
                # then subtract the self-comparability if the candidate is in the set?
                # No, the candidate must NOT be in the set.
                
                # Let's compute the "potential degree" for every mask in the universe against the current set.
                # This is expensive if done from scratch every time.
                # But we can update it incrementally?
                # Or just compute it for the current state.
                
                # Cost: 4096 * 40 comparisons. ~160k ops.
                # In NumPy:
                # others = [current_masks[k] for k in range(m) if k != i]
                # others_np = np.array(others, dtype=np.int64).reshape(-1, 1)
                # inter = all_masks_np & others_np.T  # (4096, 39)
                # sub1 = (inter == all_masks_np)
                # sub2 = (inter == others_np.T)
                # comp_all = sub1 | sub2
                # new_degrees = np.sum(comp_all, axis=1)
                
                # This is quite fast in NumPy.
                
                others = [current_masks[k] for k in range(m) if k != i]
                if not others:
                    continue
                    
                others_np = np.array(others, dtype=np.int64).reshape(-1, 1)
                
                # Broadcasting:
                # all_masks_np: (4096, 1)
                # others_np.T: (1, 39)
                # inter: (4096, 39)
                
                inter = all_masks_np & others_np.T
                sub1 = (inter == all_masks_np)
                sub2 = (inter == others_np.T)
                comp_all = sub1 | sub2
                
                # Sum over axis 1 to get degree of each candidate against the 'others'
                new_degrees = np.sum(comp_all, axis=1)
                
                # We need to exclude masks that are already in current_masks
                # And we need to find the max new_degree among valid candidates.
                
                # Create a boolean mask for invalid candidates (those in current_masks)
                # We can check if candidate is in current_masks by iterating?
                # Or use numpy isin?
                # np.isin(all_masks_np, current_masks)
                
                in_set = np.isin(all_masks_np, current_masks)
                
                # Set degrees of in-set masks to -inf (or a very low number)
                # But we also need to handle the case where the best candidate is the old mask itself?
                # No, we are replacing old_mask, so old_mask is in the set.
                # We want a NEW mask.
                
                # So, valid candidates are those where in_set is False.
                # We want max new_degrees[~in_set].
                
                # However, if we pick a mask that is in the set, we can't.
                # So:
                valid_new_degrees = new_degrees.copy()
                valid_new_degrees[in_set] = -1000 # Penalty
                
                best_idx = np.argmax(valid_new_degrees)
                best_new_degree = valid_new_degrees[best_idx]
                best_new_mask = all_masks_np[best_idx]
                
                # Check if improvement
                gain = best_new_degree - old_degree
                
                if gain > 0:
                    # Update current_masks
                    current_masks[i] = int(best_new_mask)
                    
                    # Update degrees
                    # This is the tricky part.
                    # We removed old_mask and added best_new_mask.
                    # We need to update the comparability with ALL other masks.
                    
                    # Let's just recompute the whole comp_matrix and degrees.
                    # It's O(m^2) = 1600 ops. Very fast.
                    comp_matrix = get_comparability_matrix(current_masks)
                    current_score = count_pairs(comp_matrix)
                    degrees = np.sum(comp_matrix, axis=1) - 1
                    
                    improved = True
        
        return current_masks, current_score

    best_score = -1
    best_masks = []
    
    # Run a few restarts
    for restart in range(20):
        if time.time() - start_time > max_time - 5.0:
            break
            
        init_masks = get_initial_selection()
        optimized_masks, score = optimize(init_masks)
        
        if score > best_score:
            best_score = score
            best_masks = optimized_masks

    # Fallback if no masks found (should not happen)
    if not best_masks:
        # Return a valid but poor result
        best_masks = list(range(m))
        best_score = 0

    # Convert best_masks to list of lists
    result = []
    for mask in best_masks:
        s = []
        for b in range(n):
            if mask & (1 << b):
                s.append(b)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
