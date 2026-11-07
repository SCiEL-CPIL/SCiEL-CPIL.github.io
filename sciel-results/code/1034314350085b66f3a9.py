import numpy as np
import time
import random

def construct():
    n = 12
    m = 40
    total_subsets = 1 << n  # 4096
    deadline = time.time() + 85.0  # Safety margin (90s limit)
    
    # Precompute subset sizes
    sizes = np.array([bin(i).count('1') for i in range(total_subsets)], dtype=np.int32)
    
    # Build the strict subset relation matrix M
    # M[i, j] = 1 if i is strict subset of j
    M = np.zeros((total_subsets, total_subsets), dtype=np.int8)
    
    chunk_size = 512
    for start in range(0, total_subsets, chunk_size):
        end = min(start + chunk_size, total_subsets)
        i_chunk = np.array(range(start, end), dtype=np.int32)
        j_all = np.array(range(total_subsets), dtype=np.int32)
        
        i_expanded = i_chunk[:, None]
        j_expanded = j_all[None, :]
        
        is_subset = (i_expanded & j_expanded) == i_expanded
        not_equal = i_expanded != j_expanded
        M[start:end, :] = (is_subset & not_equal).astype(np.int8)
    
    # Comparability matrix: C[i, j] = 1 if i and j are comparable
    # C[i, j] = M[i,j] + M[j,i] (but since M is not symmetric, we need both)
    # Actually, C[i,j] = 1 if M[i,j]==1 or M[j,i]==1
    # Let's compute C directly
    C = np.zeros((total_subsets, total_subsets), dtype=np.int8)
    for start in range(0, total_subsets, chunk_size):
        end = min(start + chunk_size, total_subsets)
        i_chunk = np.array(range(start, end), dtype=np.int32)
        j_all = np.array(range(total_subsets), dtype=np.int32)
        
        i_expanded = i_chunk[:, None]
        j_expanded = j_all[None, :]
        
        # M[i,j] is already computed
        # M[j,i] means j is strict subset of i, i.e., (j & i) == j and j != i
        j_i_subset = (j_expanded & i_expanded) == j_expanded
        not_equal2 = j_expanded != i_expanded
        M_ji = (j_i_subset & not_equal2).astype(np.int8)
        
        C[start:end, :] = ((M[start:end, :] | M_ji)).astype(np.int8)
    
    # Now C[i,j] = 1 if i and j are comparable (one is strict subset of other)
    # C is symmetric
    
    # Score of a family = sum over unordered pairs (i,j) of C[family[i], family[j]]
    
    def compute_score(family):
        """Compute the number of comparable pairs in family."""
        if len(family) < 2:
            return 0
        fam_arr = np.array(family, dtype=np.int32)
        # For each pair (i, j) with i < j, check C[fam[i], fam[j]]
        score = 0
        for i in range(len(fam_arr)):
            for j in range(i + 1, len(fam_arr)):
                if C[fam_arr[i], fam_arr[j]] == 1:
                    score += 1
        return score
    
    # Vectorized score computation
    def compute_score_fast(family):
        """Faster score computation using vectorization."""
        if len(family) < 2:
            return 0
        fam_arr = np.array(family, dtype=np.int32)
        # Create all pairs
        i_idx, j_idx = np.triu_indices(len(fam_arr), k=1)
        pairs = C[fam_arr[i_idx], fam_arr[j_idx]]
        return int(np.sum(pairs))
    
    # Greedy construction with incremental score tracking
    def greedy_construct(init_family, deadline):
        """
        Start with init_family, greedily add sets to maximize comparable pairs.
        Returns the final family and its score.
        """
        family = list(init_family)
        used = set(family)
        
        # Current score
        current_score = compute_score_fast(family)
        
        # For each candidate, track its comparability count with current family
        # comp_count[c] = number of sets in family that are comparable with c
        
        # Precompute for all candidates
        all_candidates = np.array(list(set(range(total_subsets)) - used), dtype=np.int32)
        
        # comp_count for each candidate
        comp_counts = np.zeros(total_subsets, dtype=np.int32)
        
        # Initialize comp_counts for current family
        for c in all_candidates:
            comp_counts[c] = np.sum(C[np.array(family, dtype=np.int32), c])
        
        while len(family) < m:
            if time.time() > deadline - 5.0:
                break
            
            if len(all_candidates) == 0:
                break
            
            # Find the candidate with highest comp_count
            best_idx = np.argmax(comp_counts[all_candidates])
            best_mask = int(all_candidates[best_idx])
            
            family.append(best_mask)
            used.add(best_mask)
            current_score += int(comp_counts[best_mask])
            
            # Remove best_mask from candidates
            all_candidates = np.array(list(set(all_candidates) - {best_mask}), dtype=np.int32)
            
            # Update comp_counts for remaining candidates
            # For each remaining candidate r, comp_count[r] += C[r, best_mask]
            if len(all_candidates) > 0:
                comp_counts[all_candidates] += C[all_candidates, best_mask]
        
        return family, current_score
    
    # Local search polish: try swapping elements
    def local_search_polish(family, deadline, max_iters=200):
        """
        Local search: try swapping each element with a non-family element.
        Accept if it improves the score.
        """
        family = list(family)
        used = set(family)
        current_score = compute_score_fast(family)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            fam_arr = np.array(family, dtype=np.int32)
            
            # Try each position
            for pos in range(len(family)):
                if time.time() > deadline - 5.0:
                    break
                
                removed = family[pos]
                
                # Score without this element
                # New score = current_score - (number of other family members comparable with removed)
                # That's current_score - sum_{j != pos} C[fam[j], removed]
                
                # Compute contribution of removed to score
                # It's the number of other elements comparable with it
                contrib = int(np.sum(C[np.delete(fam_arr, pos), removed]))
                
                # Score if we remove it
                score_without = current_score - contrib
                
                # Now try adding each non-family element
                # For each candidate c not in used, new_score = score_without + (number of family members (excluding pos) comparable with c)
                
                # Get all non-family candidates
                non_family = np.array(list(set(range(total_subsets)) - used), dtype=np.int32)
                
                if len(non_family) == 0:
                    continue
                
                # For each candidate, compute how many of the remaining family members are comparable with it
                # Remaining family = family with pos removed
                remaining_fam = np.delete(fam_arr, pos)
                
                # comp_with_remaining[c] = sum_{r in remaining_fam} C[c, r]
                comp_with_remaining = np.zeros(len(non_family), dtype=np.int32)
                
                # Vectorized: for each candidate, sum C[candidate, r] for r in remaining_fam
                # This is C[non_family[:, None], remaining_fam[None, :]].sum(axis=1)
                # But this could be large. Let's do it in chunks.
                
                chunk = 256
                for start in range(0, len(non_family), chunk):
                    end = min(start + chunk, len(non_family))
                    chunk_arr = non_family[start:end]
                    comp_with_remaining[start:end] = C[chunk_arr[:, None], remaining_fam[None, :]].sum(axis=1)
                
                new_scores = score_without + comp_with_remaining
                best_new_idx = np.argmax(new_scores)
                best_new_mask = int(non_family[best_new_idx])
                best_new_score = int(new_scores[best_new_idx])
                
                if best_new_score > current_score:
                    # Accept the swap
                    family[pos] = best_new_mask
                    used.discard(removed)
                    used.add(best_new_mask)
                    current_score = best_new_score
                    improved = True
                    break  # Restart the outer loop after a swap
            
            if not improved:
                break
        
        return family, current_score
    
    # Run multiple restarts
    best_score = -1
    best_family = None
    
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        random.seed(restart * 7 + 13)
        
        # Different initialization strategies
        if restart % 3 == 0:
            # Start with empty set and full set, then add some structured sets
            init = [0, (1 << n) - 1]
            # Add singletons
            for b in range(n):
                init.append(1 << b)
            # Add some pairs
            for b in range(n):
                for c in range(b+1, n):
                    if len(init) >= 10:
                        break
                    init.append((1 << b) | (1 << c))
        elif restart % 3 == 1:
            # Start with a random chain
            init = []
            current = 0
            init.append(current)
            for _ in range(8):
                available_bits = [b for b in range(n) if not (current & (1 << b))]
                if not available_bits:
                    break
                current |= (1 << random.choice(available_bits))
                init.append(current)
        else:
            # Random initialization
            init = random.sample(range(total_subsets), 8)
        
        # Ensure distinct
        init = list(dict.fromkeys(init))
        
        # Greedy construction
        family, score = greedy_construct(init, deadline)
        
        if len(family) < m:
            # Fill with random remaining
            used = set(family)
            remaining = [x for x in range(total_subsets) if x not in used]
            random.shuffle(remaining)
            for x in remaining:
                if len(family) >= m:
                    break
                family.append(x)
        
        if len(family) < m:
            continue
        
        # Local search polish
        family, score = local_search_polish(family, deadline, max_iters=150)
        
        if score > best_score:
            best_score = score
            best_family = list(family)
        
        # If we've done well, reduce polish for subsequent restarts
        if best_score > 130:
            max_polish = 50
        else:
            max_polish = 150
    
    # Final polish on the best family if time allows
    if best_family is not None and time.time() < deadline - 5.0:
        for _ in range(5):
            if time.time() > deadline - 3.0:
                break
            polished, pol_score = local_search_polish(best_family, deadline, max_iters=100)
            if pol_score > best_score:
                best_score = pol_score
                best_family = polished
    
    # Convert to lists of integers
    if best_family is None:
        best_family = list(range(m))
    
    result = []
    for mask in best_family:
        mask = int(mask)
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result
