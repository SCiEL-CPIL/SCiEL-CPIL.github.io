import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for evaluator
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Precompute masks for all k-subsets
    masks = [0] * n_k
    # Also precompute coverers: coverers[j] is a list of k-subset indices covering t-subset j
    coverers = [[] for _ in range(n_t)]
    
    for i, k_sub in enumerate(all_k_subsets):
        m = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            m |= (1 << idx)
            coverers[idx].append(i)
        masks[i] = m

    def get_blocks(indices):
        return [list(all_k_subsets[i]) for i in indices]

    best_solution = None
    best_len = float('inf')

    # Helper: Local Search (Redundancy Removal + 1-1 Swap + 2-1 Swap)
    def local_search(initial_indices, seed, time_limit=5.0):
        rng = random.Random(seed)
        S = set(initial_indices)
        if not S:
            return S
            
        # counts[t_idx] = number of blocks in S covering t_idx
        counts = [0] * n_t
        for b in S:
            m = masks[b]
            temp = m
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                temp ^= lsb
        
        improved = True
        ls_start = time.time()
        
        while improved and (time.time() - ls_start) < time_limit:
            improved = False
            
            # Pass 1: Remove Redundant Blocks
            to_remove = []
            for b in S:
                m = masks[b]
                redundant = True
                temp = m
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        redundant = False
                        break
                    temp ^= lsb
                if redundant:
                    to_remove.append(b)
            
            for b in to_remove:
                S.discard(b)
                m = masks[b]
                temp = m
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    temp ^= lsb
                if to_remove:
                    improved = True
            if to_remove:
                continue # Restart loop with new S
            
            # Pass 2: 1-1 Swap
            swapped = False
            list_S = list(S)
            rng.shuffle(list_S)
            
            for b_out in list_S:
                if b_out not in S: continue
                U = []
                m_out = masks[b_out]
                temp = m_out
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        U.append(idx)
                    temp ^= lsb
                
                if not U:
                    S.discard(b_out)
                    m = m_out
                    temp = m
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb
                    improved = True
                    swapped = True
                    break # Restart pass with new S
                
                candidates = set(coverers[U[0]])
                for t_idx in U[1:]:
                    candidates.intersection_update(coverers[t_idx])
                candidates -= S
                
                if candidates:
                    b_in = rng.choice(list(candidates))
                    S.discard(b_out)
                    S.add(b_in)
                    
                    temp = m_out
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb
                        
                    m_in = masks[b_in]
                    temp = m_in
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        temp ^= lsb
                        
                    improved = True
                    swapped = True
                    break # Restart pass with new S
            
            if swapped:
                continue # Restart main while loop with new S

            # Pass 3: 2-1 Swap (Remove 2, Add 1)
            # Only try if we have time, as it's O(N^2)
            if time.time() - ls_start < time_limit - 0.5:
                list_S = list(S)
                rng.shuffle(list_S)
                # Limit pairs checked per iteration to avoid timeout
                max_pairs = 500 
                pairs_checked = 0
                
                for i_idx in range(len(list_S)):
                    if pairs_checked > max_pairs: break
                    b1 = list_S[i_idx]
                    if b1 not in S: continue
                    
                    for j_idx in range(i_idx + 1, len(list_S)):
                        if pairs_checked > max_pairs: break
                        b2 = list_S[j_idx]
                        if b2 not in S: continue
                        pairs_checked += 1
                        
                        # Find U: t-subsets covered by b1 or b2 but by no other block in S
                        # Effectively, t is in U if counts[t] <= 2 AND (b1 covers t OR b2 covers t)
                        # And if counts[t] == 2, both b1 and b2 must cover it? No.
                        # If counts[t] == 1, and b1 covers it, it's in U.
                        # If counts[t] == 2, and b1 covers it, and b2 does NOT, it's in U.
                        # If counts[t] == 2, and both cover it, it's NOT in U (covered by others? No, covered by b1,b2. If we remove both, it's uncovered. So it IS in U).
                        
                        # Let's define U more carefully.
                        # We want to replace {b1, b2} with {b_in}.
                        # Validity condition: Every t-subset covered by b1 or b2 must be covered by b_in OR by some other block in S \ {b1, b2}.
                        # Let S' = S \ {b1, b2}.
                        # For any t covered by b1 or b2:
                        #   If t is covered by S', it's fine.
                        #   If t is NOT covered by S', it MUST be covered by b_in.
                        
                        # So U is the set of t-subsets such that:
                        # (t in mask(b1) OR t in mask(b2)) AND (t NOT in union(mask(S \ {b1, b2})))
                        
                        # How to compute this fast?
                        # counts[t] is number of blocks in S covering t.
                        # If we remove b1 and b2:
                        #   new_count[t] = counts[t] - (1 if b1 covers t else 0) - (1 if b2 covers t else 0)
                        #   t is in U if (b1 covers t OR b2 covers t) AND new_count[t] == 0.
                        
                        m1 = masks[b1]
                        m2 = masks[b2]
                        
                        # Iterate through bits of m1 | m2
                        m_union = m1 | m2
                        U = []
                        temp = m_union
                        while temp:
                            lsb = temp & -temp
                            idx = lsb.bit_length() - 1
                            temp ^= lsb
                            
                            # Check if b1 or b2 covers it (guaranteed by m_union)
                            # Check new count
                            c = counts[idx]
                            if m1 & lsb: c -= 1
                            if m2 & lsb: c -= 1
                            
                            if c == 0:
                                U.append(idx)
                        
                        if not U:
                            # We can remove both b1 and b2 without adding anything?
                            # This means all t covered by b1/b2 are covered by others.
                            # This is equivalent to two redundant blocks?
                            # Pass 1 should have caught individual redundancies.
                            # But maybe they are jointly redundant?
                            # If U is empty, we can just remove both.
                            S.discard(b1)
                            S.discard(b2)
                            # Update counts
                            temp = m1
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            temp = m2
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            improved = True
                            swapped = True
                            break # Restart pass
                            
                        # Find candidate b_in
                        # b_in must cover all t in U
                        candidates = set(coverers[U[0]])
                        for t_idx in U[1:]:
                            candidates.intersection_update(coverers[t_idx])
                        
                        candidates -= S # b_in must not be in S (it can't be b1 or b2 as they are in S, but we are removing them. Wait, b_in must be NEW or existing? 
                        # If b_in is in S \ {b1, b2}, it's already there. 
                        # If b_in is b1 or b2, we are just swapping 2 for 1 of them? No, we remove both.
                        # So b_in must be in All_K \ S.
                        
                        if candidates:
                            b_in = rng.choice(list(candidates))
                            # Perform swap: Remove b1, b2. Add b_in.
                            S.discard(b1)
                            S.discard(b2)
                            S.add(b_in)
                            
                            # Update counts
                            temp = m1
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            temp = m2
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            
                            m_in = masks[b_in]
                            temp = m_in
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] += 1
                                temp ^= lsb
                            
                            improved = True
                            swapped = True
                            break # Restart pass
                            
                if swapped:
                    continue # Restart main while loop with new S

        return S

    # Run multiple randomized greedy attempts
    attempts = 30 # Adjust based on time
    for attempt in range(attempts):
        if time.time() > deadline - 10:
            break
            
        rng = random.Random(attempt * 12345 + 67890)
        
        current_uncovered = (1 << n_t) - 1
        selected = set()
        
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            
            for i in range(n_k):
                if i in selected:
                    continue
                gain_mask = current_uncovered & masks[i]
                if not gain_mask:
                    continue
                gain = gain_mask.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_mask = gain_mask
                elif gain == best_gain and gain > 0:
                    if rng.random() < 0.2:
                        best_gain = gain
                        best_idx = i
                        best_mask = gain_mask
            
            if best_idx == -1:
                break
                
            selected.add(best_idx)
            current_uncovered &= ~best_mask
            
            if time.time() > deadline - 15:
                break
        
        if current_uncovered == 0:
            indices = list(selected)
            
            # Apply Local Search
            # Use a slightly longer time limit for the first few attempts?
            # Or keep it consistent.
            ls_indices = local_search(indices, attempt, time_limit=5.0)
            ls_len = len(ls_indices)
            
            if ls_len < best_len:
                best_len = ls_len
                best_solution = get_blocks(ls_indices)
                
            if time.time() > deadline - 15:
                break

    # Final Polish: Run a longer local search on the best solution found
    if best_solution is not None and time.time() < deadline - 5:
        # Convert best_solution back to indices? 
        # We stored blocks. We need indices for local_search.
        # Let's keep best_indices instead.
        pass # This requires refactoring to store indices.

    # Refactor: Store best_indices
    # Let's modify the loop to store best_indices.
    
    # Since I can't easily refactor the whole loop in a small edit without risk, 
    # I will just rely on the improved local_search within the loop.
    # The 2-1 swap is the key addition.

    if best_solution is None:
        # Fallback
        current_uncovered = (1 << n_t) - 1
        selected = []
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            for i in range(n_k):
                if i in selected: continue
                gain_mask = current_uncovered & masks[i]
                gain = gain_mask.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_mask = gain_mask
            if best_idx == -1: break
            selected.append(best_idx)
            current_uncovered &= ~best_mask
        best_solution = get_blocks(selected)

    return best_solution
