import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    masks = [0] * n_k_subsets
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # 4. Greedy Set Cover
    uncovered = all_mask
    selected_indices = []
    used_blocks = [False] * n_k_subsets
    
    # Precompute list of blocks for faster access? No, just iterate.
    while uncovered:
        best_gain = -1
        best_idx = -1
        # Optimization: If we have many blocks, this loop is the bottleneck.
        # But N=75k is manageable.
        for i in range(n_k_subsets):
            if used_blocks[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~masks[best_idx]
        
    # 5. Local Search with Frequency Counts
    # Maintain frequency of coverage for each t-subset
    # freq[j] = number of selected blocks covering t-subset j
    
    def get_freq(block_list):
        freq = [0] * n_t_subsets
        for idx in block_list:
            m = masks[idx]
            # Iterate set bits
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                freq[bit_idx] += 1
                m ^= lsb
        return freq

    def is_redundant(idx, freq):
        # A block is redundant if every t-subset it covers is covered by at least one other block
        # i.e., freq[t] > 1 for all t in masks[idx]
        m = masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            if freq[bit_idx] == 1:
                return False
            m ^= lsb
        return True

    def find_replacement_for_gap(gap_mask, exclude_set):
        # Find a block (not in exclude_set) that covers all bits in gap_mask
        # gap_mask is a bitmask of t-subsets
        if gap_mask == 0:
            return -1 # No gap, no replacement needed (just remove)
        
        # We need a block C such that (masks[C] & gap_mask) == gap_mask
        # i.e., masks[C] is a superset of gap_mask
        
        # Optimization: If gap_mask is large, few blocks will cover it.
        # If gap_mask is small, many might.
        
        # Simple check: Iterate all blocks? Too slow if done for many pairs.
        # Better: Precompute for each t-subset the list of blocks covering it?
        # Memory: 11628 * ~3000 = 35M ints. ~280MB. Might be okay if we use arrays.
        # But we are in a tight loop.
        
        # Alternative: Only try this for small gaps.
        # If gap_mask.bit_count() > 10, it's unlikely a single block covers it (max C(8,5)=56).
        # Actually, a block covers C(8,5)=56 t-subsets.
        # If gap has > 56 bits, impossible.
        if gap_mask.bit_count() > 56:
            return -1
            
        # Check all blocks?
        # To speed up, we can check if any block covers the *rarest* t-subset in the gap.
        # Let's just iterate all blocks for now, but limit the search or use a heuristic.
        # Given N=75k, iterating 75k blocks is ~1ms.
        # If we do this for O(N^2) pairs, it's too slow.
        # We should only check pairs that *might* work.
        
        # Heuristic: Only check pairs where the union of their unique contributions is small.
        
        for i in range(n_k_subsets):
            if i in exclude_set:
                continue
            if (masks[i] & gap_mask) == gap_mask:
                return i
        return -1

    # Local Search Loop
    # We will iterate until no improvement is found.
    # Improvement types:
    # 1. Remove redundant block (1-for-0)
    # 2. Replace 2 blocks with 1 block (2-for-1)
    
    improved = True
    while improved:
        improved = False
        freq = get_freq(selected_indices)
        
        # Pass 1: Remove redundant blocks
        for i in range(len(selected_indices) - 1, -1, -1):
            idx = selected_indices[i]
            if is_redundant(idx, freq):
                # Update freq
                m = masks[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    freq[bit_idx] -= 1
                    m ^= lsb
                selected_indices.pop(i)
                improved = True
                # Restart freq calculation? Or just continue?
                # Since we removed one, freq is updated.
                # But other blocks might become redundant.
                # Continue loop with updated freq.
        
        if improved:
            continue # Restart the whole local search loop to catch cascading redundancies
        
        # Pass 2: 2-for-1 replacement
        # To avoid O(N^2) * O(N) complexity, we sample pairs or use a heuristic.
        # N is ~300. N^2 = 90,000.
        # For each pair, checking replacement is O(N) = 75,000.
        # Total 6.7 * 10^9 ops. Too slow.
        
        # Optimization for Pass 2:
        # Only consider pairs (A, B) where the "gap" created by removing them is small.
        # Gap for (A, B) consists of t-subsets covered by A or B but not by others.
        # Specifically:
        # t is in Gap if:
        #   (freq[t] == 1 and (A covers t or B covers t)) OR
        #   (freq[t] == 2 and A covers t and B covers t)
        
        # We can precompute for each block A, the set of t-subsets where freq[t]==1 and A is the sole coverer.
        # Let unique[A] = { t | freq[t]==1 and A covers t }
        # Then Gap(A,B) = unique[A] U unique[B] U { t | freq[t]==2 and A covers t and B covers t }
        
        # This is still complex.
        # Let's just try a limited number of random pairs per iteration.
        
        n_selected = len(selected_indices)
        if n_selected < 2:
            break
            
        # Try 1000 random pairs
        for _ in range(1000):
            i, j = random.sample(range(n_selected), 2)
            idx_a = selected_indices[i]
            idx_b = selected_indices[j]
            
            # Calculate Gap
            # t is in gap if removing A and B uncovers it.
            # This happens if freq[t] was 1 (and A or B covered it) or freq[t] was 2 (and A and B covered it).
            
            gap_mask = 0
            
            # Iterate over t-subsets covered by A
            m_a = masks[idx_a]
            while m_a:
                lsb = m_a & -m_a
                bit_idx = lsb.bit_length() - 1
                if freq[bit_idx] == 1:
                    # Only A covers it (since B is also being removed, but if freq was 1, B doesn't cover it)
                    # Wait, if freq[bit_idx] == 1, then exactly one block covers it.
                    # If A covers it, it's the one. B does not cover it.
                    # So removing A uncovers it.
                    gap_mask |= lsb
                elif freq[bit_idx] == 2:
                    # Two blocks cover it. One is A. Is the other B?
                    if masks[idx_b] & lsb:
                        gap_mask |= lsb
                m_a ^= lsb
                
            # Iterate over t-subsets covered by B (that were not already counted via A? No, need to be careful)
            # Actually, the condition is:
            # t is uncovered after removal iff (freq[t] == 1 and (A covers t or B covers t)) or (freq[t] == 2 and A covers t and B covers t)
            
            # My loop above handled:
            # Case freq==1 and A covers t: Added to gap.
            # Case freq==2 and A covers t and B covers t: Added to gap.
            
            # What if freq==1 and B covers t (but A doesn't)?
            # Then A's loop didn't see it.
            # We need to check B's unique parts too.
            
            m_b = masks[idx_b]
            while m_b:
                lsb = m_b & -m_b
                bit_idx = lsb.bit_length() - 1
                # We only add if it wasn't already added by A's loop?
                # If freq==1 and B covers t, A doesn't cover t. So A's loop didn't add it.
                if freq[bit_idx] == 1:
                    gap_mask |= lsb
                # If freq==2 and B covers t and A covers t, it was already added in A's loop.
                # We don't need to add it again (bitwise OR is idempotent).
                m_b ^= lsb
                
            if gap_mask == 0:
                # Redundant pair! Remove both.
                # Update freq
                m_a = masks[idx_a]
                while m_a:
                    lsb = m_a & -m_a
                    bit_idx = lsb.bit_length() - 1
                    freq[bit_idx] -= 1
                    m_a ^= lsb
                m_b = masks[idx_b]
                while m_b:
                    lsb = m_b & -m_b
                    bit_idx = lsb.bit_length() - 1
                    freq[bit_idx] -= 1
                    m_b ^= lsb
                # Remove from selected_indices (handle indices carefully)
                # Remove j first if j > i, or just pop and adjust
                if i < j:
                    selected_indices.pop(j)
                    selected_indices.pop(i)
                else:
                    selected_indices.pop(i)
                    selected_indices.pop(j)
                improved = True
                break # Restart loop
                
            else:
                # Try to find a replacement block C
                # Exclude A and B
                exclude = {idx_a, idx_b}
                c_idx = find_replacement_for_gap(gap_mask, exclude)
                if c_idx != -1:
                    # Replace A, B with C
                    # Update freq for A and B (decrement)
                    m_a = masks[idx_a]
                    while m_a:
                        lsb = m_a & -m_a
                        bit_idx = lsb.bit_length() - 1
                        freq[bit_idx] -= 1
                        m_a ^= lsb
                    m_b = masks[idx_b]
                    while m_b:
                        lsb = m_b & -m_b
                        bit_idx = lsb.bit_length() - 1
                        freq[bit_idx] -= 1
                        m_b ^= lsb
                    # Update freq for C (increment)
                    m_c = masks[c_idx]
                    while m_c:
                        lsb = m_c & -m_c
                        bit_idx = lsb.bit_length() - 1
                        freq[bit_idx] += 1
                        m_c ^= lsb
                        
                    # Update selected_indices
                    if i < j:
                        selected_indices.pop(j)
                        selected_indices[i] = c_idx
                    else:
                        selected_indices.pop(i)
                        selected_indices[j] = c_idx
                    improved = True
                    break # Restart loop

    return [list(k_subsets[i]) for i in selected_indices]
