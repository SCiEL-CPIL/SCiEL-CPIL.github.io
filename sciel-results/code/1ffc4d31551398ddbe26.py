import time
import itertools
import random

def _count_comparable_bitmasks(masks):
    """Count unordered pairs where one is a strict subset of the other, using bitmasks."""
    n = len(masks)
    count = 0
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            mj = masks[j]
            # Strict subset: A subset B and A != B
            # Since all masks are distinct, A subset B implies strict subset.
            if (mi & mj) == mi:
                count += 1
            elif (mi & mj) == mj:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Universe size 12
    n_vertices = 12
    all_masks = list(range(1 << n_vertices))
    
    # Initial chain: C_i = {0, ..., i-1}
    # Mask for C_i is (1 << i) - 1
    chain_masks = [(1 << i) - 1 for i in range(n_vertices + 1)]
    chain_set = set(chain_masks)
    
    # Candidates are all other masks
    candidates = [m for m in all_masks if m not in chain_set]
    
    # To speed up greedy, we can sort candidates initially by a heuristic
    # Heuristic: distance to the chain? Or just random order?
    # Let's use a simple heuristic: prefer sets that are close to the chain in size.
    # Actually, the gain calculation will handle this.
    
    # We will maintain a list of active candidates and their current gains.
    # Since N=4096 is small, we can keep all candidates and mark used ones.
    
    used = [False] * (1 << n_vertices)
    for m in chain_masks:
        used[m] = True
        
    selected_masks = list(chain_masks)
    current_count = _count_comparable_bitmasks(selected_masks)
    
    # Precompute popcount for efficiency? Not strictly necessary for subset check.
    
    # Initialize gains for all candidates against the current chain
    gains = [0] * (1 << n_vertices)
    for m in candidates:
        g = 0
        for cm in chain_masks:
            if (m & cm) == m:
                g += 1
            elif (m & cm) == cm:
                g += 1
        gains[m] = g
        
    # Greedy phase
    num_to_add = 40 - len(selected_masks)
    
    for _ in range(num_to_add):
        if time.time() > deadline:
            break
            
        best_mask = -1
        best_gain = -1
        
        # Find candidate with max gain
        # Optimization: iterate only over unused candidates
        for m in candidates:
            if not used[m]:
                if gains[m] > best_gain:
                    best_gain = gains[m]
                    best_mask = m
                    
        if best_mask == -1:
            break
            
        # Add best_mask to selected
        used[best_mask] = True
        selected_masks.append(best_mask)
        current_count += best_gain
        
        # Update gains for all other unused candidates
        bm = best_mask
        for m in candidates:
            if not used[m]:
                if (m & bm) == m:
                    gains[m] += 1
                elif (m & bm) == bm:
                    gains[m] += 1
                    
    # Local Search Phase (Swaps)
    # Try to swap a selected non-chain set with an unused candidate if it improves the count.
    # We will do a few passes.
    
    # Identify indices of non-chain sets in selected_masks
    # Chain sets are indices 0..12.
    
    def get_comparable_count(mask, selected_list):
        """Count comparable pairs between mask and selected_list, excluding mask itself if present."""
        cnt = 0
        for sm in selected_list:
            if sm == mask:
                continue
            if (mask & sm) == mask:
                cnt += 1
            elif (mask & sm) == sm:
                cnt += 1
        return cnt

    # To speed up swaps, we can precompute comparability matrix?
    # 40x40 matrix is small.
    
    # Let's do simple local search:
    # For each non-chain selected set S_i, try swapping with a candidate C_j.
    # Gain = (new pairs with C_j) - (old pairs with S_i)
    # Note: Pairs between other selected sets remain unchanged.
    
    non_chain_indices = list(range(13, len(selected_masks)))
    
    # Sort non-chain indices by their current contribution?
    # Contribution of S_i is number of comparable pairs it has with other selected sets.
    
    for _ in range(5): # 5 passes
        if time.time() > deadline:
            break
            
        improved = False
        # Shuffle non-chain indices to avoid bias
        random.shuffle(non_chain_indices)
        
        for idx in non_chain_indices:
            if time.time() > deadline:
                break
                
            S_i = selected_masks[idx]
            
            # Calculate current contribution of S_i
            # Count comparable pairs with other selected sets
            contrib_i = 0
            for j, S_j in enumerate(selected_masks):
                if i == j: continue
                if (S_i & S_j) == S_i:
                    contrib_i += 1
                elif (S_i & S_j) == S_j:
                    contrib_i += 1
            
            best_swap_gain = 0
            best_swap_mask = -1
            
            # Try swapping with any unused candidate
            # Optimization: Only consider candidates with potential
            # We can use the `gains` array? No, gains are stale after swaps.
            # But we can estimate.
            
            # To keep it fast, limit search to top K candidates by initial heuristic?
            # Or just iterate all unused. 4096 is small.
            
            for m in candidates:
                if not used[m]:
                    # Calculate contribution of m if it were in the set
                    # It replaces S_i.
                    # New pairs = comparable(m, S_j) for all j != i
                    # Old pairs = contrib_i
                    # Delta = new_pairs - contrib_i
                    
                    new_pairs = 0
                    for j, S_j in enumerate(selected_masks):
                        if j == idx:
                            continue
                        if (m & S_j) == m:
                            new_pairs += 1
                        elif (m & S_j) == S_j:
                            new_pairs += 1
                            
                    delta = new_pairs - contrib_i
                    if delta > best_swap_gain:
                        best_swap_gain = delta
                        best_swap_mask = m
                        
            if best_swap_gain > 0 and best_swap_mask != -1:
                # Perform swap
                used[S_i] = False
                used[best_swap_mask] = True
                selected_masks[idx] = best_swap_mask
                current_count += best_swap_gain
                improved = True
                # Update non_chain_indices? The index is still valid, but the value changed.
                # We can just continue.
                
        if not improved:
            break
            
    # Convert back to lists of ints
    result = []
    for m in selected_masks:
        s = []
        for i in range(n_vertices):
            if m & (1 << i):
                s.append(i)
        result.append(s)
        
    # Final validation
    assert len(result) == 40
    seen = set()
    for s in result:
        assert isinstance(s, list)
        for x in s:
            assert isinstance(x, int) and not isinstance(x, bool)
            assert 0 <= x <= 11
        fs = frozenset(s)
        assert fs not in seen
        seen.add(fs)
        
    return result
