import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115.0
    
    # Precomputation
    tsubs = list(itertools.combinations(range(v), t))
    n_tsub = len(tsubs)
    tsub_to_bit = {sub: i for i, sub in enumerate(tsubs)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    
    # Block masks
    block_masks = []
    for comb in all_k_subsets:
        mask = 0
        for sub in itertools.combinations(comb, t):
            mask |= (1 << tsub_to_bit[sub])
        block_masks.append(mask)
        
    full_mask = (1 << n_tsub) - 1
    
    # Index: frozenset of points -> list of block indices containing it
    # We need this for fast swap lookups
    point_set_to_blocks = defaultdict(list)
    for idx, block in enumerate(all_k_subsets):
        # Generate all subsets of this block
        # Only need subsets up to size k, but specifically we look up S_i which is union of t-subsets
        # S_i size is at most k.
        for r in range(1, k + 1):
            for sub in itertools.combinations(block, r):
                point_set_to_blocks[frozenset(sub)].append(idx)
                
    def greedy_construct(randomize=False):
        selected = []
        selected_set = set()
        uncovered = full_mask
        cover_count = [0] * n_tsub
        
        while uncovered and time.time() < deadline - 10.0:
            best_gain = 0
            best_candidates = []
            uncv = uncovered
            b_masks = block_masks
            
            for i in range(n_ksub):
                g = (b_masks[i] & uncv).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_candidates = [i]
                elif g == best_gain and g > 0:
                    best_candidates.append(i)
                    
            if best_gain == 0:
                break
                
            if randomize and len(best_candidates) > 1:
                # Pick randomly from top candidates
                # To avoid bias, maybe pick from top 5?
                # For now, just random from all max gain
                best_idx = random.choice(best_candidates)
            else:
                best_idx = best_candidates[0]
                
            selected.append(best_idx)
            selected_set.add(best_idx)
            
            mask = block_masks[best_idx]
            m = mask
            while m:
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                cover_count[bit_pos] += 1
                m ^= lsb
                
            uncovered &= ~mask
            
        return selected, cover_count

    def local_search(selected, cover_count):
        selected_set = set(selected)
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # 1. Remove redundant blocks
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                b_idx = selected[i]
                mask = block_masks[b_idx]
                m = mask
                has_unique = False
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    m ^= lsb
                    if cover_count[bit_pos] == 1:
                        has_unique = True
                        break
                if not has_unique:
                    # Remove
                    m = mask
                    while m:
                        lsb = m & (-m)
                        bit_pos = lsb.bit_length() - 1
                        m ^= lsb
                        cover_count[bit_pos] -= 1
                    selected.pop(i)
                    selected_set.remove(b_idx)
                    changed = True
            
            if not selected:
                break
                
            # 2. Try swaps
            for i in range(len(selected)):
                if time.time() > deadline - 2.0:
                    break
                b_idx = selected[i]
                mask = block_masks[b_idx]
                
                # Find unique t-subsets for this block
                unique_bits = []
                m = mask
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    m ^= lsb
                    if cover_count[bit_pos] == 1:
                        unique_bits.append(bit_pos)
                        
                if not unique_bits:
                    continue # Should have been removed, but maybe race condition?
                    
                # Compute S_i: union of points in unique t-subsets
                S_i = set()
                for bit in unique_bits:
                    S_i.update(tsubs[bit])
                    
                if len(S_i) > k:
                    continue
                    
                S_i_frozen = frozenset(S_i)
                candidates = point_set_to_blocks.get(S_i_frozen, [])
                
                swapped = False
                for cand_idx in candidates:
                    if cand_idx in selected_set:
                        continue
                    
                    # Perform swap
                    # Remove b_idx
                    m1 = mask
                    while m1:
                        lsb1 = m1 & (-m1)
                        bp1 = lsb1.bit_length() - 1
                        m1 ^= lsb1
                        cover_count[bp1] -= 1
                        
                    # Add cand_idx
                    m2 = block_masks[cand_idx]
                    while m2:
                        lsb2 = m2 & (-m2)
                        bp2 = lsb2.bit_length() - 1
                        m2 ^= lsb2
                        cover_count[bp2] += 1
                        
                    selected[i] = cand_idx
                    selected_set.remove(b_idx)
                    selected_set.add(cand_idx)
                    changed = True
                    swapped = True
                    break # Restart loop to re-evaluate redundancy
                
                if swapped:
                    break
        
        return selected, cover_count

    best_selected = None
    best_count = float('inf')
    
    # Run multiple iterations
    while time.time() < deadline - 5.0:
        # Alternate between deterministic and randomized greedy
        randomize = (time.time() % 2 > 1) or random.random() > 0.5
        selected, cover_count = greedy_construct(randomize)
        
        if not selected:
            continue
            
        selected, cover_count = local_search(selected, cover_count)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_selected = selected[:]
            
        # If we hit the lower bound or a very good number, we can stop early?
        # Lower bound is 131.
        if best_count <= 131:
            break

    if best_selected is None:
        # Fallback: return empty or greedy result if something went wrong
        selected, _ = greedy_construct(False)
        return [list(all_k_subsets[idx]) for idx in selected]
        
    return [list(all_k_subsets[idx]) for idx in best_selected]
