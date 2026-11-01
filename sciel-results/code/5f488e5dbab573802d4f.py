import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # 8s safety margin

    # Precompute all t-subsets and map them to indices
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)

    # Precompute all blocks and their subset indices
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build subset index arrays for each block
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
    
    # Build a lookup from block tuple to its index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute bitmask for each block (each block covers C(k,t) subsets)
    # Use Python integers as bitmasks for fast operations
    block_masks = []
    for subs in block_subsets:
        mask = 0
        for s in subs:
            mask |= (1 << s)
        block_masks.append(mask)
    
    # Build a mapping from point set (frozenset) to list of block indices containing it
    # This helps with swap operations
    pointset_to_blocks = {}
    for i, b in enumerate(all_blocks):
        fs = frozenset(b)
        if fs not in pointset_to_blocks:
            pointset_to_blocks[fs] = []
        pointset_to_blocks[fs].append(i)
    
    # Greedy Construction using vectorized counting
    cover_count = np.zeros(n_subsets, dtype=np.int32)
    selected_blocks = []
    uncovered_mask = (1 << n_subsets) - 1  # All subsets uncovered initially
    
    while uncovered_mask != 0 and time.time() < deadline:
        best_block_idx = -1
        best_count = 0
        
        # Vectorized: count how many uncovered subsets each block covers
        # For each block, count bits in (block_mask & uncovered_mask)
        for bi in range(n_blocks):
            m = block_masks[bi] & uncovered_mask
            cnt = bin(m).count('1')
            if cnt > best_count:
                best_count = cnt
                best_block_idx = bi
                if best_count == bin(uncovered_mask).count('1'):
                    break
        
        if best_block_idx < 0 or best_count == 0:
            break
            
        selected_blocks.append(best_block_idx)
        subs = block_subsets[best_block_idx]
        for s in subs:
            cover_count[s] += 1
            if cover_count[s] == 1:
                uncovered_mask &= ~(1 << s)
    
    # Local Search with bitmask operations
    # Work with list of block indices
    blocks = list(selected_blocks)
    
    def is_redundant(block_idx):
        mask = block_masks[block_idx]
        # Check if all subsets in this block are covered by at least one other block
        # i.e., for each subset s in block, cover_count[s] > 1
        subs = block_subsets[block_idx]
        for s in subs:
            if cover_count[s] <= 1:
                return False
        return True
    
    def remove_block(block_idx):
        nonlocal blocks
        subs = block_subsets[block_idx]
        for s in subs:
            cover_count[s] -= 1
        blocks.remove(block_idx)
    
    def add_block(block_idx):
        nonlocal blocks
        subs = block_subsets[block_idx]
        for s in subs:
            cover_count[s] += 1
        blocks.append(block_idx)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Reduction Phase: remove redundant blocks
        i = 0
        while i < len(blocks):
            if is_redundant(blocks[i]):
                remove_block(blocks[i])
                improved = True
            else:
                i += 1
        
        # 2. Swap Phase: try to replace one block with another
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
                
            B = blocks[i]
            subs_B = block_subsets[B]
            
            # Find critical subsets (covered only by B)
            critical = [s for s in subs_B if cover_count[s] == 1]
            
            if not critical:
                continue
                
            # Union of points in critical subsets
            U_B = set()
            for s in critical:
                U_B.update(all_subsets[s])
                
            if len(U_B) > k:
                continue
                
            # Find best replacement block B' containing U_B
            # Score: maximize number of subsets in B' with cover_count > 1
            best_B_prime = -1
            best_score = -1
            
            remaining_points = [p for p in range(v) if p not in U_B]
            num_to_choose = k - len(U_B)
            
            for candidate_extra in itertools.combinations(remaining_points, num_to_choose):
                B_prime_tuple = tuple(sorted(list(U_B) + list(candidate_extra)))
                
                if B_prime_tuple == all_blocks[B]:
                    continue
                    
                B_prime_idx = block_to_idx.get(B_prime_tuple)
                if B_prime_idx is None:
                    continue
                    
                subs_Bp = block_subsets[B_prime_idx]
                
                # Score: count subsets in B_prime with cover_count > 1
                score = 0
                for s in subs_Bp:
                    if cover_count[s] > 1:
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_B_prime = B_prime_idx
                    
            if best_B_prime >= 0:
                # Simulate replacement
                for s in subs_B:
                    cover_count[s] -= 1
                    
                subs_Bp = block_subsets[best_B_prime]
                for s in subs_Bp:
                    cover_count[s] += 1
                    
                blocks[i] = best_B_prime
                
                # Check if any block is now redundant
                redundant_found = False
                for j in range(len(blocks)):
                    if is_redundant(blocks[j]):
                        redundant_found = True
                        break
                        
                if redundant_found:
                    improved = True
                    break
                else:
                    # Revert
                    for s in subs_Bp:
                        cover_count[s] -= 1
                    for s in subs_B:
                        cover_count[s] += 1
                    blocks[i] = B
    
    # 3. 2-for-1 Swap Phase: try to replace two blocks with one
    if time.time() < deadline:
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(blocks)):
                if time.time() > deadline:
                    break
                    
                B1 = blocks[i]
                B2 = blocks[j]
                subs_B1 = block_subsets[B1]
                subs_B2 = block_subsets[B2]
                
                # Find subsets covered only by B1 or only by B2 (critical for the pair)
                critical_union = set()
                for s in subs_B1:
                    if cover_count[s] == 1:
                        critical_union.add(s)
                for s in subs_B2:
                    if cover_count[s] == 1:
                        critical_union.add(s)
                
                if not critical_union:
                    continue
                
                # Union of points in critical subsets
                U = set()
                for s in critical_union:
                    U.update(all_subsets[s])
                    
                if len(U) > k:
                    continue
                
                # Find a block B' containing U
                remaining_points = [p for p in range(v) if p not in U]
                num_to_choose = k - len(U)
                
                found_replacement = False
                for candidate_extra in itertools.combinations(remaining_points, num_to_choose):
                    B_prime_tuple = tuple(sorted(list(U) + list(candidate_extra)))
                    B_prime_idx = block_to_idx.get(B_prime_tuple)
                    if B_prime_idx is None:
                        continue
                    
                    # Check that B' covers all critical subsets
                    subs_Bp = block_subsets[B_prime_idx]
                    bp_set = set(subs_Bp)
                    if not critical_union.issubset(bp_set):
                        continue
                    
                    # Simulate: remove B1 and B2, add B'
                    for s in subs_B1:
                        cover_count[s] -= 1
                    for s in subs_B2:
                        cover_count[s] -= 1
                    for s in subs_Bp:
                        cover_count[s] += 1
                    
                    # Check validity: no subset should have count < 1
                    valid = True
                    for s in subs_B1:
                        if cover_count[s] < 1:
                            valid = False
                            break
                    if valid:
                        for s in subs_B2:
                            if cover_count[s] < 1:
                                valid = False
                                break
                    
                    if valid:
                        # Perform the swap
                        blocks.remove(B1)
                        blocks.remove(B2)
                        blocks.append(B_prime_idx)
                        improved = True
                        found_replacement = True
                        break
                    else:
                        # Revert
                        for s in subs_Bp:
                            cover_count[s] -= 1
                        for s in subs_B2:
                            cover_count[s] += 1
                        for s in subs_B1:
                            cover_count[s] += 1
                
                if found_replacement:
                    break
    
    # Convert block indices back to tuples
    return [list(all_blocks[bi]) for bi in blocks]
