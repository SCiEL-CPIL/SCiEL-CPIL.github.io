import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Precompute for each block, which t-subsets it covers as a bitmask
    block_bitmasks = [0] * num_blocks
    block_to_tsubs = []
    for i, block in enumerate(k_subsets):
        mask = 0
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subsets_set[tsub]
            mask |= (1 << idx)
            covered.append(idx)
        block_bitmasks[i] = mask
        block_to_tsubs.append(covered)
        
    # Precompute for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
            
    # Convert to sets for faster membership testing
    tsub_to_blocks_set = [set(lst) for lst in tsub_to_blocks]
    
    def greedy_construct(randomize=False, temperature=1.0):
        """Greedy set cover with optional randomization."""
        uncovered_mask = (1 << num_tsubs) - 1
        selected = []
        
        while uncovered_mask != 0:
            if time.time() > deadline - 20:
                break
                
            # Compute gains for all blocks
            gains = []
            for b in range(num_blocks):
                gain = (block_bitmasks[b] & uncovered_mask).bit_count()
                gains.append(gain)
            
            if not any(g > 0 for g in gains):
                break
                
            if randomize:
                # Softmax sampling among top-k candidates
                max_gain = max(gains)
                if max_gain == 0:
                    break
                # Filter to blocks with positive gain
                candidates = [b for b in range(num_blocks) if gains[b] > 0]
                # Weight by gain^temperature
                weights = [gains[b] ** temperature for b in candidates]
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0
                chosen = candidates[0]
                for b, w in zip(candidates, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = b
                        break
            else:
                # Standard greedy: pick max gain
                best_gain = 0
                best_b = -1
                for b in range(num_blocks):
                    if gains[b] > best_gain:
                        best_gain = gains[b]
                        best_b = b
                if best_b == -1:
                    break
                chosen = best_b
            
            selected.append(chosen)
            uncovered_mask &= ~block_bitmasks[chosen]
        
        return selected, uncovered_mask
    
    def local_optimize(selected, max_iterations=100):
        """Aggressive local optimization: redundancy removal + swaps."""
        selected = list(selected)
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations and time.time() < deadline - 5:
            improved = False
            iter_count += 1
            
            # Step 1: Remove redundant blocks
            to_remove = []
            for b in selected:
                redundant = True
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)
            
            if to_remove:
                for b in to_remove:
                    selected.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # Step 2: Try single-block swaps (replace one block with another that covers its unique t-subsets)
            for i, b_out in enumerate(selected):
                if time.time() > deadline - 10:
                    break
                    
                # Find unique t-subsets covered only by b_out
                unique_tsubs = [s for s in block_to_tsubs[b_out] if cover_count[s] == 1]
                
                if not unique_tsubs:
                    # Fully redundant, should have been caught above
                    continue
                
                # Find candidate blocks that cover all unique t-subsets
                candidates = []
                for s in unique_tsubs:
                    candidates = tsub_to_blocks_set[s]
                    break
                if not unique_tsubs:
                    continue
                    
                first_s = unique_tsubs[0]
                candidate_set = tsub_to_blocks_set[first_s]
                
                valid_candidates = []
                for c in candidate_set:
                    if c == b_out:
                        continue
                    covers_all = True
                    for s in unique_tsubs[1:]:
                        if c not in tsub_to_blocks_set[s]:
                            covers_all = False
                            break
                    if covers_all:
                        valid_candidates.append(c)
                
                if valid_candidates:
                    # Pick one that minimizes overlap (prefer blocks that add few new t-subsets)
                    best_c = None
                    best_overlap = float('inf')
                    for c in valid_candidates:
                        # Count how many t-subsets in c are already covered by others
                        overlap = 0
                        for s in block_to_tsubs[c]:
                            if cover_count[s] > 1:
                                overlap += 1
                        if overlap < best_overlap:
                            best_overlap = overlap
                            best_c = c
                    
                    if best_c is not None:
                        # Perform swap
                        selected.remove(b_out)
                        selected.append(best_c)
                        for s in block_to_tsubs[b_out]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[best_c]:
                            cover_count[s] += 1
                        improved = True
                        break
        
        return selected
    
    def two_for_one_swap(selected, cover_count):
        """Try to replace 2 blocks with 1 block."""
        selected = list(selected)
        n = len(selected)
        
        for i in range(n):
            if time.time() > deadline - 15:
                return selected
            for j in range(i + 1, n):
                # Find t-subsets uniquely covered by block i or block j
                unique_to_pair = set()
                for s in block_to_tsubs[selected[i]]:
                    if cover_count[s] == 1:
                        unique_to_pair.add(s)
                for s in block_to_tsubs[selected[j]]:
                    if cover_count[s] == 1:
                        unique_to_pair.add(s)
                
                if not unique_to_pair:
                    continue
                
                # Find a block that covers all unique_to_pair
                first_s = next(iter(unique_to_pair))
                candidates = tsub_to_blocks_set[first_s]
                
                valid = []
                for c in candidates:
                    if c == selected[i] or c == selected[j]:
                        continue
                    covers_all = True
                    for s in unique_to_pair:
                        if c not in tsub_to_blocks_set[s]:
                            covers_all = False
                            break
                    if covers_all:
                        valid.append(c)
                
                if valid:
                    # Choose the one with best properties
                    best_c = valid[0]
                    # Remove both, add one
                    selected.remove(selected[i])
                    selected.remove(selected[j])
                    selected.append(best_c)
                    for s in block_to_tsubs[selected[i]]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[selected[j]]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[best_c]:
                        cover_count[s] += 1
                    return selected
        
        return selected
    
    # Multi-restart strategy
    best_result = None
    best_size = float('inf')
    
    # First, do a deterministic greedy
    initial_time = time.time()
    selected, uncovered = greedy_construct(randomize=False)
    if uncovered != 0:
        # Force completion
        while uncovered != 0 and time.time() < deadline - 30:
            best_gain = 0
            best_b = -1
            for b in range(num_blocks):
                gain = (block_bitmasks[b] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            if best_b == -1:
                break
            selected.append(best_b)
            uncovered &= ~block_bitmasks[best_b]
    
    # Local optimize
    optimized = local_optimize(selected)
    best_result = optimized
    best_size = len(optimized)
    
    # Now do randomized restarts with different temperatures
    remaining_time = deadline - 5 - time.time()
    if remaining_time > 10:
        # Budget for multiple restarts
        restarts_done = 1
        while time.time() < deadline - 10 and restarts_done < 10:
            restarts_done += 1
            temp = 0.5 + random.random() * 2.0  # Temperature between 0.5 and 2.5
            
            sel, unc = greedy_construct(randomize=True, temperature=temp)
            if unc != 0:
                # Force completion
                while unc != 0 and time.time() < deadline - 20:
                    best_gain = 0
                    best_b = -1
                    for b in range(num_blocks):
                        gain = (block_bitmasks[b] & unc).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                    if best_b == -1:
                        break
                    sel.append(best_b)
                    unc &= ~block_bitmasks[best_b]
            
            # Local optimize
            opt_sel = local_optimize(sel, max_iterations=50)
            
            if len(opt_sel) < best_size:
                best_size = len(opt_sel)
                best_result = opt_sel
    
    # Final local optimization pass on best result
    if time.time() < deadline - 5 and best_result is not None:
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in best_result:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        # Try 2-for-1 swaps
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            new_sel = two_for_one_swap(best_result, cover_count)
            if len(new_sel) < len(best_result):
                # Update cover_count
                cover_count = np.zeros(num_tsubs, dtype=np.int32)
                for b in new_sel:
                    for s in block_to_tsubs[b]:
                        cover_count[s] += 1
                best_result = new_sel
            else:
                break
        
        # Final redundancy removal
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            to_remove = []
            for b in best_result:
                redundant = True
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)
            if to_remove:
                for b in to_remove:
                    best_result.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
            else:
                break
    
    # Convert to actual subsets
    result = [list(k_subsets[i]) for i in best_result]
    
    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(blk)
        if key not in seen:
            seen.add(key)
            dedup.append(blk)
    
    return dedup
