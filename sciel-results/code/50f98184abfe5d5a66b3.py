import itertools
import time
import random
import numpy as np

def _precompute(v, k, t):
    """Precompute all t-subsets, all k-blocks, and coverage masks."""
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    block_to_idx = {b: i for i, b in enumerate(all_kblocks)}
    
    # Precompute coverage masks for each block
    # block_mask[i] is an integer where bit j is 1 if block i covers tset j
    block_masks = [0] * num_kblocks
    
    for b_idx, block in enumerate(all_kblocks):
        mask = 0
        # Iterate over all t-subsets of this block
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            mask |= (1 << ts_idx)
        block_masks[b_idx] = mask
        
    # Full coverage mask
    full_mask = (1 << num_tsets) - 1
    
    return all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask

def _greedy_cover(v, k, t, precomp, seed=0, deadline=None):
    """Greedy set-cover using bitmasks for speed."""
    if deadline is None:
        deadline = time.time() + 100
        
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask = precomp
    
    rng = random.Random(seed)
    
    covered_mask = 0
    is_selected = [False] * num_kblocks
    sol = []
    
    # Initial gain: number of bits in mask
    gains = [bin(m).count('1') for m in block_masks]
    
    while True:
        if time.time() > deadline:
            return None
            
        if covered_mask == full_mask:
            break
            
        # Find block with max gain
        max_gain = -1
        best_idx = -1
        # Optimization: We can just iterate, or use heap. 
        # With 75k blocks, linear scan is ~75k ops. 
        # We can cache gains and update only affected blocks.
        # But for simplicity and speed in Python, let's try a heap or just scan.
        # Scanning 75k integers is fast.
        
        # Actually, we can maintain a list of (gain, idx) and update.
        # But updating gains is expensive if we do it for all blocks.
        # Let's just scan. 75k * 300 iterations = 22.5M ops. Might be tight in Python.
        
        # Better: Use a heap.
        # But gains change.
        # Let's stick to the previous vectorized approach or optimize the scan.
        # The previous program used numpy. Let's use numpy for the greedy part to be safe.
        
        pass # Fallback to numpy greedy if this is too slow, but let's try to make bitmask greedy work.
        # Actually, the previous numpy greedy was 1.1189. 
        # Let's use the numpy greedy from the current program as the initializer, 
        # and use the GA for refinement.
        
    return sol # This function is a stub, we will use the numpy one or a simplified bitmask one.

def _greedy_cover_numpy(v, k, t, precomp, seed=0, deadline=None):
    """Vectorized greedy using numpy, adapted from current program."""
    if deadline is None:
        deadline = time.time() + 100
        
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    rng = random.Random(seed)
    
    gain = np.full(num_kblocks, len(block_to_tsets[0]), dtype=np.int32)
    is_uncovered = np.full(num_tsets, True, dtype=bool)
    is_selected = np.full(num_kblocks, False, dtype=bool)
    sol = []
    
    while True:
        if time.time() > deadline:
            return None
            
        max_gain = int(np.max(gain))
        if max_gain <= 0:
            break
            
        candidates = np.where((gain == max_gain) & (~is_selected))[0]
        if len(candidates) == 0:
            break
            
        b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
        
        sol.append(list(all_kblocks[b_idx]))
        is_selected[b_idx] = True
        gain[b_idx] = -1
        
        bts = block_to_tsets_np[b_idx]
        newly_covered_mask = is_uncovered[bts]
        newly_covered = bts[newly_covered_mask]
        
        if len(newly_covered) == 0:
            continue
            
        is_uncovered[newly_covered] = False
        
        blocks_to_decrement = []
        for ts_idx in newly_covered:
            blocks_to_decrement.extend(tset_to_blocks[ts_idx])
            
        if blocks_to_decrement:
            counts = np.bincount(blocks_to_decrement, minlength=num_kblocks)
            gain -= counts
            gain[is_selected] = -1000
            
    return sol

def _reduce_blocks(blocks, precomp, deadline):
    """Remove redundant blocks from a valid solution."""
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask = precomp
    
    blocks = list(blocks)
    
    # Map blocks to indices
    block_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx == -1:
            # Should not happen if blocks are from all_kblocks
            return blocks
        block_indices.append(idx)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(block_indices)
        for i in range(n):
            if time.time() > deadline:
                return blocks
            
            # Calculate mask without block i
            mask_without = 0
            for j in range(n):
                if j != i:
                    mask_without |= block_masks[block_indices[j]]
                    
            if mask_without == full_mask:
                # Block i is redundant
                block_indices.pop(i)
                changed = True
                break
                
    # Convert back to block lists
    return [list(all_kblocks[idx]) for idx in block_indices]

def _repair_solution(blocks, precomp, deadline):
    """Make a solution valid by adding blocks if necessary."""
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask = precomp
    
    blocks = list(blocks)
    block_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            block_indices.append(idx)
            
    covered_mask = 0
    for idx in block_indices:
        covered_mask |= block_masks[idx]
        
    if covered_mask == full_mask:
        return blocks
        
    # Greedy repair
    remaining_mask = full_mask & ~covered_mask
    is_selected = [False] * num_kblocks
    for idx in block_indices:
        is_selected[idx] = True
        
    while remaining_mask != 0 and time.time() < deadline:
        best_idx = -1
        max_gain = 0
        # Find block with max new coverage
        # We can scan all blocks
        for i in range(num_kblocks):
            if is_selected[i]:
                continue
            new_bits = block_masks[i] & remaining_mask
            gain = bin(new_bits).count('1')
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if best_idx == -1 or max_gain == 0:
            break # No improvement possible (should not happen if full_mask is reachable)
            
        block_indices.append(best_idx)
        is_selected[best_idx] = True
        covered_mask |= block_masks[best_idx]
        remaining_mask = full_mask & ~covered_mask
        
    return [list(all_kblocks[idx]) for idx in block_indices]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Precompute
    precomp = _precompute(v, k, t)
    
    # Phase 1: Greedy Initialization
    # Use the numpy greedy for speed
    # Note: _precompute in this file doesn't return the numpy arrays needed by _greedy_cover_numpy
    # We need to adjust _precompute to return them or implement a bitmask greedy.
    # Let's implement a bitmask greedy to be self-contained and fast.
    
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask = precomp
    
    def bitmask_greedy(seed=0, deadline=None):
        if deadline is None:
            deadline = start_time + 30.0
            
        rng = random.Random(seed)
        covered_mask = 0
        is_selected = [False] * num_kblocks
        sol_indices = []
        
        # Precompute gains
        gains = [bin(m).count('1') for m in block_masks]
        
        while True:
            if time.time() > deadline:
                return None
                
            if covered_mask == full_mask:
                break
                
            max_gain = -1
            best_idx = -1
            # Scan for max gain
            # Optimization: Only scan unselected
            for i in range(num_kblocks):
                if is_selected[i]:
                    continue
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    best_idx = i
                    
            if best_idx == -1 or max_gain <= 0:
                break
                
            sol_indices.append(best_idx)
            is_selected[best_idx] = True
            gains[best_idx] = 0
            
            # Update gains
            # For each block j, gain[j] -= (block_masks[j] & block_masks[best_idx] & ~covered_mask)
            # This is O(N) per step. N=75k. Steps ~300. 22M ops. Might be slow in Python.
            # Let's use a heuristic: just re-calculate gain for all blocks?
            # Or use numpy?
            
            # Let's try to optimize the update.
            # new_bits = block_masks[best_idx] & ~covered_mask
            # for j in range(num_kblocks):
            #     if not is_selected[j]:
            #         gains[j] -= bin(block_masks[j] & new_bits).count('1')
            
            # This is 75k * 300 = 22.5M bit counts. 
            # Python bit count is fast, but 22M iterations might take ~2-5 seconds.
            # Acceptable.
            
            new_bits = block_masks[best_idx] & ~covered_mask
            for j in range(num_kblocks):
                if not is_selected[j]:
                    gains[j] -= bin(block_masks[j] & new_bits).count('1')
                    
            covered_mask |= block_masks[best_idx]
            
        return [list(all_kblocks[idx]) for idx in sol_indices]
        
    best_sol = bitmask_greedy(seed=0, deadline=start_time + 40.0)
    if best_sol is None:
        # Fallback
        best_sol = [list(all_kblocks[0])]
        
    best_len = len(best_sol)
    
    # Try a few more greedy seeds
    for i in range(1, 5):
        if time.time() > start_time + 50.0:
            break
        sol = bitmask_greedy(seed=i, deadline=start_time + 50.0)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol
            
    # Phase 2: Genetic Algorithm
    # Population of valid solutions
    pop = [best_sol]
    
    # Add some variations
    for _ in range(4):
        if time.time() > start_time + 60.0:
            break
        # Mutate best_sol
        mutated = list(best_sol)
        # Remove a random block
        if mutated:
            mutated.pop(random.randint(0, len(mutated)-1))
        # Repair
        repaired = _repair_solution(mutated, precomp, start_time + 60.0)
        # Reduce
        reduced = _reduce_blocks(repaired, precomp, start_time + 60.0)
        pop.append(reduced)
        
    # Remove duplicates in pop
    unique_pop = []
    seen = set()
    for sol in pop:
        key = tuple(sorted(tuple(b) for b in sol))
        if key not in seen:
            seen.add(key)
            unique_pop.append(sol)
    pop = unique_pop
    
    rng = random.Random(42)
    
    while time.time() < start_time + 115.0:
        new_pop = []
        # Keep best
        pop.sort(key=lambda x: len(x))
        new_pop.append(pop[0])
        
        # Generate offspring
        while len(new_pop) < len(pop):
            if time.time() > start_time + 115.0:
                break
                
            # Selection: Tournament
            p1 = pop[rng.randint(0, len(pop)-1)]
            p2 = pop[rng.randint(0, len(pop)-1)]
            
            # Crossover: Uniform
            # Combine blocks
            child_blocks = list(p1) + list(p2)
            # Remove duplicates
            unique_blocks = []
            seen_blocks = set()
            for b in child_blocks:
                bt = tuple(b)
                if bt not in seen_blocks:
                    seen_blocks.add(bt)
                    unique_blocks.append(b)
            child = unique_blocks
            
            # Mutation: Random swap or removal
            r = rng.random()
            if r < 0.3 and child:
                # Remove random block
                child.pop(rng.randint(0, len(child)-1))
            elif r < 0.6 and child:
                # Swap random block
                idx = rng.randint(0, len(child)-1)
                # Pick a random new block
                new_block_idx = rng.randint(0, num_kblocks-1)
                child[idx] = list(all_kblocks[new_block_idx])
            # else: no mutation
            
            # Repair
            if time.time() < start_time + 115.0:
                child = _repair_solution(child, precomp, start_time + 115.0)
                
            # Reduce
            if time.time() < start_time + 115.0:
                child = _reduce_blocks(child, precomp, start_time + 115.0)
                
            new_pop.append(child)
            
        # Update best
        for sol in new_pop:
            if len(sol) < best_len:
                best_len = len(sol)
                best_sol = sol
                
        # Elitism: Keep top K
        new_pop.sort(key=lambda x: len(x))
        pop = new_pop[:len(pop)]
        
    return best_sol

# We need to ensure _precompute returns what _greedy_cover_numpy expects if we use it, 
# but we are using bitmask_greedy which uses the new precomp format.
# So we are good.
