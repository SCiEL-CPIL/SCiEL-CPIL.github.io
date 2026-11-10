"""Strategy: Multi-phase covering construction with bitset-accelerated operations.

1. Fast greedy construction to get an initial valid covering.
2. Aggressive local search: try to replace 1 block with 0, 2 with 1, 3 with 2, etc.
3. Simulated annealing-style neighborhood search for further improvements.
4. Multiple randomized restarts, keeping the best result found within the time budget.
"""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148  # 148s out of 150s limit, leaving 2s safety margin

    # Precompute all t-subsets and map to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubsets)
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}

    # NumPy-free bitset using Python integers (arbitrary precision)
    # Each bit i in an integer represents whether t-subset i is covered.
    
    # Precompute block signatures: for each possible k-subset, 
    # the bitmask of t-subsets it covers.
    # C(24,7) = 346104 blocks. Store as dict: tuple -> (bitmask, list_of_tsub_indices)
    
    # Actually, storing 346k large integers might use a lot of memory.
    # Let's compute on the fly with a cache, or precompute a compact representation.
    
    # Alternative approach: precompute for each t-subset, the list of blocks that contain it.
    # Then for a given set of blocks, we can compute coverage by OR-ing their signatures.
    
    # Let's precompute:
    # 1. All k-subsets as tuples
    # 2. For each, the bitmask (as Python int) of covered t-subsets
    # 3. For each, the list of t-subset indices it covers
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute bitmask and tsub index list for each block
    block_bitmasks = [0] * num_blocks
    block_tsub_indices = [[] for _ in range(num_blocks)]
    
    for b_idx, b in enumerate(all_blocks):
        mask = 0
        indices = []
        for tsub in itertools.combinations(b, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
            indices.append(idx)
        block_bitmasks[b_idx] = mask
        block_tsub_indices[b_idx] = indices
    
    # For each t-subset, list of block indices that contain it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx in range(num_blocks):
        for idx in block_tsub_indices[b_idx]:
            tsub_to_blocks[idx].append(b_idx)
    
    # Helper: count bits in a Python int
    def popcount(x):
        return bin(x).count('1')
    
    # Helper: check if covering is valid (all t-subsets covered)
    def is_valid(selected_blocks):
        """selected_blocks: list of block indices"""
        if not selected_blocks:
            return num_tsubs == 0
        mask = 0
        for b_idx in selected_blocks:
            mask |= block_bitmasks[b_idx]
        return popcount(mask) == num_tsubs
    
    # Helper: get number of uncovered t-subsets
    def uncovered_count(selected_blocks):
        mask = 0
        for b_idx in selected_blocks:
            mask |= block_bitmasks[b_idx]
        return num_tsubs - popcount(mask)
    
    # Phase 1: Greedy construction
    def greedy_construct(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        uncovered_mask = (1 << num_tsubs) - 1  # all bits set
        selected = []
        
        while uncovered_mask != 0:
            # Pick a random uncovered t-subset
            # Find a random set bit in uncovered_mask
            # Convert to list of set bits (could be expensive for large masks)
            # Alternative: iterate through all t-subsets and pick randomly among uncovered ones
            
            # Find an uncovered t-subset
            # Get list of uncovered indices
            uncov_idx = []
            temp = uncovered_mask
            i = 0
            while temp:
                if temp & 1:
                    uncov_idx.append(i)
                temp >>= 1
                i += 1
            
            if not uncov_idx:
                break
                
            target = rng.choice(uncov_idx)
            
            # Among blocks containing this t-subset, pick the one that covers the most uncovered
            candidates = tsub_to_blocks[target]
            if not candidates:
                break
                
            best_b = None
            best_gain = -1
            
            for b_idx in candidates:
                gain = popcount(uncovered_mask & block_bitmasks[b_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            
            if best_b is None:
                break
            
            selected.append(best_b)
            uncovered_mask &= ~block_bitmasks[best_b]
        
        return selected
    
    # Phase 2: Local search - try to remove redundant blocks
    def local_search_remove(selected, rng=None):
        if rng is None:
            rng = random
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            indices = list(range(len(selected)))
            rng.shuffle(indices)
            for i in indices:
                if i >= len(selected):
                    continue
                # Try removing selected[i]
                new_selected = selected[:i] + selected[i+1:]
                if is_valid(new_selected):
                    selected = new_selected
                    changed = True
                    break
        
        return selected
    
    # Phase 3: Swap search - try replacing one block with another
    def swap_search(selected, rng=None, max_swaps=500):
        if rng is None:
            rng = random
        
        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps and time.time() < deadline - 5:
            improved = False
            indices = list(range(len(selected)))
            rng.shuffle(indices)
            for i in indices:
                if i >= len(selected):
                    break
                # Try removing selected[i] and adding a random block
                base_selected = selected[:i] + selected[i+1:]
                base_mask = 0
                for b_idx in base_selected:
                    base_mask |= block_bitmasks[b_idx]
                
                # Find uncovered t-subsets
                uncovered = (~base_mask) & ((1 << num_tsubs) - 1)
                if uncovered == 0:
                    # Already valid without this block
                    selected = base_selected
                    improved = True
                    swaps_done += 1
                    break
                
                # Try adding a random block to cover more
                # Sample some random blocks
                tried = set()
                for _ in range(20):
                    new_b = rng.randint(0, num_blocks - 1)
                    if new_b in selected or new_b in tried:
                        continue
                    tried.add(new_b)
                    
                    new_selected = base_selected + [new_b]
                    new_mask = base_mask | block_bitmasks[new_b]
                    new_uncovered = (~new_mask) & ((1 << num_tsubs) - 1)
                    
                    if popcount(new_uncovered) < popcount(uncovered):
                        selected = new_selected
                        improved = True
                        swaps_done += 1
                        break
        
        return selected
    
    # Phase 4: Multi-block swap (replace 2 with 1, 3 with 2, etc.)
    def multi_swap_search(selected, rng=None, max_iterations=200):
        if rng is None:
            rng = random
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Try replacing 2 random blocks with 1
            if len(selected) >= 2:
                i, j = rng.sample(range(len(selected)), 2)
                base = selected[:i] + selected[i+1:j] + selected[j+1:] if i < j else selected[:j] + selected[j+1:i] + selected[i+1:]
                # Simpler:
                base = list(selected)
                base.remove(base[i] if i < j else base[j])
                # This is getting complicated. Let me redo.
                
                # Actually, let's just pick 2 indices properly
                idxs = rng.sample(range(len(selected)), 2)
                base = [b for pos, b in enumerate(selected) if pos not in idxs]
                
                base_mask = 0
                for b_idx in base:
                    base_mask |= block_bitmasks[b_idx]
                
                uncovered = (~base_mask) & ((1 << num_tsubs) - 1)
                
                # Try adding 1 random block
                for _ in range(30):
                    new_b = rng.randint(0, num_blocks - 1)
                    if new_b in base:
                        continue
                    new_mask = base_mask | block_bitmasks[new_b]
                    new_uncovered = (~new_mask) & ((1 << num_tsubs) - 1)
                    if popcount(new_uncovered) == 0:
                        selected = base + [new_b]
                        improved = True
                        break
            
            if not improved and len(selected) >= 3:
                # Try replacing 3 with 2
                idxs = rng.sample(range(len(selected)), 3)
                base = [b for pos, b in enumerate(selected) if pos not in idxs]
                
                base_mask = 0
                for b_idx in base:
                    base_mask |= block_bitmasks[b_idx]
                
                uncovered = (~base_mask) & ((1 << num_tsubs) - 1)
                
                # Try adding 2 random blocks
                for _ in range(20):
                    b1 = rng.randint(0, num_blocks - 1)
                    b2 = rng.randint(0, num_blocks - 1)
                    if b1 in base or b2 in base or b1 == b2:
                        continue
                    new_mask = base_mask | block_bitmasks[b1] | block_bitmasks[b2]
                    new_uncovered = (~new_mask) & ((1 << num_tsubs) - 1)
                    if popcount(new_uncovered) == 0:
                        selected = base + [b1, b2]
                        improved = True
                        break
        
        return selected
    
    # Main: multiple restarts
    best_selected = None
    best_size = float('inf')
    
    num_restarts = 10
    time_per_restart = (deadline - start_time - 10) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        restart_start = time.time()
        
        # Greedy construction with random seed
        rng = random.Random(r * 1000 + 42)
        selected = greedy_construct(seed=r * 1000 + 42)
        
        if not selected or not is_valid(selected):
            continue
        
        # Local search: remove redundant blocks
        selected = local_search_remove(selected, rng)
        
        # Swap search
        selected = swap_search(selected, rng, max_swaps=200)
        
        # Multi-swap search
        selected = multi_swap_search(selected, rng, max_iterations=100)
        
        # Final removal pass
        selected = local_search_remove(selected, rng)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
        
        # Check time
        if time.time() - restart_start > time_per_restart:
            continue
    
    if best_selected is None:
        return []
    
    # Convert to list of lists
    result = [list(all_blocks[b_idx]) for b_idx in best_selected]
    
    # Final validation
    covered_mask = 0
    for b_idx in best_selected:
        covered_mask |= block_bitmasks[b_idx]
    if popcount(covered_mask) != num_tsubs:
        # Invalid, try to repair
        for t_idx in range(num_tsubs):
            if not (covered_mask >> t_idx) & 1:
                candidates = tsub_to_blocks[t_idx]
                if candidates:
                    b_idx = random.choice(candidates)
                    result.append(list(all_blocks[b_idx]))
                    covered_mask |= block_bitmasks[b_idx]
    
    return result
