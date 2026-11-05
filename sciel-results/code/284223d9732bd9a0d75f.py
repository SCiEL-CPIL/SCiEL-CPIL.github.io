import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin before 120s limit
    
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    num_t = len(all_t_subsets)
    t_subsets_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute bitmask for each t-subset for fast membership
    t_bitmasks = np.zeros(num_t, dtype=np.uint32)
    for i, s in enumerate(all_t_subsets):
        mask = 0
        for p in s:
            mask |= (1 << p)
        t_bitmasks[i] = mask
    
    # Precompute for each k-subset: list of t-subset indices it covers
    # We'll build a mapping from block (as sorted tuple) to list of t-indices
    def get_block_t_indices(block):
        """Get list of t-subset indices covered by a block."""
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_subsets_idx[ts])
        return indices
    
    def greedy_construct(seed):
        """Run greedy construction with a given seed."""
        random.seed(seed)
        uncovered = set(range(num_t))
        blocks = []
        
        # Precompute: for each t-subset index, the list of points NOT in it
        # This helps generate candidate blocks faster
        # Actually, let's just generate on the fly
        
        while uncovered and time.time() < deadline:
            # Pick a random uncovered t-subset
            s_idx = random.choice(list(uncovered))
            s = all_t_subsets[s_idx]
            s_set = set(s)
            remaining = [p for p in points if p not in s_set]
            
            best_block = None
            best_gain = -1
            
            # Generate all k-subsets containing s
            for combo in itertools.combinations(remaining, k - t):
                block = tuple(sorted(s + combo))
                # Calculate gain
                gain = 0
                for ts in itertools.combinations(block, t):
                    if t_subsets_idx[ts] in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update uncovered
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(t_subsets_idx[ts])
        
        return blocks
    
    def optimize_solution(blocks, start_time, deadline):
        """Optimize a solution via deletion and swaps."""
        # Build reference counts and block index mapping
        ref_count = np.zeros(num_t, dtype=np.int32)
        block_indices = {}
        block_set = set()
        
        for b in blocks:
            indices = get_block_t_indices(b)
            block_indices[b] = indices
            block_set.add(b)
            for idx in indices:
                ref_count[idx] += 1
        
        current_size = len(blocks)
        best_size = current_size
        best_blocks = list(blocks)
        
        def deletion_pass(blocks, ref_count, block_indices, block_set):
            """Remove redundant blocks. Returns (new_blocks, changed)."""
            new_blocks = []
            changed = False
            for b in blocks:
                indices = block_indices[b]
                # Tentatively remove
                for idx in indices:
                    ref_count[idx] -= 1
                
                if np.min(ref_count) > 0:
                    # Redundant
                    changed = True
                    del block_indices[b]
                    block_set.discard(b)
                else:
                    # Not redundant, restore
                    for idx in indices:
                        ref_count[idx] += 1
                    new_blocks.append(b)
            
            return new_blocks, changed
        
        def try_swap(blocks, ref_count, block_indices, block_set, i):
            """Try to swap block at index i with a better one. Returns (success, new_block)."""
            b_old = blocks[i]
            old_indices = block_indices[b_old]
            
            # Find critical subsets (count == 1)
            criticals = []
            for idx in old_indices:
                if ref_count[idx] == 1:
                    criticals.append(idx)
            
            if not criticals:
                return False, None
            
            # Union of points in all critical t-subsets
            u_set = set()
            for idx in criticals:
                u_set.update(all_t_subsets[idx])
            
            if len(u_set) > k:
                return False, None
            
            # Need k - len(u_set) more points from remaining
            remaining_points = [p for p in points if p not in u_set]
            if len(remaining_points) < k - len(u_set):
                return False, None
            
            # Try several random completions
            for attempt in range(20):
                if time.time() > deadline:
                    break
                    
                extra = random.sample(remaining_points, k - len(u_set))
                b_new = tuple(sorted(list(u_set) + extra))
                
                # Skip if same as old or already exists
                if b_new == b_old or b_new in block_set:
                    continue
                
                new_indices = get_block_t_indices(b_new)
                
                # Check validity: remove old, add new
                for idx in old_indices:
                    ref_count[idx] -= 1
                for idx in new_indices:
                    ref_count[idx] += 1
                
                if np.min(ref_count) > 0:
                    # Valid swap!
                    # Perform it
                    blocks[i] = b_new
                    block_indices[b_new] = new_indices
                    del block_indices[b_old]
                    block_set.discard(b_old)
                    block_set.add(b_new)
                    return True, b_new
                else:
                    # Revert
                    for idx in new_indices:
                        ref_count[idx] -= 1
                    for idx in old_indices:
                            ref_count[idx] += 1
            
            return False, None
        
        # Main optimization loop
        max_rounds = 50
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Deletion pass
            blocks, changed = deletion_pass(blocks, ref_count, block_indices, block_set)
            current_size = len(blocks)
            
            if current_size < best_size:
                best_size = current_size
                best_blocks = list(blocks)
            
            if len(blocks) == 0:
                break
            
            # Swap phase: try to improve each block
            num_swap_attempts = 0
            max_swap_attempts = min(100, len(blocks) * 5)
            
            while num_swap_attempts < max_swap_attempts and time.time() < deadline:
                i = random.randint(0, len(blocks) - 1)
                success, _ = try_swap(blocks, ref_count, block_indices, block_set, i)
                num_swap_attempts += 1
                
                if success:
                    # After a successful swap, do a quick deletion check
                    blocks, changed = deletion_pass(blocks, ref_count, block_indices, block_set)
                    current_size = len(blocks)
                    if current_size < best_size:
                        best_size = current_size
                        best_blocks = list(blocks)
            
            # If no improvement in this round, break
            if not changed and num_swap_attempts == 0:
                break
        
        return best_blocks
    
    # Try multiple restarts with different seeds
    best_overall = None
    best_size = float('inf')
    
    # First, run one greedy + optimize to get a baseline
    blocks = greedy_construct(42)
    if blocks and time.time() < deadline:
        optimized = optimize_solution(blocks, start_time, deadline)
        if optimized:
            best_overall = optimized
            best_size = len(optimized)
    
    # Try additional restarts if time permits
    for seed in [123, 456, 789, 1024, 2048, 4096, 8192, 16384]:
        if time.time() > deadline - 30:  # Leave 30s for final optimization
            break
        
        blocks = greedy_construct(seed)
        if not blocks:
            continue
        
        optimized = optimize_solution(blocks, start_time, deadline)
        if optimized and len(optimized) < best_size:
            best_overall = optimized
            best_size = len(optimized)
    
    # If we still have time, do more optimization on the best solution
    if best_overall is not None and time.time() < deadline - 10:
        final_optimized = optimize_solution(best_overall, start_time, deadline)
        if final_optimized and len(final_optimized) <= best_size:
            best_overall = final_optimized
    
    return best_overall if best_overall else []
