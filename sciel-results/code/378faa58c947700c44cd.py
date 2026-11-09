import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 110.0  # 10s safety margin
    
    points = list(range(v))
    
    # 1. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets and map to indices
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # 3. Precompute block -> t-subsets mapping
    # block_subs[i] is a list of t-subset indices covered by block i
    block_subs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_subs.append(subs)
        
    # 4. Precompute t-subset -> blocks mapping
    # sub_blocks[j] is a list of block indices that cover t-subset j
    sub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_subs):
        for si in subs:
            sub_blocks[si].append(i)

    def run_greedy(random_seed=None, noise_scale=0.0):
        if random_seed is not None:
            random.seed(random_seed)
            
        # gain[i] = number of currently uncovered t-subsets covered by block i
        gain = np.array([len(subs) for subs in block_subs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=bool)
        uncovered_count = N_t
        selected_blocks = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            # Find best block
            # To add randomization, we can pick randomly from the top few or add noise
            # Let's pick the max gain. If there are ties, np.argmax picks the first one.
            # To randomize, we can add small noise.
            if noise_scale > 0:
                # Add small noise to break ties and explore
                noise = np.random.uniform(0, noise_scale, size=N_blocks)
                scores = gain + noise
            else:
                scores = gain
                
            best_idx = np.argmax(scores)
            
            if gain[best_idx] == 0:
                # No more uncovered subsets can be covered? Should not happen if valid cover exists
                break
                
            selected_blocks.append(best_idx)
            gain[best_idx] = -1  # Disable this block
            
            # Update gains for all blocks that share subsets with best_idx
            for si in block_subs[best_idx]:
                if not is_covered[si]:
                    is_covered[si] = True
                    uncovered_count -= 1
                    # Update all blocks that cover this subset
                    for b_idx in sub_blocks[si]:
                        if gain[b_idx] > 0:
                            gain[b_idx] -= 1
                            
        return selected_blocks

    def try_reduce(block_indices):
        if not block_indices:
            return block_indices
            
        # Count coverage for each t-subset
        cov_count = np.zeros(N_t, dtype=np.int32)
        for b_idx in block_indices:
            for si in block_subs[b_idx]:
                cov_count[si] += 1
                
        # Try to remove blocks
        # We iterate through the list and try to remove each one
        # If removing it makes any cov_count[si] == 0, we cannot remove it
        # To be efficient, we can check if min(cov_count[si] for si in block_subs[b_idx]) > 1
        # But we need to update cov_count if we do remove it.
        
        # A simpler approach: 
        # Iterate through blocks. 
        # If a block is redundant (all its subsets are covered by >1 blocks), remove it.
        # We need to be careful: removing one block might make another redundant.
        # So we loop until no changes.
        
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            # Iterate in reverse to keep indices stable? Or just use a list and filter.
            # Let's create a new list of blocks to keep.
            new_blocks = []
            for b_idx in block_indices:
                # Check if b_idx is redundant
                redundant = True
                for si in block_subs[b_idx]:
                    if cov_count[si] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    changed = True
                    for si in block_subs[b_idx]:
                        cov_count[si] -= 1
                else:
                    new_blocks.append(b_idx)
            if changed:
                block_indices = new_blocks
                
        return block_indices

    def local_search(block_indices):
        """
        Try to improve the solution by replacing blocks.
        For each block in the solution, try to find a replacement that covers
        the same t-subsets (or a superset) while allowing removal of other blocks.
        
        Strategy: 
        1. Compute coverage counts.
        2. For each block b in the solution:
           - Temporarily remove b.
           - If any t-subset becomes uncovered, find a new block that covers all
             those uncovered t-subsets.
           - If such a block exists and is not already in the solution, consider
             swapping b for the new block.
           - Additionally, try to remove redundant blocks again.
        """
        if not block_indices:
            return block_indices
            
        # Convert to set for fast lookup
        block_set = set(block_indices)
        
        # Count coverage
        cov_count = np.zeros(N_t, dtype=np.int32)
        for b_idx in block_indices:
            for si in block_subs[b_idx]:
                cov_count[si] += 1
        
        # We will iterate through the list of blocks and try to improve
        # To avoid modifying the list while iterating, we work on a copy
        current_blocks = list(block_indices)
        
        improved = True
        while improved and time.time() < DEADLINE - 10:
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                # Check if we can replace b_idx with a better block
                # Or if b_idx is redundant
                # First, check redundancy
                redundant = True
                for si in block_subs[b_idx]:
                    if cov_count[si] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    for si in block_subs[b_idx]:
                        cov_count[si] -= 1
                    block_set.remove(b_idx)
                    improved = True
                    continue
                
                # Try to find a replacement
                # Identify which t-subsets are ONLY covered by b_idx
                # If there are none, it's redundant (already handled)
                # If there are some, we need a block that covers all of them.
                critical_subs = []
                for si in block_subs[b_idx]:
                    if cov_count[si] == 1:
                        critical_subs.append(si)
                
                if critical_subs:
                    # Find a block that covers all critical_subs
                    # And is not b_idx itself
                    # And ideally covers more uncovered subsets? No, we just need to maintain coverage.
                    # Actually, we want to replace b_idx with a block that covers critical_subs.
                    # If such a block exists and is different, we might swap.
                    # But swapping doesn't reduce count. It might enable further reductions.
                    
                    # For now, let's just try to find if there's a block that covers critical_subs
                    # and allows us to remove another block? That's complex.
                    
                    # Simpler heuristic:
                    # If we can find a block that covers critical_subs and has higher "efficiency"
                    # or allows removing another block, swap.
                    
                    # Let's try a simpler move: 
                    # If b_idx covers some subsets that are also covered by other blocks,
                    # maybe we can find a block that covers the unique parts of b_idx
                    # and is "better" in some way?
                    
                    # Let's stick to a specific move:
                    # Try to find a block that covers all critical_subs.
                    # If found, and it's not b_idx, and it allows removing another block?
                    # No, let's just try to swap if the new block covers more total subsets?
                    # No, that doesn't guarantee reduction.
                    
                    # Let's try a different approach:
                    # Pick a random block from the solution.
                    # Try to remove it. If it fails, find a block that covers the failed subsets.
                    # If found, add it. Then try to reduce again.
                    
                    # This is getting complicated for a simple local search.
                    # Let's just do:
                    # 1. Redundancy removal (already done above)
                    # 2. Try to swap a block with another block that covers its critical subsets
                    #    and potentially allows removing another block.
                    
                    # For now, let's just keep b_idx.
                    new_blocks.append(b_idx)
                else:
                    # No critical subsets? Then it's redundant.
                    # But we already checked redundancy. 
                    # Wait, if cov_count[si] > 1 for all si in b_idx, it's redundant.
                    # So if we reach here, it's not redundant.
                    new_blocks.append(b_idx)
            
            if improved:
                current_blocks = new_blocks
                # Recompute cov_count? 
                # We updated cov_count incrementally during removal.
                # But if we kept blocks, cov_count is still valid.
                # If we removed blocks, cov_count was decremented.
                # So cov_count is consistent with current_blocks.
                
        return current_blocks

    best_sol = None
    best_len = float('inf')
    
    # Run deterministic greedy first
    sol = run_greedy(random_seed=None)
    sol = try_reduce(sol)
    best_sol = sol
    best_len = len(sol)
    
    # Run randomized restarts
    attempt = 0
    while time.time() < DEADLINE - 10.0:
        # Vary noise scale
        noise_scale = 0.1 if attempt % 2 == 0 else 0.5
        sol = run_greedy(random_seed=attempt, noise_scale=noise_scale)
        
        # Reduce
        sol = try_reduce(sol)
        
        # Local search
        sol = local_search(sol)
        
        # Reduce again after local search
        sol = try_reduce(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol
            
        attempt += 1
        
    # Final reduction on best solution
    best_sol = try_reduce(best_sol)
    
    # Convert indices back to blocks
    return [all_blocks[i] for i in best_sol]
