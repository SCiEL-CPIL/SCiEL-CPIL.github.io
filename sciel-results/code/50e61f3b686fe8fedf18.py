import itertools
import random
import time
import sys
import numpy as np

# EVOLVE-BLOCK-START

def _greedy_initial(v, k, t, rng, time_limit):
    """
    Fast greedy set cover to get an initial upper bound.
    """
    deadline = time.time() + time_limit
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    
    # Map tset to index
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # We represent the set of uncovered t-sets as a boolean numpy array
    uncovered = np.ones(n_tsets, dtype=bool)
    
    blocks = []
    
    while np.any(uncovered) and time.time() < deadline:
        best_block = None
        best_gain = -1
        
        # Strategy: Pick a random uncovered t-set, and try to complete it to a block.
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
            
        rand_idx = uncovered_indices[rng.randint(0, len(uncovered_indices)-1)]
        seed_tset = all_tsets[rand_idx]
        
        # Generate candidates containing seed_tset
        remaining_points = [p for p in range(v) if p not in seed_tset]
        num_to_choose = k - t
        
        candidates = []
        if num_to_choose <= 4:
            for combo in itertools.combinations(remaining_points, num_to_choose):
                block = tuple(sorted(seed_tset + combo))
                candidates.append(block)
        else:
            # Sample
            for _ in range(200):
                combo = tuple(sorted(rng.sample(remaining_points, num_to_choose)))
                block = tuple(sorted(seed_tset + combo))
                candidates.append(block)
        
        # Evaluate gain for each candidate
        for block in candidates:
            # Count how many uncovered t-sets are in this block
            gain = 0
            for ts in itertools.combinations(block, t):
                idx = tset_to_idx[ts]
                if uncovered[idx]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_gain <= 0:
            break
            
        blocks.append(list(best_block))
        
        # Update uncovered
        for ts in itertools.combinations(best_block, t):
            idx = tset_to_idx[ts]
            uncovered[idx] = False
            
    return blocks

def _simulate_annealing(v, k, t, initial_blocks, time_limit, rng):
    """
    Simulated Annealing to minimize the number of blocks.
    Objective: Minimize (len(blocks) + PENALTY * uncovered_count)
    """
    deadline = time.time() + time_limit
    
    # Precompute all t-sets and their indices
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Precompute block masks for all possible blocks? 
    # C(24,7) = 346,104. This is large but manageable for a lookup table if we use a dict.
    # However, generating all 346k blocks and their masks takes time.
    # Let's generate a pool of "candidate blocks" on the fly or pre-compute a smaller set.
    # For SA, we need to propose new blocks. Random sampling is fine.
    
    # State
    current_blocks = [tuple(b) for b in initial_blocks]
    current_blocks_set = set(current_blocks)
    
    # Coverage count: how many times each t-set is covered
    coverage = np.zeros(n_tsets, dtype=np.int32)
    
    def update_coverage(blocks, add=True):
        sign = 1 if add else -1
        for b in blocks:
            for ts in itertools.combinations(b, t):
                idx = tset_to_idx[ts]
                coverage[idx] += sign * 1
                
    # Initialize coverage
    update_coverage(current_blocks, add=True)
    
    # Initial objective
    uncovered_count = np.sum(coverage == 0)
    current_obj = len(current_blocks) + 1000.0 * uncovered_count
    
    best_blocks = list(current_blocks)
    best_obj = current_obj
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.999 # Cooling rate
    
    # Precompute a list of all possible blocks? 
    # It's better to generate random blocks or blocks derived from uncovered sets.
    
    # Helper to get a random block
    def random_block():
        return tuple(sorted(rng.sample(range(v), k)))
    
    # Helper to get a block covering a specific t-set
    def block_covering_tset(ts):
        ts_set = set(ts)
        remaining = [p for p in range(v) if p not in ts_set]
        need = k - t
        extra = tuple(sorted(rng.sample(remaining, need)))
        return tuple(sorted(list(ts) + extra))
        
    steps = 0
    while time.time() < deadline and T > T_min:
        steps += 1
        # Propose a move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Remove a block
            if len(current_blocks) > 1:
                idx = rng.randint(0, len(current_blocks)-1)
                block_to_remove = current_blocks[idx]
                
                # Check validity impact
                # If we remove it, some t-sets might become uncovered.
                # We need to check if coverage[ts] > 1 for all ts in block.
                removable = True
                for ts in itertools.combinations(block_to_remove, t):
                    idx_ts = tset_to_idx[ts]
                    if coverage[idx_ts] == 1:
                        removable = False
                        break
                
                if removable:
                    # Apply move
                    current_blocks.pop(idx)
                    current_blocks_set.discard(block_to_remove)
                    update_coverage([block_to_remove], add=False)
                    
                    new_uncovered = np.sum(coverage == 0)
                    new_obj = len(current_blocks) + 1000.0 * new_uncovered
                    
                    delta = new_obj - current_obj
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        # Accept
                        current_obj = new_obj
                        if current_obj < best_obj:
                            best_obj = current_obj
                            best_blocks = list(current_blocks)
                    else:
                        # Reject: Undo
                        current_blocks.insert(idx, block_to_remove)
                        current_blocks_set.add(block_to_remove)
                        update_coverage([block_to_remove], add=True)
                else:
                    # Cannot remove, try adding? Or just skip?
                    # Let's try swapping instead if removal fails often.
                    pass
            else:
                # Cannot remove if only 1 block
                pass
                
        elif move_type < 0.8:
            # Add a block
            # Choose a block that covers an uncovered t-set if possible, else random
            uncovered_idx = np.where(coverage == 0)[0]
            if len(uncovered_idx) > 0:
                rand_uncov_idx = uncovered_idx[rng.randint(0, len(uncovered_idx)-1)]
                ts = all_tsets[rand_uncov_idx]
                new_block = block_covering_tset(ts)
            else:
                new_block = random_block()
                
            if new_block not in current_blocks_set:
                # Apply move
                current_blocks.append(new_block)
                current_blocks_set.add(new_block)
                update_coverage([new_block], add=True)
                
                new_uncovered = np.sum(coverage == 0)
                new_obj = len(current_blocks) + 1000.0 * new_uncovered
                
                delta = new_obj - current_obj
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    # Accept
                    current_obj = new_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_blocks = list(current_blocks)
                else:
                    # Reject: Undo
                    current_blocks.pop()
                    current_blocks_set.discard(new_block)
                    update_coverage([new_block], add=False)
                    
        else:
            # Swap: Remove one, Add one
            if len(current_blocks) > 1:
                idx = rng.randint(0, len(current_blocks)-1)
                old_block = current_blocks[idx]
                
                # Try to find a new block
                uncovered_idx = np.where(coverage == 0)[0]
                if len(uncovered_idx) > 0:
                    rand_uncov_idx = uncovered_idx[rng.randint(0, len(uncovered_idx)-1)]
                    ts = all_tsets[rand_uncov_idx]
                    new_block = block_covering_tset(ts)
                else:
                    new_block = random_block()
                
                if new_block not in current_blocks_set:
                    # Apply swap
                    current_blocks[idx] = new_block
                    current_blocks_set.discard(old_block)
                    current_blocks_set.add(new_block)
                    
                    update_coverage([old_block], add=False)
                    update_coverage([new_block], add=True)
                    
                    new_uncovered = np.sum(coverage == 0)
                    new_obj = len(current_blocks) + 1000.0 * new_uncovered
                    
                    delta = new_obj - current_obj
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_obj = new_obj
                        if current_obj < best_obj:
                            best_obj = current_obj
                            best_blocks = list(current_blocks)
                    else:
                        # Undo
                        current_blocks[idx] = old_block
                        current_blocks_set.discard(new_block)
                        current_blocks_set.add(old_block)
                        update_coverage([new_block], add=False)
                        update_coverage([old_block], add=True)
                        
        T *= alpha
        
    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    total_budget = 135.0 # Leave 15s margin for safety
    
    rng = random.Random(42)
    
    # 1. Get initial solution
    initial_blocks = _greedy_initial(v, k, t, rng, 5.0)
    
    best_overall = initial_blocks
    best_len = len(initial_blocks)
    
    # 2. Run Simulated Annealing with restarts
    # We don't know how many restarts we can afford, so we loop until time is up.
    restart_count = 0
    while time.time() < start_time + total_budget:
        # Allocate time for this SA run
        # As we get closer to the deadline, make runs shorter
        remaining_time = start_time + total_budget - time.time()
        if remaining_time < 2.0:
            break
            
        # Run SA for a fraction of remaining time, or a fixed chunk
        # Let's try chunks of 10-20 seconds
        chunk_time = min(15.0, remaining_time - 1.0)
        
        # Perturb initial state for diversity in restarts
        # Remove a few random blocks and re-greedy? Or just start from the best found so far.
        # Starting from best found is safer to maintain validity.
        # To escape local minima, we can randomly swap some blocks before SA.
        
        sa_input = [list(b) for b in best_overall]
        
        # Random perturbation
        for _ in range(5):
            if len(sa_input) > 2:
                idx = rng.randint(0, len(sa_input)-1)
                # Replace a block with a random one
                new_b = tuple(sorted(rng.sample(range(v), k)))
                sa_input[idx] = list(new_b)
                
        sa_result = _simulate_annealing(v, k, t, sa_input, chunk_time, rng)
        
        # Validate SA result
        # SA should maintain validity if we accept only valid moves or penalize invalid ones.
        # My implementation penalizes invalid ones, so it might return an invalid set if it gets "stuck" in a bad state?
        # No, the objective function includes penalty. But the "best_blocks" tracking only updates if obj improves.
        # However, if the initial state is valid (obj = len), and we only accept moves with delta < 0 or via Metropolis,
        # we might accept a move that increases uncovered count (worse obj) if T is high.
        # So best_blocks might be invalid if it was accepted as a "temporary" improvement in a weird metric?
        # No, best_obj is minimized. If a state is invalid, obj is large.
        # So best_blocks should be the most valid/smallest set found.
        # But we must ensure it is VALID before returning.
        
        # Quick validity check
        all_tsets = list(itertools.combinations(range(v), t))
        tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
        covered = np.zeros(len(all_tsets), dtype=bool)
        valid = True
        for b in sa_result:
            for ts in itertools.combinations(b, t):
                idx = tset_to_idx[ts]
                covered[idx] = True
        if not np.all(covered):
            # It's invalid. We should probably discard it or repair it.
            # For safety, discard if invalid.
            continue
            
        if len(sa_result) < best_len:
            best_overall = sa_result
            best_len = len(sa_result)
            
        restart_count += 1
        
    return best_overall

# EVOLVE-BLOCK-END
