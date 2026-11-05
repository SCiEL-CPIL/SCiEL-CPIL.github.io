import itertools
import random
import time
import math

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_indices = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    # This is crucial for fast updates
    block_t_indices = []
    for block in all_blocks:
        t_indices = []
        for ts in itertools.combinations(block, t):
            t_indices.append(t_subset_indices[ts])
        block_t_indices.append(t_indices)
        
    # Helper to get a valid initial solution using Greedy
    def greedy_init():
        covered = [0] * num_t
        uncovered_count = num_t
        blocks = []
        
        # To speed up greedy, we can just pick the block with max gain
        # We don't need to be as careful as before, just get a valid cover
        while uncovered_count > 0:
            best_gain = -1
            best_block_idx = -1
            
            # Optimization: We can sample a few blocks or just iterate all.
            # Iterating all 346k blocks * 76 steps * 35 t-subsets = ~900M ops. Too slow.
            # We need a faster way to pick a good block in greedy init.
            # Let's pick a random uncovered t-subset, and pick a random block containing it.
            # This is the "randomized greedy" or "first fit" style.
            
            # Find an uncovered t-subset
            # Instead of scanning, we can keep a list of uncovered indices.
            # But for init, let's just do a few random picks to build a base, then refine.
            
            # Actually, let's just use the previous greedy logic but simplified.
            # Or, even simpler: Start with empty, add random blocks until covered?
            # That might result in >100 blocks.
            
            # Let's use a quick greedy:
            # 1. Find all uncovered t-subsets.
            # 2. For each, find a block that covers it and maximizes gain.
            # To make this fast, we can precompute for each t-subset, the list of blocks covering it.
            # t-subset -> list of block indices.
            # Size: 2024 t-subsets. Each is in C(24-3, 7-3) = C(21,4) = 5985 blocks.
            # Total entries: 2024 * 5985 ~ 12M. Memory ~ 50MB. OK.
            
            pass # Will implement below
            
        return blocks

    # Precompute t_subset_to_blocks mapping
    # This helps in greedy init and potentially in SA moves
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, block in enumerate(all_blocks):
        for t_idx in block_t_indices[b_idx]:
            t_to_blocks[t_idx].append(b_idx)
            
    def greedy_init():
        covered = [0] * num_t
        uncovered_indices = list(range(num_t))
        blocks = []
        
        while uncovered_indices:
            # Pick a random uncovered t-subset to target
            target_idx = random.choice(uncovered_indices)
            target_t = all_t_subsets[target_idx]
            
            # Find the block among those covering target_idx that has max gain
            best_gain = -1
            best_block_idx = -1
            
            for b_idx in t_to_blocks[target_idx]:
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if not covered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    if gain == 35: # Max possible gain
                        break
            
            blocks.append(best_block_idx)
            
            # Update coverage
            for t_idx in block_t_indices[best_block_idx]:
                if not covered[t_idx]:
                    covered[t_idx] = 1
            # Rebuild uncovered list (slow but init is only once)
            uncovered_indices = [i for i in range(num_t) if not covered[i]]
            
        return blocks

    # Simulated Annealing
    def simulated_annealing(initial_blocks):
        # initial_blocks is a list of block indices
        current_blocks = set(initial_blocks)
        current_cost = len(current_blocks)
        
        # Initialize cover counts
        cover_count = [0] * num_t
        for b_idx in current_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        
        uncovered_count = sum(1 for c in cover_count if c == 0)
        
        best_blocks = set(current_blocks)
        best_cost = current_cost
        
        start_time = time.time()
        time_limit = 110 # Leave 10s buffer
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        # List of blocks in current solution for easy random access
        current_block_list = list(current_blocks)
        
        while time.time() - start_time < time_limit and T > min_T:
            # Generate move
            move_type = random.random()
            
            if move_type < 0.4:
                # Remove a block
                if not current_block_list:
                    continue
                b_idx = random.choice(current_block_list)
                
                # Check if removing it keeps it valid
                # It's valid if for all t in block, cover_count[t] > 1
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Apply move
                    current_blocks.remove(b_idx)
                    current_block_list.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    new_cost = current_cost - 1
                    # Always accept cost decrease
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = set(current_blocks)
                        
            elif move_type < 0.8:
                # Add a block
                b_idx = random.choice(range(num_blocks))
                if b_idx not in current_blocks:
                    # Adding is always valid
                    current_blocks.add(b_idx)
                    current_block_list.append(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] += 1
                    new_cost = current_cost + 1
                    # Accept with probability exp(-delta / T)
                    delta = 1
                    if random.random() < math.exp(-delta / T):
                        current_cost = new_cost
                        # If we accept, we are in a worse state, but that's SA
                    else:
                        # Rollback
                        current_blocks.remove(b_idx)
                        current_block_list.remove(b_idx)
                        for t_idx in block_t_indices[b_idx]:
                            cover_count[t_idx] -= 1
                            
            else:
                # Swap: Remove one, Add one
                if not current_block_list:
                    continue
                b_remove = random.choice(current_block_list)
                b_add = random.choice(range(num_blocks))
                
                if b_remove == b_add:
                    continue
                
                # Check validity of removing b_remove assuming b_add is added
                # This is tricky to check directly without applying.
                # Let's just apply and check uncovered_count?
                # Or check: Is there any t in b_remove such that cover_count[t] == 1 AND t is NOT in b_add?
                # If such a t exists, removing b_remove without b_add covering it makes it uncovered.
                # But b_add is added simultaneously.
                
                # Let's just do the swap and check if valid.
                # Apply Remove
                for t_idx in block_t_indices[b_remove]:
                    cover_count[t_idx] -= 1
                # Apply Add
                for t_idx in block_t_indices[b_add]:
                    cover_count[t_idx] += 1
                
                # Check if valid
                valid = True
                # We only need to check t-subsets in b_remove (since b_add only increases counts)
                # If any t in b_remove has count 0, it's invalid.
                for t_idx in block_t_indices[b_remove]:
                    if cover_count[t_idx] == 0:
                        valid = False
                        break
                        
                if valid:
                    # Accept swap (cost change 0)
                    # To avoid getting stuck, we can always accept swaps or accept with prob 0.5
                    if random.random() < 0.5:
                        current_blocks.remove(b_remove)
                        current_blocks.add(b_add)
                        current_block_list.remove(b_remove)
                        current_block_list.append(b_add)
                    else:
                        # Rollback
                        for t_idx in block_t_indices[b_remove]:
                            cover_count[t_idx] += 1
                        for t_idx in block_t_indices[b_add]:
                            cover_count[t_idx] -= 1
                else:
                    # Rollback
                    for t_idx in block_t_indices[b_remove]:
                        cover_count[t_idx] += 1
                    for t_idx in block_t_indices[b_add]:
                        cover_count[t_idx] -= 1
            
            T *= cooling_rate
            
        return list(best_blocks)

    # Run SA for multiple restarts
    best_overall = float('inf')
    best_solution = None
    start_total = time.time()
    
    while time.time() - start_total < 100: # Leave 20s for overhead/last run
        init_blocks = greedy_init()
        # Convert to indices if not already
        # greedy_init returns indices
        
        sa_result = simulated_annealing(init_blocks)
        
        if len(sa_result) < best_overall:
            best_overall = len(sa_result)
            best_solution = sa_result
            
        # Early stop if we hit lower bound
        if best_overall <= 69:
            break
            
    if best_solution is None:
        # Fallback
        init_blocks = greedy_init()
        best_solution = init_blocks
        
    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[idx]) for idx in best_solution]
    
    return final_blocks
