import itertools
import random
import time
import math

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_indices = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets and their covered t-subsets
    # For v=24, k=7, this is 346,104 blocks.
    # We store: list of blocks (as tuples), and list of lists of t-subset indices.
    all_blocks = []
    block_covered_ts = []
    
    # This loop might take a few seconds.
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(t_subset_indices[ts])
        block_covered_ts.append(covered)
        
    num_blocks_total = len(all_blocks)
    
    # Helper to build a valid covering using greedy (initialization)
    def greedy_initial():
        covered = [0] * num_t
        blocks = []
        remaining = set(range(num_t))
        
        while remaining:
            # Pick a random uncovered t-subset to start with (or first)
            # Using first for determinism in init, random in SA
            target_idx = next(iter(remaining)) 
            target_t = all_t_subsets[target_idx]
            
            # Find a block containing target_t
            # Instead of scanning all blocks, we can precompute a mapping:
            # t_subset_idx -> list of block indices containing it.
            # This is C(24,3) -> list of C(21,4) ~ 5985 blocks.
            # Total memory: 2024 * 5985 * 4 bytes ~ 48MB. Feasible.
            # But we already have all_blocks. Scanning 346k blocks is slow if done many times.
            # Let's build the map.
            
            # Actually, let's just use the map for greedy.
            pass 
            
        return blocks # Placeholder

    # Let's build the map: t_idx -> [block_indices]
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, covered_list in enumerate(block_covered_ts):
        for t_idx in covered_list:
            t_to_blocks[t_idx].append(b_idx)
            
    def greedy_initial():
        covered = [0] * num_t
        blocks = []
        uncovered_indices = list(range(num_t))
        
        while uncovered_indices:
            # Pick first uncovered
            idx = uncovered_indices[0]
            target_t = all_t_subsets[idx]
            
            # Candidates are blocks in t_to_blocks[idx]
            candidates = t_to_blocks[idx]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for t_idx in block_covered_ts[b_idx]:
                    if covered[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            blocks.append(all_blocks[best_block_idx])
            for t_idx in block_covered_ts[best_block_idx]:
                if covered[t_idx] == 0:
                    covered[t_idx] = 1
                    # Remove from uncovered efficiently?
                    # Just filter at end of loop or use set.
            
            uncovered_indices = [i for i in uncovered_indices if covered[i] == 0]
            
        return blocks

    # Simulated Annealing
    def simulated_annealing(start_blocks, time_limit):
        current_blocks = list(start_blocks)
        # Build cover counts
        cover_counts = [0] * num_t
        for b in current_blocks:
            # b is a tuple
            # Need index of b in all_blocks? 
            # We stored blocks as tuples in all_blocks.
            # We need a map from tuple to index.
            pass
            
        # This approach of using indices is better.
        # Let's restart the SA logic with indices.
        
        return current_blocks

    # We need a map from block tuple to index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    def sa_search(initial_blocks, time_limit):
        # initial_blocks: list of tuples
        current_indices = [block_to_idx[b] for b in initial_blocks]
        
        # Initialize cover counts
        cover_counts = [0] * num_t
        for b_idx in current_indices:
            for t_idx in block_covered_ts[b_idx]:
                cover_counts[t_idx] += 1
                
        current_cost = len(current_indices)
        best_indices = list(current_indices)
        best_cost = current_cost
        
        start_time = time.time()
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        # Precompute sets for fast intersection/difference?
        # For swap validity:
        # B_old, B_new.
        # Invalid if exists t in B_old s.t. t not in B_new and cover_counts[t] == 1.
        
        while time.time() - start_time < time_limit:
            # Generate move
            move_type = random.random()
            
            if move_type < 0.4:
                # Delete
                if len(current_indices) <= 1:
                    continue
                b_idx = random.choice(current_indices)
                # Check validity
                valid = True
                for t_idx in block_covered_ts[b_idx]:
                    if cover_counts[t_idx] == 1:
                        valid = False
                        break
                if not valid:
                    # Try a different block? Or skip.
                    # To be more effective, pick a block that is "close" to being removable?
                    # For now, skip.
                    continue
                    
                # Apply
                current_indices.remove(b_idx)
                for t_idx in block_covered_ts[b_idx]:
                    cover_counts[t_idx] -= 1
                delta_cost = -1
                new_cost = current_cost + delta_cost
                # Always accept if improvement
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_indices = list(current_indices)
                current_cost = new_cost
                T *= cooling_rate
                
            elif move_type < 0.8:
                # Swap
                if len(current_indices) == 0:
                    continue
                b_old_idx = random.choice(current_indices)
                b_new_idx = random.randint(0, num_blocks_total - 1)
                
                if b_old_idx == b_new_idx:
                    continue
                    
                # Check validity
                # t in B_old must be covered by B_new or other blocks
                # Let's create a set of t_idx in B_new for fast lookup
                ts_new = set(block_covered_ts[b_new_idx])
                
                valid = True
                for t_idx in block_covered_ts[b_old_idx]:
                    if t_idx not in ts_new:
                        if cover_counts[t_idx] == 1:
                            valid = False
                            break
                            
                if not valid:
                    continue
                    
                # Apply
                current_indices[current_indices.index(b_old_idx)] = b_new_idx
                # Update counts
                for t_idx in block_covered_ts[b_old_idx]:
                    cover_counts[t_idx] -= 1
                for t_idx in block_covered_ts[b_new_idx]:
                    cover_counts[t_idx] += 1
                    
                delta_cost = 0
                # No cost change, but structure changed.
                # Acceptance: Always accept to explore? Or based on some metric?
                # In SA for combinatorial optimization with fixed cost objective (min size),
                # swaps are neutral. We accept them to escape local minima.
                # We might want to bias towards swaps that increase "excess" coverage
                # on non-critical items, or decrease "tightness".
                # For simplicity, accept all valid swaps.
                
                T *= cooling_rate
            else:
                # Add
                b_new_idx = random.randint(0, num_blocks_total - 1)
                if b_new_idx in current_indices:
                    continue
                    
                current_indices.append(b_new_idx)
                for t_idx in block_covered_ts[b_new_idx]:
                    cover_counts[t_idx] += 1
                delta_cost = 1
                new_cost = current_cost + delta_cost
                # Accept with prob exp(-delta/T)
                if random.random() < math.exp(-delta_cost / max(T, 1e-9)):
                    current_cost = new_cost
                else:
                    # Reject
                    current_indices.pop()
                    for t_idx in block_covered_ts[b_new_idx]:
                        cover_counts[t_idx] -= 1
                
                T *= cooling_rate
                
            if T < min_T:
                T = min_T
                
        return [all_blocks[i] for i in best_indices]

    # Run SA
    start_time = time.time()
    time_limit = 100
    
    # Get initial solution
    init_blocks = greedy_initial()
    
    best_blocks = init_blocks
    best_len = len(init_blocks)
    
    # Run multiple SA instances
    while time.time() - start_time < time_limit:
        # Perturb initial solution? Or just run SA on the current best?
        # Let's run SA on a slightly perturbed version of the best found so far
        # or just restart SA from the greedy solution with different random seeds.
        
        # To save time, we run SA for a short duration and keep the best.
        sa_time = min(5.0, time_limit - (time.time() - start_time))
        if sa_time <= 0:
            break
            
        candidate_blocks = sa_search(init_blocks, sa_time)
        
        if len(candidate_blocks) < best_len:
            best_len = len(candidate_blocks)
            best_blocks = candidate_blocks
            # Update init_blocks to the new best for subsequent runs?
            # Or keep the greedy one?
            # Using the best found as the start for the next SA can help.
            init_blocks = candidate_blocks
            
    return [list(b) for b in best_blocks]
