import time
import random
import itertools
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = 110  # Leave 10s margin
    
    # 1. Precomputation
    # Generate all t-subsets and map to IDs
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the list of subset IDs it covers
    # block_id -> list of subset_ids
    block_coverage = []
    for b in all_blocks:
        covered_ids = []
        for sub in itertools.combinations(b, t):
            covered_ids.append(subset_to_id[sub])
        block_coverage.append(covered_ids)
        
    # Helper to get coverage of a block
    # We will use block indices (0 to n_blocks-1) to represent blocks in the state
    
    # 2. Initial Solution via Greedy
    # We start with no blocks and add greedily
    cover_counts = [0] * n_subsets
    selected_blocks = []
    uncovered_set = set(range(n_subsets))
    
    # Greedy Step
    # To speed up greedy, we can pick the block that covers the most uncovered subsets.
    # Since n_blocks is ~75k and n_subsets is ~11k, checking all blocks every step is $75k \times 56$ ops.
    # Number of steps is ~300. Total ops: $300 \times 75k \times 56 \approx 1.2$ billion. This might be too slow in Python.
    
    # Optimized Greedy:
    # Instead of checking all blocks, we can pick a random uncovered subset, 
    # and consider only blocks that contain it.
    # How to find blocks containing a subset?
    # We can precompute: subset_id -> list of block_ids that cover it.
    # subset_to_blocks = [[] for _ in range(n_subsets)]
    # for b_id, subs in enumerate(block_coverage):
    #     for s_id in subs:
    #         subset_to_blocks[s_id].append(b_id)
        
    # This precomputation is $4.2$ million appends. Fast enough.
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_id, subs in enumerate(block_coverage):
        for s_id in subs:
            subset_to_blocks[s_id].append(b_id)

    # Greedy Construction
    while uncovered_set:
        # Pick a random uncovered subset to break ties and add randomness
        target_subset_id = random.choice(list(uncovered_set))
        
        best_gain = -1
        best_block_id = -1
        
        # Candidate blocks are those covering target_subset_id
        candidates = subset_to_blocks[target_subset_id]
        
        for b_id in candidates:
            gain = 0
            for s_id in block_coverage[b_id]:
                if s_id in uncovered_set:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block_id = b_id
                
        # Add the best block
        selected_blocks.append(best_block_id)
        for s_id in block_coverage[best_block_id]:
            cover_counts[s_id] += 1
            if cover_counts[s_id] == 1:
                uncovered_set.discard(s_id)
                
    best_blocks = list(selected_blocks)
    best_cost = len(best_blocks)
    
    # 3. Simulated Annealing
    # State: selected_blocks (list of block IDs), cover_counts (array)
    # We track the number of uncovered subsets to calculate cost quickly.
    
    current_blocks = list(selected_blocks)
    current_cover_counts = list(cover_counts)
    current_uncovered_count = 0
    
    # Calculate initial uncovered count (should be 0)
    for c in current_cover_counts:
        if c == 0:
            current_uncovered_count += 1
            
    current_cost = len(current_blocks) + 100 * current_uncovered_count # Penalty for uncovered
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute block membership for fast add/remove?
    # We maintain a set of selected block IDs for O(1) lookup.
    selected_set = set(current_blocks)
    
    while time.time() < deadline and T > min_T:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a random block
            if not current_blocks:
                T *= cooling_rate
                continue
            b_id = random.choice(current_blocks)
            
            # Calculate delta
            delta_blocks = -1
            delta_uncovered = 0
            for s_id in block_coverage[b_id]:
                if current_cover_counts[s_id] == 1:
                    delta_uncovered += 1
            
            new_cost = current_cost + delta_blocks + 100 * delta_uncovered
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or (delta_cost > 0 and random.random() < math.exp(-delta_cost / T)):
                # Accept
                current_blocks.remove(b_id)
                selected_set.discard(b_id)
                for s_id in block_coverage[b_id]:
                    current_cover_counts[s_id] -= 1
                current_uncovered_count += delta_uncovered
                current_cost = new_cost
                
                # Update best if valid
                if current_uncovered_count == 0 and len(current_blocks) < best_cost:
                    best_cost = len(current_blocks)
                    best_blocks = list(current_blocks)
                    
        elif move_type < 0.7:
            # Add a random block
            b_id = random.randint(0, n_blocks - 1)
            if b_id in selected_set:
                T *= cooling_rate
                continue
                
            delta_blocks = 1
            delta_uncovered = 0
            for s_id in block_coverage[b_id]:
                if current_cover_counts[s_id] == 0:
                    delta_uncovered -= 1 # Uncovered becomes covered, so count decreases
            
            new_cost = current_cost + delta_blocks + 100 * delta_uncovered
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or (delta_cost > 0 and random.random() < math.exp(-delta_cost / T)):
                # Accept
                current_blocks.append(b_id)
                selected_set.add(b_id)
                for s_id in block_coverage[b_id]:
                    current_cover_counts[s_id] += 1
                current_uncovered_count += delta_uncovered
                current_cost = new_cost
                
                # Update best if valid
                if current_uncovered_count == 0 and len(current_blocks) < best_cost:
                    best_cost = len(current_blocks)
                    best_blocks = list(current_blocks)
                    
        else:
            # Swap: Remove one, Add one
            if not current_blocks:
                T *= cooling_rate
                continue
                
            b_remove = random.choice(current_blocks)
            b_add = random.randint(0, n_blocks - 1)
            
            if b_add in selected_set:
                T *= cooling_rate
                continue
                
            # Calculate delta
            delta_blocks = 0 # Net change in block count is 0
            
            # First, remove b_remove
            delta_uncovered_remove = 0
            for s_id in block_coverage[b_remove]:
                if current_cover_counts[s_id] == 1:
                    delta_uncovered_remove += 1
                # Tentatively decrement
                current_cover_counts[s_id] -= 1
                
            # Then, add b_add
            delta_uncovered_add = 0
            for s_id in block_coverage[b_add]:
                if current_cover_counts[s_id] == 0:
                    delta_uncovered_add -= 1 # Becomes covered
                # Tentatively increment
                current_cover_counts[s_id] += 1
            
            net_delta_uncovered = delta_uncovered_remove + delta_uncovered_add
            new_cost = current_cost + 100 * net_delta_uncovered
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or (delta_cost > 0 and random.random() < math.exp(-delta_cost / T)):
                # Accept
                current_blocks.remove(b_remove)
                selected_set.discard(b_remove)
                current_blocks.append(b_add)
                selected_set.add(b_add)
                
                # The counts were already updated in the tentative phase above?
                # Wait, I did tentative updates. If I reject, I must revert.
                # It's safer to calculate deltas without modifying state, or revert on reject.
                # Let's revert if rejected.
                if delta_cost >= 0 and random.random() >= math.exp(-delta_cost / T):
                     # Reject, revert counts
                     for s_id in block_coverage[b_remove]:
                         current_cover_counts[s_id] += 1
                     for s_id in block_coverage[b_add]:
                         current_cover_counts[s_id] -= 1
                else:
                     # Accept, state is already updated
                     current_uncovered_count += net_delta_uncovered
                     current_cost = new_cost
                     
                     if current_uncovered_count == 0 and len(current_blocks) < best_cost:
                         best_cost = len(current_blocks)
                         best_blocks = list(current_blocks)
            else:
                # Reject, revert counts
                for s_id in block_coverage[b_remove]:
                    current_cover_counts[s_id] += 1
                for s_id in block_coverage[b_add]:
                    current_cover_counts[s_id] -= 1

        T *= cooling_rate

    # Return the best valid solution found
    # Convert block IDs back to lists of points
    return [list(all_blocks[b_id]) for b_id in best_blocks]
