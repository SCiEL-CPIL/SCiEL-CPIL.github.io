import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115
    
    # 1. Pre-compute subset IDs
    # Map each t-subset to a unique integer ID
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    
    # Map each block to the list of subset IDs it covers
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_ids = {}
    for b in all_blocks:
        ids = [subset_to_id[s] for s in itertools.combinations(b, t)]
        block_to_ids[b] = ids
        
    # 2. Greedy Construction for Initial Valid Solution
    uncovered = [1] * num_subsets # 1 if uncovered, 0 if covered
    uncovered_count = num_subsets
    blocks = []
    used_blocks = set()
    
    # Pre-compute for greedy: for each subset, which blocks cover it?
    # This is expensive to build fully, but we can do it on the fly or pre-compute for greedy only.
    # Given the size, pre-computing subset_to_blocks is C(24,3) * C(21,4) ~ 2000 * 6000 = 12M entries.
    # Memory might be an issue (12M ints ~ 100MB+). 
    # Alternative: In greedy, pick random uncovered subset, iterate all blocks? No, too slow.
    # Let's stick to the seed's greedy logic but optimized with IDs if possible, 
    # or just use a simpler greedy: 
    # Pick uncovered subset, find best block covering it.
    # To find blocks covering a subset S, we need to scan all blocks or have an index.
    # Let's build the index subset_to_blocks for the greedy phase only.
    
    subset_to_blocks = {i: [] for i in range(num_subsets)}
    for b, ids in block_to_ids.items():
        for sid in ids:
            subset_to_blocks[sid].append(b)
            
    # Greedy
    while uncovered_count > 0:
        # Pick a random uncovered subset to avoid bias
        # Finding a random 1 in the list is O(num_subsets) if done naively.
        # We can keep a list of uncovered indices.
        # For simplicity and speed, let's just pick the first uncovered one? 
        # Or random index, check if uncovered.
        # Let's maintain a list of uncovered indices for efficiency.
        
        # Actually, let's just use the standard greedy: pick subset with fewest available blocks?
        # Or just random uncovered.
        # Let's pick a random index, if uncovered, use it.
        # This might take a few tries.
        
        target_sid = -1
        for _ in range(100): # Try 100 times to find an uncovered one randomly
            idx = random.randint(0, num_subsets - 1)
            if uncovered[idx]:
                target_sid = idx
                break
        if target_sid == -1:
            # Fallback: linear scan
            for idx in range(num_subsets):
                if uncovered[idx]:
                    target_sid = idx
                    break
                    
        if target_sid == -1:
            break
            
        candidates = subset_to_blocks[target_sid]
        best_block = None
        best_gain = -1
        
        for b in candidates:
            if b in used_blocks:
                continue
            # Calculate gain
            gain = 0
            for sid in block_to_ids[b]:
                if uncovered[sid]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
                
        if best_block is None:
            break # Should not happen
            
        blocks.append(best_block)
        used_blocks.add(best_block)
        
        # Update uncovered
        for sid in block_to_ids[best_block]:
            if uncovered[sid]:
                uncovered[sid] = 0
                uncovered_count -= 1

    if uncovered_count > 0:
        # Fallback: return what we have (invalid) or force completion?
        # The task requires valid. If greedy fails, we return empty or try to fix?
        # Greedy should always succeed for these parameters.
        pass

    # 3. Simulated Annealing
    # State: list of blocks
    # Energy: uncovered_count
    # Cost: len(blocks)
    # Objective: Minimize len(blocks) + LAMBDA * uncovered_count
    
    LAMBDA = 1000 # High penalty for uncovered
    
    # Initialize coverage counts
    coverage_counts = [0] * num_subsets
    for b in blocks:
        for sid in block_to_ids[b]:
            coverage_counts[sid] += 1
            
    def get_energy():
        # Count uncovered
        u = 0
        for c in coverage_counts:
            if c == 0:
                u += 1
        return u
        
    current_energy = get_energy()
    current_cost = len(blocks)
    current_obj = current_cost + LAMBDA * current_energy
    
    best_blocks = list(blocks)
    best_obj = current_obj
    
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.99
    
    steps = 0
    max_steps = 500000 # Limit steps to ensure time safety
    
    while time.time() - start_time < time_limit and steps < max_steps:
        steps += 1
        move_type = random.random()
        
        new_blocks = list(blocks)
        
        if move_type < 0.4:
            # Remove a block
            if not blocks:
                continue
            idx = random.randint(0, len(blocks) - 1)
            b = blocks[idx]
            new_blocks.pop(idx)
            
        elif move_type < 0.8:
            # Add a block
            # Pick a random block not in current set? Or any block?
            # Adding a block increases cost. Only useful if it reduces energy significantly.
            # To be efficient, pick a block that covers many uncovered subsets.
            # Simple: Pick random block from all_blocks.
            b = random.choice(all_blocks)
            new_blocks.append(b)
            
        else:
            # Swap a block
            if not blocks:
                continue
            idx = random.randint(0, len(blocks) - 1)
            old_b = blocks[idx]
            new_b = random.choice(all_blocks)
            new_blocks[idx] = new_b
            
        # Calculate change in energy and cost
        # This is expensive if we recalculate from scratch.
        # We need to update coverage_counts incrementally.
        
        # Let's simulate the move on a copy of coverage_counts? 
        # No, that's O(num_subsets) = 2024. 
        # Better: Apply move to coverage_counts, calculate delta, then revert if rejected.
        
        # Identify blocks added and removed
        # This logic is tricky with lists. 
        # Let's just recalculate energy for the new_blocks? 
        # O(len(new_blocks) * 35). If len is ~70, that's 2450 ops. 
        # 500k steps * 2450 ops = 1.2B ops. Too slow in Python.
        
        # We MUST use incremental updates.
        
        # Let's define the move explicitly:
        # If Remove: remove b from counts.
        # If Add: add b to counts.
        # If Swap: remove old, add new.
        
        # We need to track which blocks were added/removed to update counts.
        
        # Let's refactor the move generation to return (added_blocks, removed_blocks)
        
        if move_type < 0.4:
            if not blocks:
                continue
            idx = random.randint(0, len(blocks) - 1)
            b = blocks[idx]
            added = []
            removed = [b]
        elif move_type < 0.8:
            b = random.choice(all_blocks)
            added = [b]
            removed = []
        else:
            if not blocks:
                continue
            idx = random.randint(0, len(blocks) - 1)
            old_b = blocks[idx]
            new_b = random.choice(all_blocks)
            added = [new_b]
            removed = [old_b]
            
        # Apply changes to coverage_counts
        delta_energy = 0
        for b in added:
            for sid in block_to_ids[b]:
                if coverage_counts[sid] == 0:
                    delta_energy -= 1 # Was uncovered, now covered
                coverage_counts[sid] += 1
                
        for b in removed:
            for sid in block_to_ids[b]:
                coverage_counts[sid] -= 1
                if coverage_counts[sid] == 0:
                    delta_energy += 1 # Was covered, now uncovered
                    
        new_energy = current_energy + delta_energy
        new_cost = current_cost + len(added) - len(removed)
        new_obj = new_cost + LAMBDA * new_energy
        
        delta_obj = new_obj - current_obj
        
        # Acceptance
        if delta_obj < 0 or (T > 0 and random.random() < math.exp(-delta_obj / T)):
            # Accept
            # Update state
            # We need to update the blocks list to match the move
            if move_type < 0.4:
                blocks.pop(idx)
            elif move_type < 0.8:
                blocks.append(b)
            else:
                blocks[idx] = new_b
                
            current_energy = new_energy
            current_cost = new_cost
            current_obj = new_obj
            
            if new_energy == 0 and current_obj < best_obj:
                best_obj = current_obj
                best_blocks = list(blocks)
        else:
            # Revert coverage_counts
            for b in added:
                for sid in block_to_ids[b]:
                    coverage_counts[sid] -= 1
            for b in removed:
                for sid in block_to_ids[b]:
                    coverage_counts[sid] += 1
                    
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return [list(b) for b in best_blocks]
