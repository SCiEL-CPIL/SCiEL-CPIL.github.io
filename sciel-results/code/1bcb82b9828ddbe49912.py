import time
import random
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = 110  # Leave 10s margin
    
    # Precompute all t-subsets and map them to IDs
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Helper to get ID of a subset
    # Since we generate subsets from blocks, we need to check if they are in the map
    # For speed, we can just use the tuple directly as key in a dict or set
    
    # For the greedy step, we need to know which subsets are uncovered.
    # We will use a set of uncovered subset tuples.
    
    def greedy_construct(uncovered_set):
        blocks = []
        # Work with a copy of the uncovered set? 
        # No, we need to update it as we add blocks.
        # But we are passed a set. We should probably work with a boolean array or a set we can modify.
        # Let's use a set of tuples for simplicity, as n_subsets is small (11628).
        
        current_uncovered = set(uncovered_set)
        
        while current_uncovered:
            # Pick a subset to cover
            # Using min is deterministic. Random might help escape local minima in perturbation?
            # For the initial greedy, min is fine.
            target_subset = min(current_uncovered)
            
            # We need to form a block of size k containing target_subset
            # target_subset has size t. We need k-t more points.
            missing_points = [x for x in range(v) if x not in target_subset]
            num_to_pick = k - t
            
            best_gain = -1
            best_combo = None
            
            # Iterate over combinations of missing_points
            # If num_to_pick is small (3), this is fast.
            for combo in itertools.combinations(missing_points, num_to_pick):
                candidate_block = tuple(sorted(target_subset + combo))
                
                # Calculate gain: how many uncovered subsets are covered by this block
                gain = 0
                for sub in itertools.combinations(candidate_block, t):
                    if sub in current_uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_combo = combo
                    if gain == len(current_uncovered): # Perfect, break early
                        break
            
            # Add the best block
            final_block = tuple(sorted(target_subset + best_combo))
            blocks.append(final_block)
            
            # Update uncovered set
            for sub in itertools.combinations(final_block, t):
                current_uncovered.discard(sub)
                
        return blocks

    def minimize(blocks):
        # Convert blocks to list of lists for mutability if needed, but tuples are fine
        # Calculate coverage counts
        cover_count = [0] * n_subsets
        
        # Map block to list of subset IDs it covers
        block_subsets = []
        for b in blocks:
            subs = []
            for sub in itertools.combinations(b, t):
                subs.append(subset_to_id[sub])
            block_subsets.append(subs)
            for id_ in subs:
                cover_count[id_] += 1
        
        # Iteratively remove redundant blocks
        removed = True
        while removed:
            removed = False
            new_blocks = []
            new_block_subsets = []
            for i, b in enumerate(blocks):
                # Check if any subset in this block has count 1
                is_redundant = True
                for id_ in block_subsets[i]:
                    if cover_count[id_] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block
                    for id_ in block_subsets[i]:
                        cover_count[id_] -= 1
                    removed = True
                else:
                    new_blocks.append(b)
                    new_block_subsets.append(block_subsets[i])
            
            blocks = new_blocks
            block_subsets = new_block_subsets
            
        return blocks

    # Initial Solution
    initial_uncovered = set(all_subsets)
    initial_blocks = greedy_construct(initial_uncovered)
    best_blocks = minimize(initial_blocks)
    
    # Local Search / Iterated Perturbation
    while time.time() < deadline:
        # Perturbation: Remove 1-3 random blocks and repair
        num_remove = random.randint(1, 3)
        perturbed_blocks = list(best_blocks)
        
        # Remove random blocks
        for _ in range(num_remove):
            if not perturbed_blocks:
                break
            idx = random.randint(0, len(perturbed_blocks) - 1)
            perturbed_blocks.pop(idx)
            
        # Find uncovered subsets
        cover_count = [0] * n_subsets
        for b in perturbed_blocks:
            for sub in itertools.combinations(b, t):
                cover_count[subset_to_id[sub]] += 1
                
        uncovered = set()
        for i, cnt in enumerate(cover_count):
            if cnt == 0:
                uncovered.add(all_subsets[i])
                
        if not uncovered:
            continue
            
        # Repair with greedy
        repair_blocks = greedy_construct(uncovered)
        
        # Combine and minimize
        candidate_blocks = perturbed_blocks + repair_blocks
        candidate_blocks = minimize(candidate_blocks)
        
        if len(candidate_blocks) < len(best_blocks):
            best_blocks = candidate_blocks
            # Optional: update deadline if we made a big jump? No, keep it simple.
            
        # Additional: Try point swaps?
        # If time allows, try swapping a point in a random block to see if it helps minimize.
        # This is more complex. Let's stick to remove-repair for now as it's robust.

    return [list(b) for b in best_blocks]
