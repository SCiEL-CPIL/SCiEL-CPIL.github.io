"""Simulated annealing / local search for covering design C(v,k,t)."""
import itertools
import random
import time
import math

# EVOLVE-BLOCK-START


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a combination of:
    1. Greedy initialization to get a reasonable starting point.
    2. Local search with simulated annealing to reduce the number of blocks.
    
    The objective is to minimize the number of blocks such that every t-subset
    is covered by at least one block.
    """
    # Time budget: leave some margin for the 120s limit
    start_time = time.time()
    time_budget = 110  # seconds

    # Precompute all t-subsets for quick coverage checking
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast coverage tracking
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Coverage count: how many blocks cover each t-subset
    # We'll maintain this as a list of integers
    
    def block_covers(block):
        """Return the list of indices of t-subsets covered by this block."""
        return [subset_to_idx[tuple(s)] for s in itertools.combinations(block, t)]
    
    # --- Step 1: Greedy initialization ---
    uncovered_count = [1] * n_t_subsets  # initially all uncovered
    blocks = []
    
    # Greedy: repeatedly pick the block that covers the most currently uncovered t-subsets
    # For efficiency, we use a heuristic: start from an uncovered t-subset, build a block
    # around it by adding points that cover the most new uncovered t-subsets.
    
    while any(uncovered_count):
        # Find an uncovered t-subset
        uncovered_indices = [i for i in range(n_t_subsets) if uncovered_count[i] == 0]
        if not uncovered_indices:
            break
        start_subset = all_t_subsets[random.choice(uncovered_indices)]
        block = list(start_subset)
        
        while len(block) < k:
            # For each candidate point not in block, compute how many new uncovered t-subsets it would cover
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: number of t-subsets containing x and t-1 elements from block that are currently uncovered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx[sub]
                    if uncovered_count[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for idx in block_covers(block):
            uncovered_count[idx] += 1
    
    # Now we have a valid covering. Let's try to reduce the number of blocks.
    
    # --- Step 2: Local search to minimize number of blocks ---
    # Objective: minimize len(blocks)
    # We can try:
    # a) Removing blocks that are redundant (their t-subsets are covered by other blocks)
    # b) Replacing blocks with different ones that allow further deletions
    
    def is_redundant(block_idx, blocks):
        """Check if block at block_idx is redundant, i.e., all its t-subsets are covered by other blocks."""
        block = blocks[block_idx]
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            # Count how many blocks cover this t-subset
            count = 0
            for j, b in enumerate(blocks):
                if j == block_idx:
                    continue
                # Check if sub is contained in b
                if all(x in b for x in sub):
                    count += 1
                    if count > 0:
                        break
            if count == 0:
                return False
        return True
    
    # First pass: remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if is_redundant(i, blocks):
                del blocks[i]
                changed = True
    
    # Now do simulated annealing / local search
    # State: list of blocks
    # Moves:
    # 1. Remove a random block (if it keeps the covering valid)
    # 2. Replace a random block with a new random block
    # 3. Add a random block (to allow escape from local minima)
    # 4. Modify a block by swapping one element
    
    # To check validity efficiently, maintain coverage counts
    def build_coverage(blocks):
        """Build coverage count array for the given blocks."""
        cov = [0] * n_t_subsets
        for block in blocks:
            for idx in block_covers(block):
                cov[idx] += 1
        return cov
    
    def is_valid(cov):
        """Check if all t-subsets are covered at least once."""
        return all(c >= 1 for c in cov)
    
    # Start with current blocks
    current_blocks = list(blocks)
    current_cov = build_coverage(current_blocks)
    current_cost = len(current_blocks)
    
    best_blocks = list(current_blocks)
    best_cost = current_cost
    
    # Simulated annealing parameters
    temperature = 5.0
    cooling_rate = 0.999
    min_temperature = 0.1
    max_iterations = 50000
    
    rng = random.Random(42)
    
    iteration = 0
    while iteration < max_iterations and temperature > min_temperature:
        # Check time
        if time.time() - start_time > time_budget - 10:
            break
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.3 and len(current_blocks) > 1:
            # Try to remove a random block
            idx = rng.randint(0, len(current_blocks) - 1)
            block = current_blocks[idx]
            # Check if removing it keeps the covering valid
            # We need to check that all t-subsets of this block are covered by other blocks
            can_remove = True
            for sub in itertools.combinations(block, t):
                sub_idx = subset_to_idx[sub]
                if current_cov[sub_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                new_blocks = current_blocks[:idx] + current_blocks[idx+1:]
                new_cov = list(current_cov)
                for sub_idx in block_covers(block):
                    new_cov[sub_idx] -= 1
                new_cost = len(new_blocks)
                delta = new_cost - current_cost
                if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                    current_blocks = new_blocks
                    current_cov = new_cov
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
            iteration += 1
            temperature *= cooling_rate
        
        elif move_type < 0.5:
            # Replace a random block with a new random block
            if len(current_blocks) == 0:
                # Add a block
                new_block = sorted(rng.sample(range(v), k))
                new_blocks = current_blocks + [new_block]
                new_cov = list(current_cov)
                for sub_idx in block_covers(new_block):
                    new_cov[sub_idx] += 1
                new_cost = len(new_blocks)
                delta = new_cost - current_cost
                if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                    current_blocks = new_blocks
                    current_cov = new_cov
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
            else:
                idx = rng.randint(0, len(current_blocks) - 1)
                old_block = current_blocks[idx]
                new_block = sorted(rng.sample(range(v), k))
                new_blocks = list(current_blocks)
                new_blocks[idx] = new_block
                new_cov = list(current_cov)
                # Subtract old block coverage
                for sub_idx in block_covers(old_block):
                    new_cov[sub_idx] -= 1
                # Add new block coverage
                for sub_idx in block_covers(new_block):
                    new_cov[sub_idx] += 1
                new_cost = len(new_blocks)
                # Check validity
                if not is_valid(new_cov):
                    # Reject if invalid
                    iteration += 1
                    temperature *= cooling_rate
                    continue
                delta = new_cost - current_cost
                if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                    current_blocks = new_blocks
                    current_cov = new_cov
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
            iteration += 1
            temperature *= cooling_rate
        
        elif move_type < 0.7:
            # Modify a block by swapping one element
            if len(current_blocks) > 0:
                idx = rng.randint(0, len(current_blocks) - 1)
                block = list(current_blocks[idx])
                if len(block) > 0:
                    pos = rng.randint(0, len(block) - 1)
                    old_elem = block[pos]
                    new_elem = rng.choice([x for x in range(v) if x not in block])
                    new_block = sorted(block[:pos] + [new_elem] + block[pos+1:])
                    new_blocks = list(current_blocks)
                    new_blocks[idx] = new_block
                    new_cov = list(current_cov)
                    # Subtract old
                    for sub_idx in block_covers(block):
                        new_cov[sub_idx] -= 1
                    # Add new
                    for sub_idx in block_covers(new_block):
                        new_cov[sub_idx] += 1
                    if not is_valid(new_cov):
                        iteration += 1
                        temperature *= cooling_rate
                        continue
                    new_cost = len(new_blocks)
                    delta = new_cost - current_cost
                    if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                        current_blocks = new_blocks
                        current_cov = new_cov
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                iteration += 1
                temperature *= cooling_rate
        
        else:
            # Add a random block (to escape local minima)
            new_block = sorted(rng.sample(range(v), k))
            new_blocks = current_blocks + [new_block]
            new_cov = list(current_cov)
            for sub_idx in block_covers(new_block):
                new_cov[sub_idx] += 1
            new_cost = len(new_blocks)
            delta = new_cost - current_cost
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current_blocks = new_blocks
                current_cov = new_cov
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = list(current_blocks)
            iteration += 1
            temperature *= cooling_rate
    
    # Return the best solution found
    return best_blocks


# EVOLVE-BLOCK-END
