import itertools
import random
import time

def construct(v, k, t):
    # Parameters for the specific task, but function should be general
    # v=19, k=7, t=4 expected
    
    # 1. Greedy Initialization
    def greedy_cover():
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            # Pick the first uncovered subset to start the block
            start_subset = next(iter(uncovered))
            block = list(start_subset)
            
            while len(block) < k:
                # Find the point that covers the most uncovered subsets
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Calculate gain: how many new uncovered subsets are covered
                    gain = 0
                    # For each (t-1)-subset of the current block, check if adding x covers an uncovered t-subset
                    for rest in itertools.combinations(block, t - 1):
                        # Create the t-subset
                        new_subset = tuple(sorted(rest + (x,)))
                        if new_subset in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    # Fallback: pick any random point not in block
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        best_point = random.choice(candidates)
                    else:
                        break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        return blocks

    # 2. Simulated Annealing Optimizer
    def optimize(blocks, time_limit=110):
        # Precompute all possible blocks? No, too many (C(19,7)=50388). 
        # We will sample random blocks on the fly.
        
        # Helper to calculate energy
        def get_energy(blist):
            # Count uncovered subsets
            covered = set()
            for b in blist:
                for sub in itertools.combinations(b, t):
                    covered.add(sub)
            # Total t-subsets
            total_t_subsets = set(itertools.combinations(range(v), t))
            uncovered = total_t_subsets - covered
            if uncovered:
                # Heavy penalty for invalidity
                return 100000 + len(uncovered)
            else:
                return len(blist)
        
        current_blocks = [list(b) for b in blocks]
        current_energy = get_energy(current_blocks)
        
        best_blocks = [list(b) for b in current_blocks]
        best_energy = current_energy
        
        start_time = time.time()
        T = 100.0
        T_min = 0.1
        cooling_rate = 0.9999
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                break
            
            # Generate a neighbor
            new_blocks = [list(b) for b in current_blocks]
            move_type = random.random()
            
            if move_type < 0.3 and new_blocks:
                # Remove a random block
                idx = random.randint(0, len(new_blocks) - 1)
                new_blocks.pop(idx)
            elif move_type < 0.6:
                # Add a random block
                # Sample a random k-subset
                new_block = random.sample(range(v), k)
                new_block.sort()
                new_blocks.append(new_block)
            elif move_type < 0.8 and new_blocks:
                # Swap a random block with a random new block
                idx = random.randint(0, len(new_blocks) - 1)
                new_block = random.sample(range(v), k)
                new_block.sort()
                new_blocks[idx] = new_block
            else:
                # Perturb: Remove 1-2 blocks and re-greedy fill? 
                # Or just a larger jump: replace 2 blocks with 2 random ones
                if len(new_blocks) >= 2:
                    idx1 = random.randint(0, len(new_blocks) - 1)
                    idx2 = random.randint(0, len(new_blocks) - 1)
                    while idx2 == idx1 and len(new_blocks) > 1:
                        idx2 = random.randint(0, len(new_blocks) - 1)
                    new_blocks[idx1] = sorted(random.sample(range(v), k))
                    new_blocks[idx2] = sorted(random.sample(range(v), k))
                else:
                    # Fallback to add
                    new_block = random.sample(range(v), k)
                    new_block.sort()
                    new_blocks.append(new_block)
            
            new_energy = get_energy(new_blocks)
            
            # Acceptance
            delta = new_energy - current_energy
            if delta < 0:
                current_blocks = new_blocks
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = [list(b) for b in current_blocks]
            else:
                # Metropolis
                if random.random() < pow(2.718281828, -delta / T):
                    current_blocks = new_blocks
                    current_energy = new_energy
            
            T *= cooling_rate
        
        return best_blocks

    # 3. Cleanup: Remove redundant blocks
    def clean_up(blocks):
        # Convert to set of tuples for fast lookup
        block_set = [tuple(b) for b in blocks]
        # Count how many times each t-subset is covered
        coverage_count = {}
        for b in block_set:
            for sub in itertools.combinations(b, t):
                coverage_count[sub] = coverage_count.get(sub, 0) + 1
        
        # Try to remove blocks that are redundant
        # A block is redundant if all its t-subsets are covered by other blocks
        # i.e., for all sub in block, coverage_count[sub] > 1
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, b in enumerate(block_set):
                redundant = True
                for sub in itertools.combinations(b, t):
                    if coverage_count[sub] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    changed = True
                    # Update coverage counts
                    for sub in itertools.combinations(b, t):
                        coverage_count[sub] -= 1
                else:
                    new_blocks.append(b)
            block_set = new_blocks
        
        return [list(b) for b in block_set]

    # Main Execution
    # Initial greedy solution
    initial_blocks = greedy_cover()
    
    # Run SA
    optimized_blocks = optimize(initial_blocks, time_limit=110)
    
    # Final cleanup
    final_blocks = clean_up(optimized_blocks)
    
    return final_blocks
