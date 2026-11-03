import itertools
import time
import numpy as np
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0 # 5s safety margin

    def time_left():
        return deadline - (time.time() - start_time)

    # 1. Precomputation
    # All t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # All k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Incidence: block -> list of t-indices
    block_t = []
    for block in k_blocks:
        covered = [t_index[combo] for combo in itertools.combinations(block, t)]
        block_t.append(covered)
        
    # Incidence: t-index -> list of block indices
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, covered in enumerate(block_t):
        for t_idx in covered:
            t_to_blocks[t_idx].append(b_idx)

    # 2. Greedy Initialization
    def run_greedy(seed, weight_mode='inverse'):
        rng = np.random.default_rng(seed)
        # Weights for t-subsets
        if weight_mode == 'inverse':
            # Initial weights: 1 / (number of blocks covering t)
            # We can approximate this or just use 1.0 for all initially and update
            # Standard greedy: pick block covering most uncovered
            # Weighted greedy: pick block maximizing sum of weights of uncovered t's it covers
            t_weights = np.ones(num_t, dtype=np.float64)
        elif weight_mode == 'random':
            t_weights = rng.random(num_t)
        else:
            t_weights = np.ones(num_t, dtype=np.float64)

        uncovered = np.ones(num_t, dtype=bool)
        selected = []
        selected_mask = np.zeros(num_k, dtype=bool)
        
        # Precompute gain for all blocks? No, update incrementally.
        # gain[b] = sum(t_weights[t] for t in block_t[b] if uncovered[t])
        gain = np.zeros(num_k, dtype=np.float64)
        for b in range(num_k):
            # Optimization: only compute if block covers any uncovered?
            # Initially all uncovered.
            # gain[b] = sum(t_weights[block_t[b]])
            # This is O(num_k * 56). 75582 * 56 ~ 4M ops. Fast enough.
            if block_t[b]:
                gain[b] = np.sum(t_weights[block_t[b]])
        
        while np.any(uncovered) and time_left() > 10:
            # Find best block
            # Exclude already selected
            masked_gain = gain.copy()
            masked_gain[selected_mask] = -1e10
            best_idx = int(np.argmax(masked_gain))
            if gain[best_idx] <= 0:
                break
                
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update uncovered and gains
            for t_idx in block_t[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    w = t_weights[t_idx]
                    # Decrease gain of all blocks covering this t_idx
                    for b in t_to_blocks[t_idx]:
                        if not selected_mask[b]:
                            gain[b] -= w
                            
        # Ensure validity (greedy might stop early if gains go weird, but usually doesn't)
        # If not fully covered, force add
        if np.any(uncovered):
            for t_idx in range(num_t):
                if uncovered[t_idx]:
                    # Pick any block covering it
                    candidates = [b for b in t_to_blocks[t_idx] if not selected_mask[b]]
                    if not candidates:
                        # Should not happen if we have all blocks
                        candidates = t_to_blocks[t_idx]
                    # Pick one (first one)
                    b = candidates[0]
                    selected.append(b)
                    selected_mask[b] = True
                    for t2 in block_t[b]:
                        uncovered[t2] = False
                        
        return selected

    def prune_cover(selected):
        """Remove redundant blocks from a valid cover."""
        if not selected:
            return selected
            
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in selected:
            for t_idx in block_t[b]:
                cover_count[t_idx] += 1
                
        selected_set = set(selected)
        new_selected = []
        
        # We can try to remove blocks in any order. 
        # Random order might help escape local minima in pruning? 
        # Actually, if a block is redundant, it's redundant regardless of order.
        # But removing one might make another redundant.
        # So we iterate until no change.
        
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias? Or just iterate.
            # Iterating is faster.
            for b in selected:
                if b not in selected_set:
                    continue
                # Check if redundant
                redundant = True
                for t_idx in block_t[b]:
                    if cover_count[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove
                    for t_idx in block_t[b]:
                        cover_count[t_idx] -= 1
                    selected_set.discard(b)
                    changed = True
            
            if changed:
                selected = [b for b in selected if b in selected_set]
                
        return selected

    def repair_cover(selected, max_time=2.0):
        """Add blocks to make the cover valid."""
        if not selected:
            return run_greedy(random.randint(0, 10000), 'inverse')
            
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in selected:
            for t_idx in block_t[b]:
                cover_count[t_idx] += 1
                
        uncovered = np.where(cover_count == 0)[0]
        if len(uncovered) == 0:
            return selected
            
        selected_set = set(selected)
        
        # Greedy repair
        # Weights: 1.0 for uncovered
        # We can just pick blocks that cover the most uncovered t's
        # Or just pick blocks covering specific uncovered t's
        
        # Efficient repair:
        # While there are uncovered t's:
        #   Pick an uncovered t
        #   Pick a block covering it that covers the most *other* uncovered t's
        #   Add it
        
        remaining_uncovered = set(uncovered)
        
        while remaining_uncovered and time_left() > max_time:
            # Pick a random uncovered t to start
            t_idx = random.choice(list(remaining_uncovered))
            
            best_block = -1
            best_count = -1
            
            candidates = t_to_blocks[t_idx]
            for b in candidates:
                if b in selected_set:
                    continue
                # Count how many remaining_uncovered it covers
                count = 0
                for t2 in block_t[b]:
                    if t2 in remaining_uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = b
                    
            if best_block == -1:
                # Should not happen
                break
                
            selected.append(best_block)
            selected_set.add(best_block)
            
            # Update remaining_uncovered
            for t2 in block_t[best_block]:
                if t2 in remaining_uncovered:
                    remaining_uncovered.discard(t2)
                    
        return selected

    def crossover(p1, p2):
        """
        Uniform crossover with repair.
        """
        if len(p1) > len(p2):
            p1, p2 = p2, p1
            
        # Start with a random subset of blocks from p1 and p2
        # To ensure diversity, we can pick blocks randomly from the union
        # But we want to combine "good" blocks.
        
        # Strategy:
        # 1. Create a candidate set by taking blocks from p1 and p2 randomly.
        # 2. Repair it.
        
        # Let's try: Take 50% of blocks from p1 and 50% from p2 (randomly selected)
        # This might be too small and require heavy repair.
        
        # Alternative: 
        # Child = set()
        # For each t-subset, pick a block from p1 or p2 that covers it.
        # This is a "t-wise" crossover.
        
        # Let's use a simpler heuristic:
        # Child starts as a copy of the smaller parent.
        # Then we try to add blocks from the larger parent if they improve "density" or cover gaps.
        # But we need to reduce size.
        
        # Let's stick to: 
        # 1. Merge blocks from p1 and p2.
        # 2. Prune.
        # 3. If still too large, try to swap.
        
        # Actually, the directive says: "crossover combines two valid covers by randomly selecting blocks from each parent and repairing coverage via greedy insertion"
        
        # Implementation:
        # 1. Create a list of blocks. 
        # 2. Randomly select blocks from p1 and p2 to form an initial child.
        #    How many? Maybe len(p1) // 2 + len(p2) // 2?
        #    Or just take a random subset of p1 and a random subset of p2.
        
        # Let's try taking a random subset of the union, biased towards smaller parents?
        # No, let's just take a random permutation of p1 and p2 blocks, 
        # and pick the first N blocks where N is min(len(p1), len(p2)).
        
        all_blocks = list(p1) + list(p2)
        random.shuffle(all_blocks)
        
        # Pick a target size, e.g., min(len(p1), len(p2))
        target_size = min(len(p1), len(p2))
        
        # Ensure we don't pick duplicates if possible, but set handles it.
        child_set = set()
        for b in all_blocks:
            if len(child_set) >= target_size:
                break
            child_set.add(b)
            
        child = list(child_set)
        
        # Repair
        child = repair_cover(child, max_time=1.0)
        
        # Prune
        child = prune_cover(child)
        
        return child

    def mutate(individual):
        """
        Mutation: 
        1. Remove a random block, repair, prune.
        2. Swap a random block with a random other block, repair, prune.
        """
        if not individual:
            return individual
            
        new_ind = individual[:]
        
        if random.random() < 0.5:
            # Removal mutation
            if len(new_ind) > 0:
                idx = random.randint(0, len(new_ind) - 1)
                new_ind.pop(idx)
                # Repair
                new_ind = repair_cover(new_ind, max_time=1.0)
                new_ind = prune_cover(new_ind)
        else:
            # Swap mutation
            if len(new_ind) > 0:
                idx = random.randint(0, len(new_ind) - 1)
                # Pick a random block from all blocks
                new_block = random.randint(0, num_k - 1)
                if new_block not in new_ind:
                    new_ind[idx] = new_block
                    # Repair (in case the new block is bad or we lost coverage)
                    new_ind = repair_cover(new_ind, max_time=1.0)
                    new_ind = prune_cover(new_ind)
                    
        return new_ind

    # 3. Main GA Loop
    population_size = 10
    population = []
    
    # Initialize population with diverse greedy solutions
    seeds = [42, 123, 777, 555, 111, 999, 2024, 314, 666, 888]
    modes = ['inverse', 'random', 'inverse', 'random', 'inverse', 'random', 'inverse', 'random', 'inverse', 'random']
    
    for i in range(population_size):
        if time_left() < 20:
            break
        sel = run_greedy(seeds[i % len(seeds)], modes[i % len(modes)])
        sel = prune_cover(sel)
        population.append(sel)
        
    # Keep track of best solution found
    best_solution = min(population, key=len) if population else []
    
    while time_left() > 5:
        # Generate offspring
        offspring = []
        
        # Number of offspring to generate
        num_offspring = population_size // 2 + 1
        
        for _ in range(num_offspring):
            if time_left() < 5:
                break
                
            # Selection: Tournament
            p1_idx = random.randint(0, len(population) - 1)
            p2_idx = random.randint(0, len(population) - 1)
            
            # Ensure distinct?
            if p1_idx == p2_idx and len(population) > 1:
                p2_idx = (p1_idx + 1) % len(population)
                
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover or Mutation
            if random.random() < 0.5:
                child = crossover(p1, p2)
            else:
                # Pick a random parent to mutate
                if random.random() < 0.5:
                    child = mutate(p1)
                else:
                    child = mutate(p2)
                    
            offspring.append(child)
            
        # Evaluate offspring (they are already pruned/valid)
        # Combine population and offspring
        all_candidates = population + offspring
        
        # Sort by length (fitness)
        all_candidates.sort(key=len)
        
        # Keep best population_size individuals
        population = all_candidates[:population_size]
        
        # Update best
        if population and len(population[0]) < len(best_solution):
            best_solution = population[0][:]

    # Final polish: try to prune the best solution one more time
    if best_solution and time_left() > 2:
        best_solution = prune_cover(best_solution)
        
    # Convert indices to blocks
    result = [list(k_blocks[i]) for i in best_solution]
    
    # Ensure validity (safety check)
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in best_solution:
            for idx in block_t[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            # Fallback to greedy if GA produced invalid (shouldn't happen)
            best_solution = run_greedy(1, 'inverse')
            best_solution = prune_cover(best_solution)
            result = [list(k_blocks[i]) for i in best_solution]
            
    return result
