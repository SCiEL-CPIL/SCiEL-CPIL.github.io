import itertools
import random
import time
import copy

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Pre-compute all k-subsets
    all_k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(all_k_subsets)
    
    # Map block index to list of t-subset indices it covers
    block_t_indices = []
    for block in all_k_subsets:
        covered_ts = []
        for ts in itertools.combinations(block, t):
            covered_ts.append(t_subset_index[ts])
        block_t_indices.append(covered_ts)
        
    # Build reverse map: t_subset_index -> list of block indices
    # This is crucial for finding candidates that cover a specific subset
    ts_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, ts_list in enumerate(block_t_indices):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(b_idx)

    # --- Phase 1: Greedy Initialization ---
    def greedy_construct():
        coverage = [0] * n_t_subsets
        selected = []
        while True:
            uncovered = [i for i, c in enumerate(coverage) if c == 0]
            if not uncovered:
                break
            
            # Pick a random uncovered subset
            target = random.choice(uncovered)
            
            # Find best block covering 'target'
            best_block = -1
            best_gain = -1
            
            for b_idx in ts_to_blocks[target]:
                gain = 0
                for ts_idx in block_t_indices[b_idx]:
                    if coverage[ts_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected.append(best_block)
            for ts_idx in block_t_indices[best_block]:
                coverage[ts_idx] += 1
                
            if time.time() - start_time > time_limit * 0.1:
                break
        return selected, coverage

    init_blocks, init_cov = greedy_construct()
    
    # Ensure we have a valid solution to start
    # If greedy didn't finish, force cover remaining
    if any(c == 0 for c in init_cov):
        for i, c in enumerate(init_cov):
            if c == 0:
                # Pick any block covering i
                candidates = ts_to_blocks[i]
                if candidates:
                    b = candidates[0]
                    if b not in init_blocks:
                        init_blocks.append(b)
                        for ts_idx in block_t_indices[b]:
                            init_cov[ts_idx] += 1

    # --- Phase 2: Genetic Algorithm / Local Search Hybrid ---
    
    class Individual:
        def __init__(self, blocks, coverage):
            self.blocks = blocks
            self.coverage = coverage
            self.size = len(blocks)
            self.uncovered = sum(1 for c in coverage if c == 0)
            
        def fitness(self):
            # Primary: Minimize uncovered. Secondary: Minimize size
            return self.uncovered * 100000 + self.size

    # Population
    pop_size = 50
    population = []
    
    # Add the greedy solution
    population.append(Individual(copy.deepcopy(init_blocks), copy.deepcopy(init_cov)))
    
    # Add some random valid perturbations
    for _ in range(pop_size - 1):
        b = copy.deepcopy(init_blocks)
        cov = copy.deepcopy(init_cov)
        
        # Randomly swap some blocks to create diversity
        for _ in range(random.randint(5, 20)):
            if not b: break
            i = random.randrange(len(b))
            old_b = b[i]
            unique_ts = [ts for ts in block_t_indices[old_b] if cov[ts] == 1]
            
            if unique_ts:
                # Find a candidate that covers all unique_ts
                # Intersect lists of blocks covering each unique ts
                cands = set(ts_to_blocks[unique_ts[0]])
                for ts in unique_ts[1:]:
                    cands.intersection_update(ts_to_blocks[ts])
                
                # Exclude old_b
                cands.discard(old_b)
                
                if cands:
                    new_b = random.choice(list(cands))
                    # Apply swap
                    for ts in block_t_indices[old_b]:
                        cov[ts] -= 1
                    for ts in block_t_indices[new_b]:
                        cov[ts] += 1
                    b[i] = new_b
            
        population.append(Individual(b, cov))

    best_individual = min(population, key=lambda x: x.fitness())
    
    generations = 0
    max_generations = 5000
    
    while time.time() - start_time < time_limit and generations < max_generations:
        generations += 1
        new_population = []
        
        # Selection: Keep best
        new_population.append(copy.deepcopy(best_individual))
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Pick parent
            parent = random.choice(population)
            
            # Mutation Strategy:
            # 1. Try to remove a block
            # 2. Try to swap a block
            # 3. Try to add a block (if uncovered exists)
            
            child_blocks = copy.deepcopy(parent.blocks)
            child_cov = copy.deepcopy(parent.coverage)
            
            # Mutation 1: Removal
            if child_blocks:
                i = random.randrange(len(child_blocks))
                b_idx = child_blocks[i]
                can_remove = True
                for ts in block_t_indices[b_idx]:
                    if child_cov[ts] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts in block_t_indices[b_idx]:
                        child_cov[ts] -= 1
                    child_blocks.pop(i)
                else:
                    # Mutation 2: Swap
                    # Pick a block to swap
                    i = random.randrange(len(child_blocks))
                    old_b = child_blocks[i]
                    unique_ts = [ts for ts in block_t_indices[old_b] if child_cov[ts] == 1]
                    
                    if unique_ts:
                        cands = set(ts_to_blocks[unique_ts[0]])
                        for ts in unique_ts[1:]:
                            cands.intersection_update(ts_to_blocks[ts])
                        cands.discard(old_b)
                        
                        if cands:
                            # Pick best candidate (covers most low-coverage ts)
                            best_new = -1
                            best_score = -1
                            # Sample a few candidates if list is huge
                            cand_list = list(cands)
                            if len(cand_list) > 50:
                                cand_list = random.sample(cand_list, 50)
                            
                            for nb in cand_list:
                                score = 0
                                for ts in block_t_indices[nb]:
                                    if child_cov[ts] == 1:
                                        score += 1
                                if score > best_score:
                                    best_score = score
                                    best_new = nb
                            
                            if best_new != -1:
                                for ts in block_t_indices[old_b]:
                                    child_cov[ts] -= 1
                                for ts in block_t_indices[best_new]:
                                    child_cov[ts] += 1
                                child_blocks[i] = best_new
                    else:
                        # No unique ts, can remove
                        for ts in block_t_indices[old_b]:
                            child_cov[ts] -= 1
                        child_blocks.pop(i)
                        
            # If there are still uncovered subsets, try to add a block
            if any(c == 0 for c in child_cov):
                uncovered = [i for i, c in enumerate(child_cov) if c == 0]
                if uncovered:
                    target = random.choice(uncovered)
                    cands = ts_to_blocks[target]
                    if cands:
                        # Pick best
                        best_b = -1
                        best_gain = -1
                        for nb in cands:
                            if nb in child_blocks: continue
                            gain = 0
                            for ts in block_t_indices[nb]:
                                if child_cov[ts] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_b = nb
                        if best_b != -1:
                            child_blocks.append(best_b)
                            for ts in block_t_indices[best_b]:
                                child_cov[ts] += 1

            child = Individual(child_blocks, child_cov)
            new_population.append(child)
            
        # Update population
        population = new_population
        
        # Find new best
        current_best = min(population, key=lambda x: x.fitness())
        if current_best.fitness() < best_individual.fitness():
            best_individual = current_best
            # Optionally, re-seed population with best
            # population[0] = copy.deepcopy(best_individual)
            
        # Check time
        if time.time() - start_time > time_limit * 0.9:
            break
            
    # Final cleanup: Try to remove any redundant blocks from best_individual
    final_blocks = copy.deepcopy(best_individual.blocks)
    final_cov = copy.deepcopy(best_individual.coverage)
    
    changed = True
    while changed:
        changed = False
        for i in range(len(final_blocks)):
            b_idx = final_blocks[i]
            can_remove = True
            for ts in block_t_indices[b_idx]:
                if final_cov[ts] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts in block_t_indices[b_idx]:
                    final_cov[ts] -= 1
                final_blocks.pop(i)
                changed = True
                break # Restart loop

    return [list(all_k_subsets[i]) for i in final_blocks]
