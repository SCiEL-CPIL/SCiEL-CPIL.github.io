<<<<<<< SEARCH
    def minimize_with_swaps(blocks_indices, coverage_counts, time_limit, random_seed=None):
        """Local search with deletion and swap operations."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        current_block_set = set(current_blocks)
        
        if not current_blocks:
            return current_blocks, coverage_counts.copy()
            
        # Work with a copy of coverage counts
        cover_count = coverage_counts.copy()
        
        improved = True
        start_time = time.time()
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Deletion pass
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if i >= len(current_blocks):
                    continue
                if (time.time() - start_time) > time_limit:
                    break
                b_idx = current_blocks[i]
                is_redundant = True
                for s_idx in block_subset_arrays[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in block_subset_arrays[b_idx]:
                        cover_count[s_idx] -= 1
                    current_block_set.discard(b_idx)
                    current_blocks.pop(i)
                    improved = True
                    break  # Restart scan after removal
            
            # Swap pass
            if (time.time() - start_time) > time_limit:
                break
                
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            swaps_done = 0
            max_swaps = 30
            
            for i in indices:
                if (time.time() - start_time) > time_limit:
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(current_blocks):
                    continue
                    
                old_block = current_blocks[i]
                
                # Find unique subsets covered by old_block
                unique_subs = []
                for s_idx in block_subset_arrays[old_block]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    # Block is not critical - try swapping with a random block
                    # to diversify the solution
                    new_block = rng.randint(0, num_blocks - 1)
                    if new_block == old_block:
                        continue
                    if new_block in current_block_set:
                        continue
                        
                    # Perform swap
                    for s_idx in block_subset_arrays[old_block]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_subset_arrays[new_block]:
                        cover_count[s_idx] += 1
                    current_block_set.discard(old_block)
                    current_block_set.add(new_block)
                    current_blocks[i] = new_block
                    swaps_done += 1
                else:
                    # Block is critical - find candidates that cover all unique_subs
                    candidates = None
                    for s_idx in unique_subs:
                        if candidates is None:
                            candidates = set(subset_to_blocks_sets[s_idx])
                        else:
                            candidates.intersection_update(subset_to_blocks_sets[s_idx])
                            if not candidates:
                                break
                    
                    if candidates:
                        candidates.discard(old_block)
                        if candidates:
                            # Pick a random candidate
                            new_block = rng.choice(list(candidates))
                            # Check if new_block is already in current_blocks
                            if new_block in current_block_set:
                                continue
                                
                            # Perform swap
                            for s_idx in block_subset_arrays[old_block]:
                                cover_count[s_idx] -= 1
                            for s_idx in block_subset_arrays[new_block]:
                                cover_count[s_idx] += 1
                            current_block_set.discard(old_block)
                            current_block_set.add(new_block)
                            current_blocks[i] = new_block
                            swaps_done += 1
                            improved = True
        
        return current_blocks, cover_count
=======
    def minimize_with_swaps(blocks_indices, coverage_counts, time_limit, random_seed=None):
        """Local search with deletion and swap operations."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        current_block_set = set(current_blocks)
        
        if not current_blocks:
            return current_blocks, coverage_counts.copy()
            
        # Work with a copy of coverage counts
        cover_count = coverage_counts.copy()
        
        improved = True
        start_time = time.time()
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Deletion pass
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if i >= len(current_blocks):
                    continue
                if (time.time() - start_time) > time_limit:
                    break
                b_idx = current_blocks[i]
                is_redundant = True
                for s_idx in block_subset_arrays[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in block_subset_arrays[b_idx]:
                        cover_count[s_idx] -= 1
                    current_block_set.discard(b_idx)
                    current_blocks.pop(i)
                    improved = True
                    break  # Restart scan after removal
            
            # Swap pass
            if (time.time() - start_time) > time_limit:
                break
                
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            swaps_done = 0
            max_swaps = 30
            
            for i in indices:
                if (time.time() - start_time) > time_limit:
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(current_blocks):
                    continue
                    
                old_block = current_blocks[i]
                
                # Find unique subsets covered by old_block
                unique_subs = []
                for s_idx in block_subset_arrays[old_block]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    # Block is not critical - try swapping with a random block
                    # to diversify the solution
                    new_block = rng.randint(0, num_blocks - 1)
                    if new_block == old_block:
                        continue
                    if new_block in current_block_set:
                        continue
                        
                    # Perform swap
                    for s_idx in block_subset_arrays[old_block]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_subset_arrays[new_block]:
                        cover_count[s_idx] += 1
                    current_block_set.discard(old_block)
                    current_block_set.add(new_block)
                    current_blocks[i] = new_block
                    swaps_done += 1
                else:
                    # Block is critical - find candidates that cover all unique_subs
                    candidates = None
                    for s_idx in unique_subs:
                        if candidates is None:
                            candidates = set(subset_to_blocks_sets[s_idx])
                        else:
                            candidates.intersection_update(subset_to_blocks_sets[s_idx])
                            if not candidates:
                                break
                    
                    if candidates:
                        candidates.discard(old_block)
                        if candidates:
                            # Pick a random candidate
                            new_block = rng.choice(list(candidates))
                            # Check if new_block is already in current_blocks
                            if new_block in current_block_set:
                                continue
                                
                            # Perform swap
                            for s_idx in block_subset_arrays[old_block]:
                                cover_count[s_idx] -= 1
                            for s_idx in block_subset_arrays[new_block]:
                                cover_count[s_idx] += 1
                            current_block_set.discard(old_block)
                            current_block_set.add(new_block)
                            current_blocks[i] = new_block
                            swaps_done += 1
                            improved = True
        
        return current_blocks, cover_count
    
    def run_genetic_algorithm(initial_pop, population_size, num_generations, time_limit, random_seed=None):
        """Run a genetic algorithm to evolve better coverings."""
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        # Initialize population
        population = []
        for blocks in initial_pop:
            cover_count = compute_coverage_counts(blocks)
            if np.all(cover_count > 0):
                population.append((list(blocks), cover_count.copy()))
        
        # If we don't have enough valid solutions, repair them
        while len(population) < population_size and time.time() < start_time + time_limit:
            # Take a random existing solution and perturb it
            if population:
                base, _ = population[rng.integers(0, len(population))]
                perturbed = list(base)
                # Remove some blocks
                num_remove = max(1, len(perturbed) // 20)
                remove_idx = rng.choice(len(perturbed), size=min(num_remove, len(perturbed)), replace=False)
                remove_set = set(perturbed[i] for i in remove_idx)
                perturbed = [b for i, b in enumerate(perturbed) if i not in set(remove_idx.tolist())]
                repaired = repair_covering(perturbed, max_time=1.0, random_seed=int(rng.integers(0, 2**31)))
                if repaired:
                    cover_count = compute_coverage_counts(repaired)
                    if np.all(cover_count > 0):
                        population.append((repaired, cover_count.copy()))
            else:
                break
        
        if not population:
            return None
        
        best_solution = min(population, key=lambda x: len(x[0]))
        best_blocks = best_solution[0]
        best_count = best_solution[1]
        best_len = len(best_blocks)
        
        start_time = time.time()
        
        for gen in range(num_generations):
            if time.time() - start_time > time_limit:
                break
            
            # Selection: tournament selection
            new_population = []
            
            for _ in range(population_size):
                # Tournament of size 3
                candidates = rng.choice(len(population), size=3, replace=False)
                winner = min(candidates, key=lambda i: len(population[i][0]))
                new_population.append((list(population[winner][0]), population[winner][1].copy()))
            
            # Crossover and mutation
            for i in range(0, len(new_population) - 1, 2):
                parent1 = new_population[i][0]
                parent2 = new_population[i + 1][0]
                
                # Crossover
                child = crossover(parent1, parent2, rng)
                
                # Repair if needed
                cover_count = compute_coverage_counts(child)
                if not np.all(cover_count > 0):
                    child = repair_covering(child, max_time=1.0, random_seed=int(rng.integers(0, 2**31)))
                    cover_count = compute_coverage_counts(child)
                
                # Mutation
                if cover_count is not None and np.all(cover_count > 0):
                    child = mutate_criticality_weighted(child, cover_count, rng, num_mutations=2)
                    cover_count = compute_coverage_counts(child)
                
                if cover_count is not None and np.all(cover_count > 0):
                    new_population[i] = (child, cover_count)
                
                # Keep parent if child is worse
                if len(new_population[i][0]) > len(parent1):
                    new_population[i] = (list(parent1), compute_coverage_counts(parent1))
            
            # Elitism: keep the best solution
            if population:
                old_best = min(population, key=lambda x: len(x[0]))
                if len(new_population[0][0]) > len(old_best[0]):
                    new_population[0] = (list(old_best[0]), old_best[1].copy())
            
            # Update best
            gen_best = min(new_population, key=lambda x: len(x[0]))
            if len(gen_best[0]) < best_len:
                best_len = len(gen_best[0])
                best_blocks = gen_best[0]
                best_count = gen_best[1]
            
            population = new_population
        
        return best_blocks
>>>>>>> REPLACE

