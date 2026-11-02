import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
    
    # Convert to numpy arrays for fast operations
    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed, explore_prob=0.0):
        """Vectorized greedy with optional exploration."""
        random.seed(seed)
        gain = np.full(num_k_blocks, int(np.prod([k - i for i in range(t)])) // int(np.prod([i + 1 for i in range(t)])), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 10.0:
                return None
            
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            if explore_prob > 0 and random.random() < explore_prob:
                explore_cands = np.where((gain >= max_gain - 2) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            is_uncovered[newly_covered] = False
            
            blocks_to_decrement = []
            for ts_idx in newly_covered:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    def optimize_redundancy(blocks):
        """Remove redundant blocks."""
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)
        
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks
                
                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    block_ts_indices.pop(i)
                    changed = True
                    break
                    
        return blocks

    def local_search_swap(blocks, max_iterations=100):
        """1-exchange local search."""
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)
        
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            if time.time() > deadline - 5.0:
                return blocks
            
            improved = False
            iteration += 1
            random.shuffle(blocks)
            block_ts_indices = []
            for b in blocks:
                idx = get_block_idx(b)
                if idx >= 0:
                    block_ts_indices.append(block_to_ts[idx])
                else:
                    ts_list = []
                    for ts in itertools.combinations(b, t):
                        ts_list.append(ts_index_map[ts])
                    block_ts_indices.append(ts_list)
            
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks
                
                unique_ts = []
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)
                        
                if not unique_ts:
                    continue
                
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                if not valid_candidates:
                    continue
                    
                new_block_idx = random.choice(valid_candidates)
                new_block = list(all_k_blocks[new_block_idx])
                
                old_block = blocks[i]
                old_block_idx = get_block_idx(old_block)
                
                # Update cover counts
                for ts_idx in block_ts_indices[i]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[new_block_idx]:
                    cover_count[ts_idx] += 1
                
                blocks[i] = new_block
                sol_block_indices.discard(old_block_idx)
                sol_block_indices.add(new_block_idx)
                block_ts_indices[i] = block_to_ts[new_block_idx]
                improved = True
                break
        
        return blocks

    def genetic_algorithm(initial_blocks, generations=5, pop_size=20):
        """
        Genetic algorithm with multi-objective fitness:
        - Primary: minimize number of blocks
        - Secondary: minimize uncovered t-subsets (should be 0 for valid solutions)
        
        Crossover: uniform crossover on block lists
        Mutation: replace a random block with a random new block, or add/remove blocks
        """
        # Encode solution as list of block indices
        def encode(blocks):
            result = []
            for b in blocks:
                idx = get_block_idx(b)
                if idx >= 0:
                    result.append(idx)
            return result
        
        def decode(block_indices):
            return [list(all_k_blocks[idx]) for idx in block_indices]
        
        def fitness(block_indices):
            """Multi-objective: (num_blocks, num_uncovered)"""
            if not block_indices:
                return (float('inf'), num_t_subsets)
            
            cover_count = np.zeros(num_t_subsets, dtype=np.int32)
            for idx in block_indices:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            
            uncovered = int(np.sum(cover_count == 0))
            return (len(block_indices), uncovered)
        
        def crossover(parent1, parent2):
            """Uniform crossover."""
            child = []
            for i in range(max(len(parent1), len(parent2))):
                if random.random() < 0.5:
                    if i < len(parent1):
                        child.append(parent1[i])
                else:
                    if i < len(parent2):
                        child.append(parent2[i])
            # Remove duplicates while preserving order
            seen = set()
            unique_child = []
            for idx in child:
                if idx not in seen:
                    seen.add(idx)
                    unique_child.append(idx)
            return unique_child
        
        def mutate(block_indices):
            """Mutation: replace, add, or remove a block."""
            if not block_indices:
                return [random.randint(0, num_k_blocks - 1)]
            
            mutated = list(block_indices)
            op = random.random()
            
            if op < 0.5 and len(mutated) > 1:
                # Replace a random block
                i = random.randint(0, len(mutated) - 1)
                mutated[i] = random.randint(0, num_k_blocks - 1)
            elif op < 0.7:
                # Add a random block
                mutated.append(random.randint(0, num_k_blocks - 1))
            else:
                # Remove a random block
                i = random.randint(0, len(mutated) - 1)
                mutated.pop(i)
            
            # Remove duplicates
            seen = set()
            unique_mutated = []
            for idx in mutated:
                if idx not in seen:
                    seen.add(idx)
                    unique_mutated.append(idx)
            return unique_mutated
        
        # Initialize population
        initial_encoded = encode(initial_blocks)
        population = [initial_encoded]
        
        # Add diverse individuals
        for _ in range(pop_size - 1):
            if time.time() > deadline - 10.0:
                break
            sol = run_greedy(random.randint(0, 10000), explore_prob=0.1)
            if sol is not None:
                population.append(encode(sol))
        
        # Remove duplicates
        unique_pop = []
        seen_sets = set()
        for ind in population:
            key = frozenset(ind)
            if key not in seen_sets:
                seen_sets.add(key)
                unique_pop.append(ind)
        population = unique_pop if unique_pop else [initial_encoded]
        
        best_individual = initial_encoded
        best_fitness = fitness(initial_encoded)
        
        for gen in range(generations):
            if time.time() > deadline - 15.0:
                break
            
            # Evaluate fitness
            fitnesses = [fitness(ind) for ind in population]
            
            # Update best
            for i, fit in enumerate(fitnesses):
                if fit < best_fitness:
                    best_fitness = fit
                    best_individual = population[i]
            
            # Selection: tournament selection
            new_population = []
            for _ in range(len(population)):
                if time.time() > deadline - 15.0:
                    break
                
                # Tournament of 3
                tournament = random.sample(range(len(population)), min(3, len(population)))
                winner = min(tournament, key=lambda i: fitnesses[i])
                new_population.append(list(population[winner]))
            
            # Crossover and mutation
            next_population = []
            for i in range(0, len(new_population) - 1, 2):
                if time.time() > deadline - 15.0:
                    break
                
                p1 = new_population[i]
                p2 = new_population[i + 1]
                
                child = crossover(p1, p2)
                child = mutate(child)
                next_population.append(child)
                
                # Also keep parents with some probability
                if random.random() < 0.3:
                    next_population.append(list(p1))
                if random.random() < 0.3:
                    next_population.append(list(p2))
            
            # Ensure population isn't empty
            if not next_population:
                next_population = [list(best_individual)]
            
            population = next_population
        
        return decode(best_individual)

    # Generate initial solutions using vectorized greedy
    best_solution = None
    best_len = float('inf')
    
    # First, deterministic greedy
    current_blocks = run_greedy(0, explore_prob=0.0)
    if current_blocks is not None:
        current_blocks = optimize_redundancy(current_blocks)
        current_blocks = local_search_swap(current_blocks)
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_solution = current_blocks
    
    # Run multiple randomized greedy constructions
    num_runs = 80
    for i in range(1, num_runs):
        if time.time() > deadline - 25.0:
            break
        
        # Vary exploration probability
        if i < 30:
            explore = 0.0
        elif i < 50:
            explore = 0.05
        elif i < 70:
            explore = 0.1
        else:
            explore = 0.15
        
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None:
            sol = optimize_redundancy(sol)
            sol = local_search_swap(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
    
    # Apply genetic algorithm to the best solution found so far
    if best_solution is not None and time.time() < deadline - 40.0:
        ga_solution = genetic_algorithm(best_solution, generations=4, pop_size=15)
        ga_solution = optimize_redundancy(ga_solution)
        ga_solution = local_search_swap(ga_solution)
        if len(ga_solution) < best_len:
            best_len = len(ga_solution)
            best_solution = ga_solution
    
    # Final polish: remove any remaining redundant blocks
    if best_solution is not None:
        best_solution = optimize_redundancy(best_solution)
    
    # Fallback if something went wrong
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    
    return best_solution
