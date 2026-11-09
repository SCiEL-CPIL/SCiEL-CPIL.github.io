import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # 10s safety margin
    
    points = list(range(v))
    
    # 1. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets and map to indices
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # 3. Precompute block -> t-subsets mapping
    # block_subs[i] is a list of t-subset indices covered by block i
    block_subs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_subs.append(subs)
        
    # 4. Precompute t-subset -> blocks mapping
    # sub_blocks[j] is a list of block indices that cover t-subset j
    sub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_subs):
        for si in subs:
            sub_blocks[si].append(i)

    def run_greedy(random_seed=None, noise_scale=0.0):
        if random_seed is not None:
            random.seed(random_seed)
            
        # gain[i] = number of currently uncovered t-subsets covered by block i
        gain = np.array([len(subs) for subs in block_subs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=bool)
        uncovered_count = N_t
        selected_blocks = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            # Find best block
            if noise_scale > 0:
                noise = np.random.uniform(0, noise_scale, size=N_blocks)
                scores = gain + noise
            else:
                scores = gain
                
            best_idx = np.argmax(scores)
            
            if gain[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            gain[best_idx] = -1  # Disable this block
            
            # Update gains for all blocks that share subsets with best_idx
            for si in block_subs[best_idx]:
                if not is_covered[si]:
                    is_covered[si] = True
                    uncovered_count -= 1
                    for b_idx in sub_blocks[si]:
                        if gain[b_idx] > 0:
                            gain[b_idx] -= 1
                            
        return selected_blocks

    def try_reduce(block_indices):
        if not block_indices:
            return block_indices
            
        cov_count = np.zeros(N_t, dtype=np.int32)
        for b_idx in block_indices:
            for si in block_subs[b_idx]:
                cov_count[si] += 1
                
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            new_blocks = []
            for b_idx in block_indices:
                redundant = True
                for si in block_subs[b_idx]:
                    if cov_count[si] <= 1:
                        redundant = False
                        break
                if redundant:
                    changed = True
                    for si in block_subs[b_idx]:
                        cov_count[si] -= 1
                else:
                    new_blocks.append(b_idx)
            if changed:
                block_indices = new_blocks
                
        return block_indices

    def compute_coverage(block_indices):
        """Compute coverage count for each t-subset given a set of blocks."""
        cov_count = np.zeros(N_t, dtype=np.int32)
        for b_idx in block_indices:
            for si in block_subs[b_idx]:
                cov_count[si] += 1
        return cov_count

    def is_valid_cover(block_indices):
        """Check if block_indices covers all t-subsets."""
        cov_count = compute_coverage(block_indices)
        return np.all(cov_count > 0)

    def local_search(block_indices):
        """
        Improve solution by trying block swaps and removals.
        """
        if not block_indices:
            return block_indices
            
        block_set = set(block_indices)
        cov_count = compute_coverage(block_indices)
        current_blocks = list(block_indices)
        
        improved = True
        while improved and time.time() < DEADLINE - 15:
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                # Check redundancy
                redundant = True
                for si in block_subs[b_idx]:
                    if cov_count[si] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    for si in block_subs[b_idx]:
                        cov_count[si] -= 1
                    block_set.discard(b_idx)
                    improved = True
                    continue
                
                # Find critical subsets (only covered by this block)
                critical_subs = [si for si in block_subs[b_idx] if cov_count[si] == 1]
                
                if critical_subs:
                    # Try to find a replacement block that covers all critical_subs
                    # and is not already in the solution
                    best_replacement = None
                    best_gain = 0
                    
                    # Sample some candidate blocks for speed
                    candidates = list(block_set.symmetric_difference(
                        set(sub_blocks[critical_subs[0]])
                    ))
                    if not candidates:
                        candidates = list(range(N_blocks))
                    
                    # Limit candidates for speed
                    max_candidates = min(len(candidates), 500)
                    sampled = random.sample(candidates, max_candidates) if len(candidates) > max_candidates else candidates
                    
                    for cand in sampled:
                        if cand == b_idx or cand in block_set:
                            continue
                        # Check if cand covers all critical_subs
                        cand_set = set(block_subs[cand])
                        if all(si in cand_set for si in critical_subs):
                            # Compute how many additional subsets this would cover
                            extra = 0
                            for si in block_subs[cand]:
                                if si not in block_set and cov_count[si] == 0:
                                    extra += 1
                            if extra > best_gain:
                                best_gain = extra
                                best_replacement = cand
                    
                    if best_replacement is not None and best_gain > 0:
                        # Perform the swap
                        # Remove old block
                        for si in block_subs[b_idx]:
                            cov_count[si] -= 1
                        block_set.discard(b_idx)
                        
                        # Add new block
                        for si in block_subs[best_replacement]:
                            cov_count[si] += 1
                        block_set.add(best_replacement)
                        
                        new_blocks.append(best_replacement)
                        improved = True
                    else:
                        new_blocks.append(b_idx)
                else:
                    new_blocks.append(b_idx)
            
            if improved:
                current_blocks = new_blocks
                
        return current_blocks

    def genetic_swap(parent1, parent2, mutation_rate=0.3):
        """
        Create child by combining two parent solutions.
        - Start with union of both parents
        - Reduce to remove redundancies
        - Apply random mutations (swap blocks)
        """
        if time.time() > DEADLINE - 15:
            return try_reduce(parent1)
        
        # Combine: take union, then reduce
        combined = list(set(parent1) | set(parent2))
        combined = try_reduce(combined)
        
        if not combined:
            return try_reduce(parent1)
        
        # Apply mutations: randomly swap some blocks
        if random.random() < mutation_rate:
            n_swaps = random.randint(1, min(3, len(combined) // 4))
            for _ in range(n_swaps):
                if not combined:
                    break
                # Pick a block to potentially swap
                idx = random.randint(0, len(combined) - 1)
                b_idx = combined[idx]
                
                # Find blocks that share at least one t-subset with b_idx
                shared_blocks = set()
                for si in block_subs[b_idx]:
                    shared_blocks.update(sub_blocks[si])
                shared_blocks.discard(b_idx)
                
                if shared_blocks:
                    new_block = random.choice(list(shared_blocks))
                    # Try the swap: remove b_idx, add new_block, check validity
                    test_set = set(combined)
                    test_set.discard(b_idx)
                    test_set.add(new_block)
                    
                    if is_valid_cover(list(test_set)):
                        combined = list(test_set)
        
        combined = try_reduce(combined)
        return combined

    def genetic_mutation(block_indices, n_mutations=2):
        """
        Create child by mutating a parent solution.
        - For each mutation, try to swap a block with another
        """
        if not block_indices:
            return block_indices
        
        result = list(block_indices)
        
        for _ in range(n_mutations):
            if time.time() > DEADLINE - 15:
                break
            if not result:
                break
            
            # Pick a random block
            idx = random.randint(0, len(result) - 1)
            b_idx = result[idx]
            
            # Find candidate replacements
            shared_blocks = set()
            for si in block_subs[b_idx]:
                shared_blocks.update(sub_blocks[si])
            shared_blocks.discard(b_idx)
            
            if not shared_blocks:
                continue
            
            # Try a few random replacements
            candidates = list(shared_blocks)
            random.shuffle(candidates)
            
            for cand in candidates[:10]:
                test_set = set(result)
                test_set.discard(b_idx)
                test_set.add(cand)
                
                if is_valid_cover(list(test_set)):
                    result = list(test_set)
                    break
        
        result = try_reduce(result)
        return result

    def run_genetic_algorithm(initial_solutions, n_generations=20, pop_size=8):
        """
        Run a genetic algorithm starting from initial solutions.
        """
        if not initial_solutions:
            return []
        
        # Initialize population
        population = [list(sol) for sol in initial_solutions[:pop_size]]
        
        # Ensure minimum population size by duplicating
        while len(population) < pop_size:
            population.append(list(random.choice(initial_solutions)))
        
        best_sol = min(population, key=len)
        best_len = len(best_sol)
        
        gen = 0
        while gen < n_generations and time.time() < DEADLINE - 15:
            gen += 1
            
            # Evaluate fitness (length, shorter is better)
            population.sort(key=len)
            
            # Create next generation
            next_gen = []
            
            # Elitism: keep top few
            n_elite = max(2, pop_size // 4)
            for i in range(n_elite):
                next_gen.append(list(population[i]))
            
            # Fill rest with children
            while len(next_gen) < pop_size and time.time() < DEADLINE - 15:
                # Select parents (tournament selection)
                n_tournament = min(3, len(population))
                parents = random.sample(population, n_tournament)
                p1 = min(parents, key=len)
                
                if len(population) > 1:
                    remaining = [p for p in population if p is not p1]
                    if remaining:
                        parents2 = random.sample(remaining, min(3, len(remaining)))
                        p2 = min(parents2, key=len)
                    else:
                        p2 = p1
                else:
                    p2 = p1
                
                # Crossover with some probability, otherwise mutation
                if random.random() < 0.5 and p1 is not p2:
                    child = genetic_swap(p1, p2, mutation_rate=0.3)
                else:
                    child = genetic_mutation(p1, n_mutations=random.randint(1, 3))
                
                if child:
                    next_gen.append(child)
            
            population = next_gen
            
            # Update best
            for sol in population:
                if len(sol) < best_len:
                    best_len = len(sol)
                    best_sol = list(sol)
        
        return best_sol

    best_sol = None
    best_len = float('inf')
    
    # Phase 1: Run deterministic greedy first
    sol = run_greedy(random_seed=None)
    sol = try_reduce(sol)
    best_sol = sol
    best_len = len(sol)
    
    # Phase 2: Run several randomized greedy to build initial population
    initial_solutions = [list(best_sol)]
    attempt = 0
    max_initial = 12
    while time.time() < DEADLINE - 40.0 and attempt < max_initial:
        noise_scale = 0.1 if attempt % 2 == 0 else 0.5
        sol = run_greedy(random_seed=attempt, noise_scale=noise_scale)
        sol = try_reduce(sol)
        sol = local_search(sol)
        sol = try_reduce(sol)
        
        if sol:
            initial_solutions.append(list(sol))
            if len(sol) < best_len:
                best_len = len(sol)
                best_sol = list(sol)
        
        attempt += 1
    
    # Phase 3: Run genetic algorithm
    if time.time() < DEADLINE - 20.0 and len(initial_solutions) >= 2:
        ga_sol = run_genetic_algorithm(
            initial_solutions,
            n_generations=30,
            pop_size=8
        )
        if ga_sol and len(ga_sol) < best_len:
            best_len = len(ga_sol)
            best_sol = ga_sol
    
    # Phase 4: Final local search on best solution
    if best_sol and time.time() < DEADLINE - 10.0:
        best_sol = local_search(best_sol)
        best_sol = try_reduce(best_sol)
    
    # Final reduction
    best_sol = try_reduce(best_sol)
    
    # Convert indices back to blocks
    return [all_blocks[i] for i in best_sol]
