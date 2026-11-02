"""Covering design C(19,8,5) - genetic algorithm with multi-objective fitness."""
import itertools
import time
import random
import numpy as np


def _precompute(v, k, t):
    """Precompute all t-subsets, all k-blocks, and their relationships."""
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    block_to_idx = {b: i for i, b in enumerate(all_kblocks)}
    
    # For each block, which t-sets does it cover?
    block_to_tsets = []
    for block in all_kblocks:
        tsets_in_block = [tset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_tsets.append(tsets_in_block)
    
    # For each t-set, which blocks cover it?
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, tsets_list in enumerate(block_to_tsets):
        for ts_idx in tsets_list:
            tset_to_blocks[ts_idx].append(b_idx)
    
    # Convert to numpy arrays
    block_to_tsets_np = [np.array(lst, dtype=np.int32) for lst in block_to_tsets]
    tset_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tset_to_blocks]
    
    return all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
           tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks


def _greedy_cover(v, k, t, precomp, seed=0, explore_prob=0.0, deadline=None):
    """Greedy set-cover with vectorized gain computation."""
    if deadline is None:
        deadline = time.time() + 100
    
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    rng = random.Random(seed)
    
    gain = np.full(num_kblocks, len(block_to_tsets[0]), dtype=np.int32)
    is_uncovered = np.full(num_tsets, True, dtype=bool)
    is_selected = np.full(num_kblocks, False, dtype=bool)
    sol = []
    
    while True:
        if time.time() > deadline:
            return None
        
        max_gain = int(np.max(gain))
        if max_gain <= 0:
            break
        
        candidates = np.where((gain == max_gain) & (~is_selected))[0]
        if len(candidates) == 0:
            break
        
        if explore_prob > 0 and rng.random() < explore_prob:
            explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
            if len(explore_cands) > 0:
                b_idx = int(explore_cands[rng.randint(0, len(explore_cands) - 1)])
            else:
                b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
        else:
            b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
        
        sol.append(list(all_kblocks[b_idx]))
        is_selected[b_idx] = True
        gain[b_idx] = -1
        
        bts = block_to_tsets_np[b_idx]
        newly_covered_mask = is_uncovered[bts]
        newly_covered = bts[newly_covered_mask]
        
        if len(newly_covered) == 0:
            continue
        
        is_uncovered[newly_covered] = False
        
        blocks_to_decrement = []
        for ts_idx in newly_covered:
            blocks_to_decrement.extend(tset_to_blocks[ts_idx])
        
        if blocks_to_decrement:
            counts = np.bincount(blocks_to_decrement, minlength=num_kblocks)
            gain -= counts
            gain[is_selected] = -1000
    
    return sol


def _local_search(blocks, precomp, deadline):
    """Local search to reduce block count via redundancy removal and swaps."""
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    blocks = [list(b) for b in blocks]
    
    cover_count = np.zeros(num_tsets, dtype=np.int32)
    block_ts_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            block_ts_indices.append(block_to_tsets[idx])
        else:
            ts_list = [tset_to_idx[ts] for ts in itertools.combinations(b, t)]
            block_ts_indices.append(ts_list)
        for ts_idx in block_ts_indices[-1]:
            cover_count[ts_idx] += 1
    
    # Phase 1: Remove redundant blocks
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            redundant = True
            for ts_idx in block_ts_indices[bi]:
                if cover_count[ts_idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts_idx in block_ts_indices[bi]:
                    cover_count[ts_idx] -= 1
                blocks.pop(bi)
                block_ts_indices.pop(bi)
                improved = True
                break
    
    # Phase 2: Try swaps to reduce block count
    sol_block_indices = set()
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            sol_block_indices.add(idx)
    
    rng = random.Random(42)
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            
            unique_ts = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
            
            if not unique_ts:
                for ts_idx in block_ts_indices[i]:
                    cover_count[ts_idx] -= 1
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved = True
                break
            
            candidates = set(tset_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(tset_to_blocks[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            if not valid_candidates:
                continue
            
            new_block_idx = rng.choice(valid_candidates)
            new_block = list(all_kblocks[new_block_idx])
            
            old_block = blocks[i]
            old_block_idx = block_to_idx.get(tuple(old_block), -1)
            
            new_unique_count = 0
            for ts_idx in block_to_tsets[new_block_idx]:
                if cover_count[ts_idx] == 1:
                    new_unique_count += 1
            
            if new_unique_count >= len(unique_ts):
                blocks[i] = new_block
                sol_block_indices.discard(old_block_idx)
                sol_block_indices.add(new_block_idx)
                
                for ts_idx in block_to_tsets[old_block_idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_tsets[new_block_idx]:
                    cover_count[ts_idx] += 1
                
                block_ts_indices[i] = block_to_tsets[new_block_idx]
                improved = True
                break
        
        if not improved:
            break
    
    return blocks


def _fitness(blocks, precomp):
    """Multi-objective fitness: penalize number of blocks and uncovered t-subsets."""
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    cover_count = np.zeros(num_tsets, dtype=np.int32)
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            for ts_idx in block_to_tsets[idx]:
                cover_count[ts_idx] += 1
        else:
            for ts in itertools.combinations(b, 5):
                ts_idx = tset_to_idx[ts]
                cover_count[ts_idx] += 1
    
    uncovered = np.sum(cover_count == 0)
    # Fitness: lower is better. Penalize uncovered heavily.
    return len(blocks) + 100 * uncovered


def _crossover(parent1, parent2, precomp, rng):
    """Crossover: take blocks from parent1 that cover unique t-sets, fill rest from parent2."""
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    # Build coverage map for parent1
    cover_count_p1 = np.zeros(num_tsets, dtype=np.int32)
    for b in parent1:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            for ts_idx in block_to_tsets[idx]:
                cover_count_p1[ts_idx] += 1
    
    # Start with blocks from parent1
    child = [list(b) for b in parent1]
    
    # Find uncovered t-sets in child
    uncovered_ts = np.where(cover_count_p1 == 0)[0]
    
    # Try to fill gaps using blocks from parent2
    for ts_idx in uncovered_ts:
        # Find a block from parent2 that covers this t-set
        candidates = [b for b in parent2 if ts_idx in block_to_tsets.get(block_to_idx.get(tuple(b), -1), [])]
        if candidates:
            child.append(list(candidates[0]))
        else:
            # Pick a random block that covers this t-set
            possible_blocks = tset_to_blocks[ts_idx]
            if possible_blocks:
                b_idx = possible_blocks[rng.randint(0, len(possible_blocks) - 1)]
                child.append(list(all_kblocks[b_idx]))
    
    return child


def _mutate(blocks, precomp, rng, mutation_rate=0.1):
    """Mutation: randomly replace some blocks with new ones."""
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    mutated = [list(b) for b in blocks]
    
    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            # Replace with a random block
            new_idx = rng.randint(0, num_kblocks - 1)
            mutated[i] = list(all_kblocks[new_idx])
    
    return mutated


def _genetic_algorithm(v, k, t, precomp, initial_pop, deadline, pop_size=20, generations=50):
    """Genetic algorithm with multi-objective fitness."""
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    rng = random.Random(123)
    
    # Initialize population
    population = [list(b) for b in initial_pop]
    # Add some random individuals
    while len(population) < pop_size:
        random_blocks = []
        for _ in range(rng.randint(200, 400)):
            b_idx = rng.randint(0, num_kblocks - 1)
            random_blocks.append(list(all_kblocks[b_idx]))
        population.append(random_blocks)
    
    best_solution = min(population, key=lambda b: _fitness(b, precomp))
    best_fitness = _fitness(best_solution, precomp)
    
    for gen in range(generations):
        if time.time() > deadline:
            break
        
        # Evaluate fitness
        fitnesses = [_fitness(ind, precomp) for ind in population]
        
        # Find best in this generation
        gen_best_idx = np.argmin(fitnesses)
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_solution = [list(b) for b in population[gen_best_idx]]
        
        # Selection: tournament selection
        def tournament_select(pop, fit, k=3):
            indices = rng.sample(range(len(pop)), min(k, len(pop)))
            best_idx = min(indices, key=lambda i: fit[i])
            return [list(b) for b in pop[best_idx]]
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        sorted_indices = np.argsort(fitnesses)
        for idx in sorted_indices[:max(2, pop_size // 5)]:
            new_population.append([list(b) for b in population[idx]])
        
        # Crossover and mutation to fill rest
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            
            child = _crossover(p1, p2, precomp, rng)
            child = _mutate(child, precomp, rng, mutation_rate=0.05)
            
            new_population.append(child)
        
        population = new_population
    
    return best_solution


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precomputation
    precomp = _precompute(v, k, t)
    
    # Phase 1: Multiple greedy restarts to get diverse initial solutions
    initial_solutions = []
    best_solution = None
    best_len = float('inf')
    
    num_greedy_runs = 30
    for i in range(num_greedy_runs):
        if time.time() > deadline - 60:
            break
        
        explore = 0.0 if i < 10 else (0.1 if i < 20 else 0.2)
        sol = _greedy_cover(v, k, t, precomp, seed=i, explore_prob=explore, deadline=deadline - 60)
        if sol is not None:
            initial_solutions.append(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
    
    if best_solution is None:
        best_solution = [list(precomp[1][0])]
    
    # Phase 2: Local search on best greedy solution
    if time.time() < deadline - 30:
        best_solution = _local_search(best_solution, precomp, deadline - 30)
        best_len = len(best_solution)
    
    # Phase 3: Genetic algorithm
    if time.time() < deadline - 10 and len(initial_solutions) >= 2:
        # Use top solutions as initial population
        initial_solutions.sort(key=len)
        ga_pop = initial_solutions[:min(10, len(initial_solutions))]
        
        ga_result = _genetic_algorithm(v, k, t, precomp, ga_pop, deadline - 5, pop_size=15, generations=30)
        if ga_result is not None and len(ga_result) < best_len:
            best_solution = ga_result
            best_len = len(best_solution)
    
    # Phase 4: Final local search
    if time.time() < deadline - 2:
        best_solution = _local_search(best_solution, precomp, deadline - 1)
    
    return best_solution
