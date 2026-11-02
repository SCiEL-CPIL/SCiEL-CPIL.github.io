import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Helper function for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Build incidence structures
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
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    # Precompute block indices as numpy arrays
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # Vectorized greedy with optional randomization
    def run_greedy(seed, explore_prob=0.0, weight_temp=0.0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            if explore_prob > 0 and random.random() < explore_prob:
                # Explore: pick from blocks with gain >= max_gain - 3
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            elif weight_temp > 0:
                # Weighted random selection (Rödl nibble style)
                # Weight proportional to gain^temp for diversity
                valid_mask = (~is_selected) & (gain > 0)
                weights = np.zeros(num_k_blocks, dtype=np.float64)
                weights[valid_mask] = gain[valid_mask] ** weight_temp
                total_w = weights.sum()
                if total_w <= 0:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
                else:
                    r = random.random() * total_w
                    cumsum = np.cumsum(weights)
                    b_idx = int(np.searchsorted(cumsum, r))
                    if b_idx >= num_k_blocks:
                        b_idx = num_k_blocks - 1
                    if is_selected[b_idx] or gain[b_idx] <= 0:
                        b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            # Vectorized gain update using bincount
            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            # Collect all blocks that contain any of the newly covered t-subsets
            blocks_to_decrement = np.concatenate(
                [ts_to_blocks_np[int(ts_idx)] for ts_idx in newly_covered]
            ) if len(newly_covered) > 0 else np.array([], dtype=np.int32)

            if len(blocks_to_decrement) > 0:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    # Run multiple randomized greedy with varying strategies
    num_runs = 200
    for i in range(1, num_runs):
        if time.time() > deadline - 20.0:
            break

        # Vary strategy: explore, weighted random, or standard random
        if i < 60:
            sol = run_greedy(i, explore_prob=0.15)
        elif i < 120:
            sol = run_greedy(i, weight_temp=1.5)
        elif i < 160:
            sol = run_greedy(i, explore_prob=0.3, weight_temp=1.0)
        else:
            sol = run_greedy(i, explore_prob=0.5)

        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Rödl nibble style: random greedy with probability-based selection
    def rodl_nibble(seed, p=0.5):
        random.seed(seed)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        # Phase 1: Randomly select blocks with probability proportional to coverage
        remaining_uncovered = num_t_subsets
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            if remaining_uncovered == 0:
                break

            # Pick a random block
            b_idx = random.randint(0, num_k_blocks - 1)
            if is_selected[b_idx]:
                continue

            bts = block_to_ts_np[b_idx]
            newly_covered = bts[is_uncovered[bts]]

            if len(newly_covered) == 0:
                continue

            # Accept with probability p (nibble)
            if random.random() < p:
                sol.append(list(all_k_blocks[b_idx]))
                is_selected[b_idx] = True
                is_uncovered[newly_covered] = False
                remaining_uncovered -= len(newly_covered)

        # Phase 2: Greedy completion for uncovered t-subsets
        if remaining_uncovered > 0:
            gain = np.zeros(num_k_blocks, dtype=np.int32)
            for j in range(num_k_blocks):
                if not is_selected[j]:
                    bts = block_to_ts_np[j]
                    gain[j] = int(np.sum(is_uncovered[bts]))

            while True:
                if time.time() > deadline - 5.0:
                    break
                max_gain = int(np.max(gain))
                if max_gain <= 0:
                    break

                candidates = np.where((gain == max_gain) & (~is_selected))[0]
                if len(candidates) == 0:
                    break

                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
                sol.append(list(all_k_blocks[b_idx]))
                is_selected[b_idx] = True
                gain[b_idx] = -1

                bts = block_to_ts_np[b_idx]
                newly_covered = bts[is_uncovered[bts]]
                if len(newly_covered) == 0:
                    continue
                is_uncovered[newly_covered] = False

                blocks_to_decrement = np.concatenate(
                    [ts_to_blocks_np[int(ts_idx)] for ts_idx in newly_covered]
                ) if len(newly_covered) > 0 else np.array([], dtype=np.int32)
                if len(blocks_to_decrement) > 0:
                    counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                    gain -= counts
                    gain[is_selected] = -1000

        return sol

    # Run Rödl nibble variants
    for i in range(30):
        if time.time() > deadline - 10.0:
            break
        p_val = 0.3 + 0.05 * i
        sol = rodl_nibble(1000 + i, p=p_val)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Local Search: Remove redundant blocks
    def optimize_redundancy(blocks):
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
                if time.time() > deadline - 2.0:
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

    def local_search_swap(blocks):
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

        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 2.0:
                return blocks, improved

            unique_ts = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)

            if not unique_ts:
                continue

            # Find candidate blocks that cover ALL unique_ts
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

            blocks[i] = new_block
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)

            # Update cover counts
            for ts_idx in block_ts_indices[i]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            block_ts_indices[i] = block_to_ts[new_block_idx]

            improved = True

        return blocks, improved

    def local_search_2exchange(blocks):
        if len(blocks) < 2:
            return blocks, False

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

        improved = False
        n = len(blocks)

        for i in range(n):
            if time.time() > deadline - 1.5:
                return blocks, improved
            for j in range(i + 1, n):
                if time.time() > deadline - 1.5:
                    return blocks, improved

                unique_ts_i = []
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        unique_ts_i.append(ts_idx)

                unique_ts_j = []
                for ts_idx in block_ts_indices[j]:
                    if cover_count[ts_idx] == 1:
                        unique_ts_j.append(ts_idx)

                if not unique_ts_i and not unique_ts_j:
                    continue

                all_unique = unique_ts_i + unique_ts_j
                if not all_unique:
                    continue

                # Try to find a single block covering all unique ts from both
                candidates = set(ts_to_blocks[all_unique[0]])
                for ts_idx in all_unique[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break

                if candidates:
                    valid = [c for c in candidates if c not in sol_block_indices]
                    if valid:
                        new_block_idx = random.choice(valid)
                        new_block = list(all_k_blocks[new_block_idx])

                        for ts_idx in block_ts_indices[i]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_ts_indices[j]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[new_block_idx]:
                            cover_count[ts_idx] += 1

                        sol_block_indices.discard(get_block_idx(blocks[i]))
                        sol_block_indices.discard(get_block_idx(blocks[j]))
                        sol_block_indices.add(new_block_idx)

                        blocks.pop(j)
                        blocks[i] = new_block
                        block_ts_indices.pop(j)
                        block_ts_indices[i] = block_to_ts[new_block_idx]
                        improved = True
                        break

                if improved:
                    break

        return blocks, improved

    # Run local search iterations
    for _ in range(50):
        if time.time() > deadline - 3.0:
            break
        best_solution, improved = local_search_swap(best_solution)
        if improved:
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            best_solution, improved2 = local_search_2exchange(best_solution)
            if improved2:
                best_solution = optimize_redundancy(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                break

    # === Genetic Algorithm with Multi-Objective Fitness ===
    def ga_fitness(blocks):
        """Multi-objective fitness: penalize block count and uncovered t-subsets."""
        if not blocks:
            return (num_t_subsets, len(blocks))
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            else:
                for ts in itertools.combinations(b, t):
                    ts_idx = ts_index_map.get(ts)
                    if ts_idx is not None:
                        cover_count[ts_idx] += 1
        
        uncovered = int(np.sum(cover_count == 0))
        # Fitness: primarily minimize uncovered, secondarily minimize block count
        return (uncovered, len(blocks))

    def ga_crossover(parent1, parent2, seed=42):
        """Uniform crossover preserving block validity."""
        rng = random.Random(seed)
        n1, n2 = len(parent1), len(parent2)
        n = max(n1, n2)
        
        child = []
        seen_blocks = set()
        
        for i in range(n):
            if rng.random() < 0.5:
                if i < n1 and parent1[i] not in seen_blocks:
                    child.append(parent1[i])
                    seen_blocks.add(parent1[i])
            else:
                if i < n2 and parent2[i] not in seen_blocks:
                    child.append(parent2[i])
                    seen_blocks.add(parent2[i])
        
        # Fill remaining with random blocks if needed
        while len(child) < n:
            b_idx = rng.randint(0, num_k_blocks - 1)
            block = list(all_k_blocks[b_idx])
            if block not in seen_blocks:
                child.append(block)
                seen_blocks.add(block)
        
        return child

    def ga_mutation(blocks, mutation_rate=0.1, seed=42):
        """Mutation: replace some blocks with random ones, add/remove blocks."""
        rng = random.Random(seed)
        n = len(blocks)
        
        # Remove some blocks
        num_remove = int(n * mutation_rate * 0.5)
        for _ in range(num_remove):
            if blocks:
                idx = rng.randint(0, len(blocks) - 1)
                blocks.pop(idx)
        
        # Add some blocks
        num_add = int(n * mutation_rate * 0.5)
        seen = set(blocks)
        for _ in range(num_add):
            b_idx = rng.randint(0, num_k_blocks - 1)
            block = list(all_k_blocks[b_idx])
            if block not in seen:
                blocks.append(block)
                seen.add(block)
        
        return blocks

    def ga_local_polish(blocks, max_iters=5):
        """Apply local search to a GA offspring."""
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            blocks, improved = local_search_swap(blocks)
            if not improved:
                break
            blocks = optimize_redundancy(blocks)
        return blocks

    def run_genetic_algorithm(initial_pops, num_generations=20, pop_size=8, seed=999):
        """Genetic algorithm with multi-objective fitness."""
        rng = random.Random(seed)
        
        # Initialize population
        population = []
        for sol in initial_pops:
            population.append(list(sol))
        
        # Pad population with randomized solutions
        while len(population) < pop_size:
            sol = run_greedy(rng.randint(0, 10000), explore_prob=0.3)
            if sol is not None:
                population.append(sol)
            else:
                population.append(list(best_solution))
        
        best_pop_fitness = None
        best_pop = None
        
        for gen in range(num_generations):
            if time.time() > deadline - 10.0:
                break
            
            # Evaluate fitness
            fitnesses = []
            for ind in population:
                fit = ga_fitness(ind)
                fitnesses.append(fit)
            
            # Sort by fitness (uncovered first, then block count)
            sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i])
            
            # Track best
            if best_pop_fitness is None or fitnesses[sorted_indices[0]] < best_pop_fitness:
                best_pop_fitness = fitnesses[sorted_indices[0]]
                best_pop = list(population[sorted_indices[0]])
            
            # Elitism: keep top 2
            new_population = [list(population[sorted_indices[0]]), list(population[sorted_indices[1]])]
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Tournament selection
                p1_idx = sorted_indices[rng.randint(0, min(3, len(sorted_indices)-1))]
                p2_idx = sorted_indices[rng.randint(0, min(3, len(sorted_indices)-1))]
                
                child = ga_crossover(population[p1_idx], population[p2_idx], seed=seed + gen * 100 + len(new_population))
                child = ga_mutation(child, mutation_rate=0.15, seed=seed + gen * 200 + len(new_population))
                child = ga_local_polish(child, max_iters=3)
                
                new_population.append(child)
            
            population = new_population
        
        return best_pop, best_pop_fitness

    # Run GA if we have time
    if time.time() < deadline - 30.0:
        # Create initial population from best solutions found so far
        initial_pops = [best_solution]
        
        # Add some diverse solutions
        for i in range(5):
            if time.time() > deadline - 25.0:
                break
            sol = run_greedy(5000 + i, explore_prob=0.4)
            if sol is not None:
                initial_pops.append(sol)
        
        ga_result, ga_fitness_val = run_genetic_algorithm(initial_pops, num_generations=15, pop_size=6, seed=777)
        
        if ga_result is not None:
            ga_result = optimize_redundancy(ga_result)
            if len(ga_result) < best_len:
                best_len = len(ga_result)
                best_solution = ga_result

    # Final cleanup
    best_solution = optimize_redundancy(best_solution)

    return best_solution
