import numpy as np
import itertools
import time
import random
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    # Precompute numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    # Precompute subset_to_blocks as sets for fast membership and intersection in swap
    subset_to_blocks_sets = [set(blocks.tolist()) for blocks in subset_block_arrays]
    
    # Precompute block sizes (number of t-subsets per block) for tie-breaking
    block_sizes = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
    
    # Precompute for GA: for each block, the set of t-subsets it covers (as frozenset for fast ops)
    block_subsets_frozen = [frozenset(subs) for subs in block_to_subsets]
    
    def compute_coverage_counts(blocks_indices):
        """Compute coverage count array for a list of block indices."""
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            for s_idx in block_subset_arrays[b_idx]:
                cover_count[s_idx] += 1
        return cover_count
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        
        # Maintain count of remaining blocks covering each subset
        remaining_count = np.array([len(blocks) for blocks in subset_block_arrays], dtype=np.int32)
        
        selected_blocks = []
        used_blocks = np.zeros(num_blocks, dtype=bool)
        
        while True:
            if not np.any(uncovered):
                break
                
            # Find top candidates based on simple gain first
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            scores[used_blocks] = -1
            top_indices = np.argpartition(scores, -200)[-200:]
            
            best_idx = -1
            best_score = -1
            
            # For the top candidates, compute a more sophisticated score
            # Score = Gain + Sum(1/remaining_count[s] for s in uncovered & block)
            for idx in top_indices:
                if gains[idx] == 0:
                    continue
                b_subs = block_subset_arrays[idx]
                # Filter to uncovered subsets
                unc_subs = b_subs[uncovered[b_subs]]
                if len(unc_subs) == 0:
                    continue
                
                # Calculate criticality bonus
                crit_bonus = np.sum(1.0 / remaining_count[unc_subs])
                final_score = gains[idx] + 0.5 * crit_bonus + rng.uniform(0, 0.1)
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx = idx
            
            if best_idx == -1 or gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            used_blocks[best_idx] = True
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            uncovered[newly_covered] = False
            
            # Update remaining counts
            remaining_count[b_subs] -= 1
            
            # Update gains for other blocks
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    gains[b] -= 1
                    
        return selected_blocks
    
    def repair_covering(blocks_list, max_time=2.0, random_seed=None):
        """Repair a potentially invalid covering by adding blocks to cover missing subsets."""
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_list:
            for s_idx in block_subset_arrays[b_idx]:
                cover_count[s_idx] += 1
        
        uncovered_mask = (cover_count == 0)
        if not np.any(uncovered_mask):
            return list(blocks_list)
        
        result = list(blocks_list)
        result_set = set(blocks_list)
        
        deadline = time.time() + max_time
        while np.any(uncovered_mask) and time.time() < deadline:
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Find blocks that cover the most uncovered subsets
            gains = np.zeros(num_blocks, dtype=np.int32)
            for s_idx in uncovered_indices:
                for b_idx in subset_block_arrays[s_idx]:
                    if b_idx not in result_set:
                        gains[b_idx] += 1
            
            if np.max(gains) == 0:
                break
            
            noise = rng.uniform(0, 0.1, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
            
            result.append(best_idx)
            result_set.add(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered_mask[b_subs]]
            if len(newly_covered) == 0:
                continue
            
            uncovered_mask[newly_covered] = False
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    if b not in result_set:
                        gains[b] -= 1
        
        return result
    
    def crossover(parent1, parent2, rng):
        """Merge two valid coverings into a new one, then try to minimize."""
        set1 = set(parent1)
        set2 = set(parent2)
        
        common = set1 & set2
        only1 = set1 - set2
        only2 = set2 - set1
        
        child = list(common)
        
        for b in only1:
            if rng.random() < 0.5:
                child.append(b)
        for b in only2:
            if rng.random() < 0.5:
                child.append(b)
        
        return child
    
    def mutate_criticality_weighted(blocks_list, cover_count, rng, num_mutations=3):
        """Replace blocks based on criticality. Less critical blocks are more likely to be replaced."""
        result = list(blocks_list)
        result_set = set(blocks_list)
        
        if not result:
            return result
        
        for _ in range(num_mutations):
            if not result:
                break
            
            criticality = {}
            for b_idx in result:
                unique_count = 0
                for s_idx in block_subset_arrays[b_idx]:
                    if cover_count[s_idx] == 1:
                        unique_count += 1
                criticality[b_idx] = unique_count
            
            total_weight = sum(1.0 / (1.0 + c) for c in criticality.values())
            
            r = rng.random() * total_weight
            cumsum = 0.0
            selected_block = result[0]
            for b_idx in result:
                cumsum += 1.0 / (1.0 + criticality[b_idx])
                if r <= cumsum:
                    selected_block = b_idx
                    break
            
            unique_subs = []
            for s_idx in block_subset_arrays[selected_block]:
                if cover_count[s_idx] == 1:
                    unique_subs.append(s_idx)
            
            if unique_subs:
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(subset_to_blocks_sets[s_idx])
                    else:
                        candidates.intersection_update(subset_to_blocks_sets[s_idx])
                        if not candidates:
                            break
                
                if candidates:
                    candidates.discard(selected_block)
                    candidates -= result_set
                    if candidates:
                        new_block = rng.choice(list(candidates))
                        for s_idx in block_subset_arrays[selected_block]:
                            cover_count[s_idx] -= 1
                        for s_idx in block_subset_arrays[new_block]:
                            cover_count[s_idx] += 1
                        result_set.discard(selected_block)
                        result_set.add(new_block)
                        result[result.index(selected_block)] = new_block
            
            else:
                for s_idx in block_subset_arrays[selected_block]:
                    cover_count[s_idx] -= 1
                result_set.discard(selected_block)
                result.remove(selected_block)
        
        return result

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
                    break
            
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
                
                unique_subs = []
                for s_idx in block_subset_arrays[old_block]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    new_block = rng.randint(0, num_blocks - 1)
                    if new_block == old_block:
                        continue
                    if new_block in current_block_set:
                        continue
                        
                    for s_idx in block_subset_arrays[old_block]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_subset_arrays[new_block]:
                        cover_count[s_idx] += 1
                    current_block_set.discard(old_block)
                    current_block_set.add(new_block)
                    current_blocks[i] = new_block
                    swaps_done += 1
                else:
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
                            new_block = rng.choice(list(candidates))
                            if new_block in current_block_set:
                                continue
                                
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
    
    def run_simulated_annealing(initial_blocks, max_time, random_seed=42):
        """
        Simulated Annealing on block selection.
        State: List of block indices.
        Energy: len(blocks) + penalty * num_uncovered
        Moves: 
          1. Remove a block (if valid, always accept; else prob accept)
          2. Swap a block with another that covers its unique subsets (if valid, always accept; else prob accept)
        """
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in block_subset_arrays[b_idx]:
                cover_count[t_idx] += 1
        
        current_energy = len(current_blocks) + 1000.0 * np.sum(cover_count == 0)
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        T = 50.0
        cooling_rate = 0.99
        min_T = 0.1
        penalty = 1000.0
        
        rng = random.Random(random_seed)
        start_time = time.time()
        
        while time.time() - start_time < max_time and T > min_T:
            if len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                valid_removal = True
                new_uncovered = 0
                for t_idx in block_subset_arrays[b_idx]:
                    if cover_count[t_idx] == 1:
                        valid_removal = False
                        new_uncovered += 1
                
                if valid_removal:
                    for t_idx in block_subset_arrays[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                else:
                    delta_E = 1.0 + penalty * new_uncovered
                    if rng.random() < math.exp(-delta_E / T):
                        for t_idx in block_subset_arrays[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        current_energy += delta_E
            else:
                # If empty, add a random block
                if num_blocks > 0:
                    new_b = rng.randint(0, num_blocks - 1)
                    current_blocks.append(new_b)
                    for t_idx in block_subset_arrays[new_b]:
                        cover_count[t_idx] += 1
                    current_energy += 1.0 - penalty * np.sum(cover_count == 0) + penalty
            
            # Swap move
            if len(current_blocks) > 1 and rng.random() < 0.5:
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                new_b = rng.randint(0, num_blocks - 1)
                if new_b != old_b and new_b not in current_blocks:
                    # Tentative swap
                    for t_idx in block_subset_arrays[old_b]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_subset_arrays[new_b]:
                        cover_count[t_idx] += 1
                    
                    new_uncovered = np.sum(cover_count == 0)
                    new_energy = len(current_blocks) + penalty * new_uncovered
                    delta_E = new_energy - current_energy
                    
                    if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                        current_blocks[i] = new_b
                        current_energy = new_energy
                    else:
                        # Revert
                        for t_idx in block_subset_arrays[new_b]:
                            cover_count[t_idx] -= 1
                        for t_idx in block_subset_arrays[old_b]:
                            cover_count[t_idx] += 1
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)
            
            T *= cooling_rate
        
        return best_blocks

    def run_genetic_algorithm(initial_pop, population_size, num_generations, time_limit, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        start_time = time.time()
        
        population = []
        for blocks in initial_pop:
            cover_count = compute_coverage_counts(blocks)
            if np.all(cover_count > 0):
                population.append((list(blocks), cover_count.copy()))
        
        while len(population) < population_size and time.time() < start_time + time_limit:
            if population:
                base, _ = population[rng.integers(0, len(population))]
                perturbed = list(base)
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
            
            new_population = []
            
            for _ in range(population_size):
                candidates = rng.choice(len(population), size=3, replace=False)
                winner = min(candidates, key=lambda i: len(population[i][0]))
                new_population.append((list(population[winner][0]), population[winner][1].copy()))
            
            for i in range(0, len(new_population) - 1, 2):
                parent1 = new_population[i][0]
                parent2 = new_population[i + 1][0]
                
                child = crossover(parent1, parent2, rng)
                
                cover_count = compute_coverage_counts(child)
                if not np.all(cover_count > 0):
                    child = repair_covering(child, max_time=1.0, random_seed=int(rng.integers(0, 2**31)))
                    cover_count = compute_coverage_counts(child)
                
                if cover_count is not None and np.all(cover_count > 0):
                    child = mutate_criticality_weighted(child, cover_count, rng, num_mutations=2)
                    cover_count = compute_coverage_counts(child)
                
                if cover_count is not None and np.all(cover_count > 0):
                    new_population[i] = (child, cover_count)
                
                if len(new_population[i][0]) > len(parent1):
                    new_population[i] = (list(parent1), compute_coverage_counts(parent1))
            
            if population:
                old_best = min(population, key=lambda x: len(x[0]))
                if len(new_population[0][0]) > len(old_best[0]):
                    new_population[0] = (list(old_best[0]), old_best[1].copy())
            
            gen_best = min(new_population, key=lambda x: len(x[0]))
            if len(gen_best[0]) < best_len:
                best_len = len(gen_best[0])
                best_blocks = gen_best[0]
                best_count = gen_best[1]
            
            population = new_population
        
        return best_blocks

    best_blocks = None
    best_len = float('inf')
    best_coverage = None
    
    # Generate initial diverse population using multiple greedy runs
    initial_solutions = []
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        greedy_res = run_greedy_full(seed)
        coverage = compute_coverage_counts(greedy_res)
        min_res, min_cov = minimize_with_swaps(greedy_res, coverage, time_limit=1.5, random_seed=seed)
        if np.all(min_cov > 0):
            initial_solutions.append(min_res)
            if len(min_res) < best_len:
                best_len = len(min_res)
                best_blocks = min_res
                best_coverage = min_cov
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Genetic Algorithm
        if initial_solutions and time.time() < deadline - 15:
            ga_time = min(8.0, deadline - time.time() - 5)
            ga_result = run_genetic_algorithm(
                initial_solutions, 
                population_size=8, 
                num_generations=10, 
                time_limit=ga_time,
                random_seed=trial + 100
            )
            if ga_result is not None:
                ga_cov = compute_coverage_counts(ga_result)
                if np.all(ga_cov > 0) and len(ga_result) < best_len:
                    best_len = len(ga_result)
                    best_blocks = ga_result
                    best_coverage = ga_cov
                    initial_solutions[0] = ga_result
        
        # Strategy 2: Full Greedy + Local Search
        if time.time() < deadline - 10:
            greedy_res = run_greedy_full(trial + 10)
            coverage = compute_coverage_counts(greedy_res)
            
            min_res, min_cov = minimize_with_swaps(greedy_res, coverage, time_limit=2.0, random_seed=trial + 10)
            
            if len(min_res) < best_len:
                best_len = len(min_res)
                best_blocks = min_res
                best_coverage = min_cov
                if len(initial_solutions) < 5:
                    initial_solutions.append(min_res)
                elif random.random() < 0.3:
                    initial_solutions[random.randint(0, 4)] = min_res
        
        # Strategy 3: Simulated Annealing (New from Inspiration)
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 15:
            sa_time = min(10.0, deadline - time.time() - 5)
            sa_result = run_simulated_annealing(best_blocks, sa_time, random_seed=trial + 777)
            if sa_result:
                sa_cov = compute_coverage_counts(sa_result)
                if np.all(sa_cov > 0) and len(sa_result) < best_len:
                    best_len = len(sa_result)
                    best_blocks = sa_result
                    best_coverage = sa_cov
        
        # Strategy 4: Perturb Best + Repair + Local Search
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 10:
            perturbed = list(best_blocks)
            num_remove = min(max(2, len(perturbed) // 15), len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_subset_arrays[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                # Simple greedy repair
                repair_res = []
                current_uncovered = uncovered_mask.copy()
                while np.any(current_uncovered) and time.time() < deadline - 5:
                    gains = np.zeros(num_blocks, dtype=np.int32)
                    unc_idx = np.where(current_uncovered)[0]
                    for s_idx in unc_idx:
                        for b_idx in subset_block_arrays[s_idx]:
                            if b_idx not in blocks_to_remove and b_idx not in repair_res:
                                gains[b_idx] += 1
                    if np.max(gains) == 0:
                        break
                    best_idx = int(np.argmax(gains))
                    repair_res.append(best_idx)
                    b_subs = block_subset_arrays[best_idx]
                    newly_covered = b_subs[current_uncovered[b_subs]]
                    current_uncovered[newly_covered] = False
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                new_coverage = compute_coverage_counts(new_set)
                
                min_res2, min_cov2 = minimize_with_swaps(new_set, new_coverage, time_limit=2.0, random_seed=trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2
                    best_coverage = min_cov2
        
        # Strategy 5: Pure swap-based diversification
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 5:
            swap_res, swap_cov = minimize_with_swaps(best_blocks, best_coverage, time_limit=3.0, random_seed=trial + 5000)
            
            if len(swap_res) < best_len:
                best_len = len(swap_res)
                best_blocks = swap_res
                best_coverage = swap_cov

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        coverage = compute_coverage_counts(greedy_res)
        best_blocks, best_coverage = minimize_with_swaps(greedy_res, coverage, time_limit=5.0, random_seed=0)
        
    return [list(all_blocks[i]) for i in best_blocks]
