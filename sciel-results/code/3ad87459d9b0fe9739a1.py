import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

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

    # 1. Precomputation - optimized
    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Generate all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Map t-subset to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures efficiently
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    block_to_ts_np = []
    
    # Precompute bitsets for faster coverage checking
    # Each block gets a bitset of which t-subsets it covers
    # We'll use numpy arrays of uint64 for bitsets
    bits_per_ts = 64
    num_words = (num_t_subsets + bits_per_ts - 1) // bits_per_ts
    
    block_bitsets = np.zeros((num_k_blocks, num_words), dtype=np.uint64)
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
            word_idx = idx // bits_per_ts
            bit_idx = idx % bits_per_ts
            block_bitsets[j, word_idx] |= (1 << bit_idx)
        block_to_ts.append(ts_list)
        block_to_ts_np.append(np.array(ts_list, dtype=np.int32))

    # State for Greedy initialization
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks_idx = []
    
    # Greedy Phase (Deterministic) to get a valid initial solution
    while True:
        if time.time() > deadline - 10.0:
            break
            
        max_gain = gain.max()
        if max_gain <= 0:
            break
        
        # Find all blocks with max gain, break ties by block index for determinism
        candidates = np.where(gain == max_gain)[0]
        max_idx = candidates[0]
        
        solution_blocks_idx.append(max_idx)
        is_selected[max_idx] = True
        gain[max_idx] = -1
        
        # Update gains: for each newly covered t-subset, decrement gain of all blocks covering it
        for ts_idx in block_to_ts[max_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1
                    
    best_solution_idx = solution_blocks_idx[:]
    best_len = len(solution_blocks_idx)
    
    # Fast cover count calculator
    def get_cover_counts(x):
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for j in range(num_k_blocks):
            if x[j]:
                counts[block_to_ts_np[j]] += 1
        return counts
    
    # Initial state from Greedy
    x = np.zeros(num_k_blocks, dtype=bool)
    for j in best_solution_idx:
        x[j] = True
        
    cover_count = get_cover_counts(x)
    
    # Optimized Local Search with better swap strategy
    def local_search_hill_climb(x, cover_count, max_iters=5000):
        x = x.copy()
        cover_count = cover_count.copy()
        current_cost = int(np.sum(x))
        best_cost = current_cost
        best_x = x.copy()
        
        no_improvement = 0
        lb = (C(v, t) + C(k, t) - 1) // C(k, t)
        
        # Maintain a list of active blocks to avoid np.where every time
        active_blocks = np.where(x)[0].tolist()
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if not active_blocks:
                break
                
            improved = False
            random.shuffle(active_blocks)
            
            for j in active_blocks:
                # Vectorized check: is cover_count[ts] > 1 for all ts in block j?
                if np.all(cover_count[block_to_ts_np[j]] > 1):
                    # Remove block j
                    x[j] = False
                    cover_count[block_to_ts_np[j]] -= 1
                    current_cost -= 1
                    active_blocks.remove(j)
                    improved = True
                    break
                    
            if improved:
                no_improvement = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
                    if best_cost == lb:
                        return best_x, best_cost
            else:
                no_improvement += 1
                if no_improvement > 50:
                    break
                    
        return best_x, best_cost

    def local_search_with_swap(x, cover_count, max_iters=2000):
        x = x.copy()
        cover_count = cover_count.copy()
        current_cost = int(np.sum(x))
        best_cost = current_cost
        best_x = x.copy()
        lb = (C(v, t) + C(k, t) - 1) // C(k, t)
        
        active_blocks = np.where(x)[0].tolist()
        inactive_blocks = np.where(~x)[0].tolist()
        
        # Precompute set versions of block_to_ts for faster subset checks
        block_to_ts_set = [set(block_to_ts_np[j]) for j in range(num_k_blocks)]
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if not active_blocks:
                break
                
            # 1. Try removals (fast path)
            removed_any = False
            random.shuffle(active_blocks)
            for j in active_blocks:
                if np.all(cover_count[block_to_ts_np[j]] > 1):
                    x[j] = False
                    cover_count[block_to_ts_np[j]] -= 1
                    current_cost -= 1
                    active_blocks.remove(j)
                    inactive_blocks.append(j)
                    removed_any = True
                    break
                    
            if removed_any:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
                    if best_cost == lb:
                        return best_x, best_cost
                continue
                
            # 2. Try 1-swap (remove j, add j')
            if not inactive_blocks:
                break
                
            # Pick a random active block to try to swap
            j = random.choice(active_blocks)
            unique_ts_j = set(ts_idx for ts_idx in block_to_ts_np[j] if cover_count[ts_idx] == 1)
            
            if not unique_ts_j:
                if np.all(cover_count[block_to_ts_np[j]] > 1):
                     x[j] = False
                     cover_count[block_to_ts_np[j]] -= 1
                     current_cost -= 1
                     active_blocks.remove(j)
                     inactive_blocks.append(j)
                     continue
                else:
                    continue
            
            # Find a block in inactive that covers unique_ts_j
            found = False
            sample_size = min(100, len(inactive_blocks))
            sample = random.sample(inactive_blocks, sample_size)
            
            for j_prime in sample:
                if unique_ts_j.issubset(block_to_ts_set[j_prime]):
                    x[j] = False
                    cover_count[block_to_ts_np[j]] -= 1
                    x[j_prime] = True
                    cover_count[block_to_ts_np[j_prime]] += 1
                    
                    if np.min(cover_count) >= 1:
                        active_blocks.remove(j)
                        active_blocks.append(j_prime)
                        inactive_blocks.remove(j_prime)
                        inactive_blocks.append(j)
                        found = True
                        break
                    else:
                        x[j_prime] = False
                        cover_count[block_to_ts_np[j_prime]] -= 1
                        x[j] = True
                        cover_count[block_to_ts_np[j]] += 1
                        
            if found:
                continue
                
            # 3. Try 2-exchange (remove j1, j2, add j3)
            if len(active_blocks) < 2:
                break
                
            critical_blocks = []
            for j in active_blocks:
                if not np.all(cover_count[block_to_ts_np[j]] > 1):
                    critical_blocks.append(j)
                    
            if len(critical_blocks) < 2:
                continue
                
            j1 = random.choice(critical_blocks)
            j2_candidates = [j for j in critical_blocks if j != j1]
            if not j2_candidates:
                continue
            j2 = random.choice(j2_candidates)
            
            unique_ts_12 = set()
            for ts_idx in block_to_ts_np[j1]:
                if cover_count[ts_idx] == 1:
                    unique_ts_12.add(ts_idx)
            for ts_idx in block_to_ts_np[j2]:
                if cover_count[ts_idx] == 1:
                    unique_ts_12.add(ts_idx)
            
            if not unique_ts_12:
                continue
                
            sample_size = min(150, len(inactive_blocks))
            sample = random.sample(inactive_blocks, sample_size)
            
            found_swap = False
            for j3 in sample:
                if unique_ts_12.issubset(block_to_ts_set[j3]):
                    x[j1] = False
                    cover_count[block_to_ts_np[j1]] -= 1
                    x[j2] = False
                    cover_count[block_to_ts_np[j2]] -= 1
                    x[j3] = True
                    cover_count[block_to_ts_np[j3]] += 1
                    
                    if np.min(cover_count) >= 1:
                        active_blocks.remove(j1)
                        active_blocks.remove(j2)
                        active_blocks.append(j3)
                        inactive_blocks.remove(j3)
                        inactive_blocks.append(j1)
                        inactive_blocks.append(j2)
                        current_cost -= 1
                        found_swap = True
                        
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_x = x.copy()
                            if best_cost == lb:
                                return best_x, best_cost
                        break
                    else:
                        x[j3] = False
                        cover_count[block_to_ts_np[j3]] -= 1
                        x[j2] = True
                        cover_count[block_to_ts_np[j2]] += 1
                        x[j1] = True
                        cover_count[block_to_ts_np[j1]] += 1
                        current_cost += 1

        return best_x, best_cost

    # Genetic Algorithm for better exploration
    def genetic_algorithm(x_init, max_generations=50, pop_size=30):
        """
        GA that evolves solutions to minimize number of blocks while maintaining coverage.
        Uses bitset-based fitness for fast evaluation.
        """
        if time.time() > deadline - 10.0:
            return x_init, int(np.sum(x_init))
        
        # Initialize population with x_init and some random perturbations
        population = [x_init.copy()]
        
        # Add diverse initial solutions
        for _ in range(pop_size - 1):
            new_x = x_init.copy()
            # Randomly remove some blocks
            active = np.where(new_x)[0].tolist()
            num_remove = random.randint(1, min(5, len(active)))
            for _ in range(num_remove):
                if active:
                    j = random.choice(active)
                    new_x[j] = False
                    active.remove(j)
            # Randomly add some blocks
            inactive = np.where(~new_x)[0].tolist()
            if inactive:
                num_add = random.randint(0, min(3, len(inactive)))
                for _ in range(num_add):
                    if inactive:
                        j = random.choice(inactive)
                        new_x[j] = True
                        inactive.remove(j)
            population.append(new_x)
        
        best_x = x_init.copy()
        best_cost = int(np.sum(x_init))
        
        for gen in range(max_generations):
            if time.time() > deadline - 10.0:
                break
            
            # Evaluate fitness for each individual
            # Fitness = number of blocks (lower is better), but must maintain coverage
            scored_population = []
            for ind in population:
                cc = get_cover_counts(ind)
                if np.min(cc) >= 1:  # Valid solution
                    cost = int(np.sum(ind))
                    scored_population.append((cost, ind))
                else:
                    # Invalid: add penalty
                    uncovered = np.sum(cc == 0)
                    cost = int(np.sum(ind)) + uncovered * 100
                    scored_population.append((cost, ind))
            
            # Sort by cost (ascending)
            scored_population.sort(key=lambda x: x[0])
            
            # Update best
            if scored_population[0][0] < best_cost:
                best_cost = scored_population[0][0]
                best_x = scored_population[0][1].copy()
            
            # Elitism: keep top 5
            elite = [ind.copy() for _, ind in scored_population[:5]]
            
            # Selection: tournament
            def tournament_select(pop):
                candidates = random.sample(pop, min(3, len(pop)))
                return min(candidates, key=lambda x: x[0])[1].copy()
            
            # Generate new population
            new_population = elite[:]
            
            while len(new_population) < pop_size:
                # Crossover: take blocks from two parents
                if len(population) < 2:
                    new_population.append(tournament_select(scored_population))
                    continue
                    
                parent1 = tournament_select(scored_population)
                parent2 = tournament_select(scored_population)
                
                # Create child by combining blocks
                child = np.zeros(num_k_blocks, dtype=bool)
                # Take blocks from parent1 with probability 0.6, parent2 with 0.4
                for j in range(num_k_blocks):
                    if parent1[j]:
                        if random.random() < 0.6:
                            child[j] = True
                    elif parent2[j]:
                        if random.random() < 0.4:
                            child[j] = True
                
                # Mutation: randomly add/remove blocks
                if random.random() < 0.3:
                    active = np.where(child)[0].tolist()
                    if active:
                        j = random.choice(active)
                        child[j] = False
                if random.random() < 0.3:
                    inactive = np.where(~child)[0].tolist()
                    if inactive:
                        j = random.choice(inactive)
                        child[j] = True
                        
                new_population.append(child)
            
            population = new_population
        
        return best_x, best_cost

    # Run the search
    current_x = x
    current_cc = cover_count
    
    # First pass: Pure hill climbing (removals)
    current_x, current_cost = local_search_hill_climb(current_x, current_cc, max_iters=2000)
    if current_cost < best_len:
        best_len = current_cost
        best_solution_idx = np.where(current_x)[0].tolist()
        current_cc = get_cover_counts(current_x)
        
    # Second pass: Hill climbing with swaps (perturbation)
    # Run multiple times
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        random.seed(restart)
        new_x, new_cost = local_search_with_swap(current_x, current_cc, max_iters=1000)
        
        if new_cost < best_len:
            best_len = new_cost
            best_solution_idx = np.where(new_x)[0].tolist()
            current_x = new_x
            current_cc = get_cover_counts(current_x)
            
            # After improvement, do another round of pure removals
            current_x, current_cost = local_search_hill_climb(current_x, current_cc, max_iters=1000)
            if current_cost < best_len:
                best_len = current_cost
                best_solution_idx = np.where(current_x)[0].tolist()
                current_cc = get_cover_counts(current_x)
    
    # Third pass: Genetic Algorithm for better exploration
    if time.time() < deadline - 15.0:
        ga_x, ga_cost = genetic_algorithm(current_x, max_generations=30, pop_size=25)
        if ga_cost < best_len:
            best_len = ga_cost
            best_solution_idx = np.where(ga_x)[0].tolist()
            current_x = ga_x
            current_cc = get_cover_counts(current_x)
            
            # Polish with hill climbing
            current_x, current_cost = local_search_hill_climb(current_x, current_cc, max_iters=1000)
            if current_cost < best_len:
                best_len = current_cost
                best_solution_idx = np.where(current_x)[0].tolist()
                current_cc = get_cover_counts(current_x)
            
            # Another round of swaps
            if time.time() < deadline - 10.0:
                new_x, new_cost = local_search_with_swap(current_x, current_cc, max_iters=500)
                if new_cost < best_len:
                    best_len = new_cost
                    best_solution_idx = np.where(new_x)[0].tolist()
                    current_x = new_x
                    current_cc = get_cover_counts(current_x)
    
    # Final cleanup
    final_x = np.zeros(num_k_blocks, dtype=bool)
    for j in best_solution_idx:
        final_x[j] = True
    final_cc = get_cover_counts(final_x)
    
    final_x, final_cost = local_search_hill_climb(final_x, final_cc, max_iters=1000)
    if final_cost < best_len:
        best_len = final_cost
        best_solution_idx = np.where(final_x)[0].tolist()

    # Convert indices back to blocks
    final_blocks = [list(all_k_blocks[j]) for j in best_solution_idx]
    
    return final_blocks
