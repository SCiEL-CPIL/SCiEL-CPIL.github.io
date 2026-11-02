import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    if t == 0 or k == 0 or t > k or v <= 0:
        return []

    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subsets)
    sub_id = {s: i for i, s in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    block_to_ts = []
    ts_to_blocks = [[] for _ in range(n_t_subs)]
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = sub_id[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

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

    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
        gain = np.full(n_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(n_t_subs, True, dtype=bool)
        is_selected = np.full(n_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 8.0:
                return None
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            if explore_prob > 0 and random.random() < explore_prob:
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
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
                counts = np.bincount(blocks_to_decrement, minlength=n_k_blocks)
                gain -= counts
                gain[is_selected] = -1000
        return sol

    def compute_cover_count(blocks):
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        block_ts = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(sub_id[ts])
                block_ts.append(ts_list)
        for ts_list in block_ts:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
        return cover_count, block_ts

    def optimize_redundancy(blocks):
        cover_count, block_ts = compute_cover_count(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks
                redundant = True
                for ts_idx in block_ts[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts[i]:
                        cover_count[ts_idx] -= 1
                    block_ts.pop(i)
                    changed = True
                    break
        return blocks

    def local_search_swap(blocks, seed):
        random.seed(seed)
        cover_count, block_ts = compute_cover_count(blocks)
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)

        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 3.0:
                break
            unique_ts = []
            for ts_idx in block_ts[i]:
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
            old_block = blocks[i]
            old_block_idx = get_block_idx(old_block)
            blocks[i] = list(all_k_blocks[new_block_idx])
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)
            for ts_idx in block_ts[i]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            block_ts[i] = block_to_ts[new_block_idx]
            improved = True
        return blocks, improved

    def crossover(p1, p2):
        random.seed(random.randint(0, 2**31 - 1))
        n = len(p1)
        if n == 0:
            return []
        m = len(p2)
        if m == 0:
            return p1[:]
        # Create child by sampling from both parents
        # Keep blocks that appear in both (intersection) with priority
        set1 = set(tuple(b) for b in p1)
        set2 = set(tuple(b) for b in p2)
        common = set1 & set2
        child = [list(b) for b in common]
        # Fill remaining from p1, then p2, preferring those not already present
        remaining1 = [b for b in p1 if tuple(b) not in common]
        remaining2 = [b for b in p2 if tuple(b) not in common]
        # Shuffle and interleave
        random.shuffle(remaining1)
        random.shuffle(remaining2)
        for b in remaining1 + remaining2:
            if tuple(b) not in {tuple(c) for c in child}:
                child.append(b)
        return child

    def mutate(blocks, mutation_rate=0.1):
        random.seed(random.randint(0, 2**31 - 1))
        if not blocks:
            return blocks
        for i in range(len(blocks)):
            if random.random() < mutation_rate:
                # Replace block i with a random block
                blocks[i] = list(random.choice(all_k_blocks))
        return blocks

    def genetic_search(initial_pops, pop_size, generations, seed):
        random.seed(seed)
        # Initialize population
        population = []
        for i in range(pop_size):
            if i < len(initial_pops):
                population.append(initial_pops[i][:])
            else:
                # Generate random solution via greedy
                sol = run_greedy(seed + i * 1000, explore_prob=0.15)
                if sol is not None:
                    population.append(sol)
                else:
                    population.append(initial_pops[0][:])

        # Ensure population has pop_size entries
        while len(population) < pop_size:
            population.append(population[0][:])

        best_sol = min(population, key=len)
        best_len = len(best_sol)

        for gen in range(generations):
            if time.time() > deadline - 10.0:
                break

            # Evaluate fitness: (num_blocks, num_uncovered)
            def fitness(sol):
                if not sol:
                    return (999999, 999999)
                cover_count, _ = compute_cover_count(sol)
                uncovered = int(np.sum(cover_count == 0))
                return (len(sol), uncovered)

            # Sort by fitness
            scored = [(fitness(ind), i, ind) for i, ind in enumerate(population)]
            scored.sort(key=lambda x: (x[0][0], x[0][1]))

            # Update best
            if scored[0][0][0] < best_len:
                best_len = scored[0][0][0]
                best_sol = scored[0][2][:]

            # Elitism: keep top 20%
            elite_count = max(1, pop_size // 5)
            new_population = [scored[i][2][:] for i in range(elite_count)]

            # Generate offspring
            while len(new_population) < pop_size:
                if time.time() > deadline - 10.0:
                    break
                # Tournament selection
                def select():
                    idx = random.randint(0, len(scored) - 1)
                    return scored[idx][2]

                parent1 = select()
                parent2 = select()
                # Crossover
                if random.random() < 0.7:
                    child = crossover(parent1, parent2)
                else:
                    child = parent1[:]
                # Mutation
                child = mutate(child, mutation_rate=0.05)
                # Ensure validity: if any t-subset is uncovered, add blocks greedily
                if child:
                    cover_count, _ = compute_cover_count(child)
                    uncovered = np.where(cover_count == 0)[0]
                    if len(uncovered) > 0:
                        # Greedily add blocks to cover uncovered
                        existing = set(tuple(b) for b in child)
                        for ts_idx in uncovered:
                            if time.time() > deadline - 10.0:
                                break
                            candidates = [b for b in ts_to_blocks[ts_idx] if tuple(all_k_blocks[b]) not in existing]
                            if candidates:
                                best_b = random.choice(candidates)
                                child.append(list(all_k_blocks[best_b]))
                                existing.add(tuple(all_k_blocks[best_b]))
                new_population.append(child)

            population = new_population[:pop_size]
            # Trim if over
            if len(population) > pop_size:
                population = population[:pop_size]

        return best_sol, best_len

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    # Run multiple randomized greedy
    num_greedy = 60
    initial_pops = []
    for i in range(num_greedy):
        if time.time() > deadline - 30.0:
            break
        explore = 0.0 if i < 20 else (0.1 if i < 40 else 0.2)
        sol = run_greedy(i + 1, explore_prob=explore)
        if sol is not None:
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol[:]
            initial_pops.append(sol)

    # Apply redundancy removal
    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)

    # Local search on best
    for _ in range(5):
        if time.time() > deadline - 25.0:
            break
        blocks, improved = local_search_swap(best_solution, random.randint(0, 10000))
        if not improved:
            break
        best_solution = optimize_redundancy(blocks)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Genetic algorithm
    if time.time() < deadline - 40.0:
        pop_size = 12
        generations = 15
        # Use initial_pops as seed for GA
        ga_pops = initial_pops[:pop_size]
        while len(ga_pops) < pop_size:
            ga_pops.append(ga_pops[0][:])
        ga_sol, ga_len = genetic_search(ga_pops, pop_size, generations, seed=42)
        if ga_len < best_len:
            best_len = ga_len
            best_solution = ga_sol[:]

    # Final local search polish
    if time.time() < deadline - 15.0:
        for _ in range(8):
            if time.time() > deadline - 5.0:
                break
            blocks, improved = local_search_swap(best_solution, random.randint(0, 10000))
            if not improved:
                break
            best_solution = optimize_redundancy(blocks)
            if len(best_solution) < best_len:
                best_len = len(best_solution)

    # Final redundancy check
    best_solution = optimize_redundancy(best_solution)

    return best_solution
