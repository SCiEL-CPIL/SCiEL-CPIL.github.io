import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    def time_ok():
        return time.time() < deadline

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build incidence: for each block, list of t-subset indices it covers
    block_to_tsubsets = []
    tsub_to_blocks = [[] for _ in range(N)]
    for bi, b in enumerate(all_blocks):
        covers = []
        for s in itertools.combinations(b, t):
            ti = t_idx[s]
            covers.append(ti)
            tsub_to_blocks[ti].append(bi)
        block_to_tsubsets.append(np.array(covers, dtype=np.int32))

    # --- Co-occurrence-based block pool (diversity-aware subset of all blocks) ---
    # Score each block by how "rare" the t-subsets it covers are.
    # A t-subset is rare if few blocks cover it -> blocks covering rare subsets are valuable.
    tsub_rarity = np.zeros(N, dtype=np.float64)
    for ti in range(N):
        cnt = len(tsub_to_blocks[ti])
        tsub_rarity[ti] = 1.0 / max(cnt, 1)

    block_scores = np.zeros(num_blocks, dtype=np.float64)
    for bi in range(num_blocks):
        cov = block_to_tsubsets[bi]
        block_scores[bi] = np.sum(tsub_rarity[cov])

    # Build a ranked pool: top blocks by score, plus a random sample for diversity
    pool_size = min(num_blocks, 5000)
    ranked_order = np.argsort(-block_scores)
    pool = list(ranked_order[:min(pool_size, num_blocks)])
    # Ensure pool is a set for O(1) lookup
    pool_set = set(pool)

    def block_in_pool(bi):
        return bi in pool_set

    # --- Helper: coverage count for a set of blocks ---
    def coverage_count(blocks_list):
        cc = np.zeros(N, dtype=np.int32)
        for bi in blocks_list:
            cc[block_to_tsubsets[bi]] += 1
        return cc

    def is_valid(blocks_list):
        cc = coverage_count(blocks_list)
        return np.all(cc > 0)

    def uncovered_mask(blocks_list):
        cc = coverage_count(blocks_list)
        return cc == 0

    # --- Greedy initial solution (fast, to seed the population) ---
    def greedy_initial(seed=0):
        random.seed(seed)
        uncovered = np.ones(N, dtype=bool)
        selected = []
        used = set()
        while np.any(uncovered):
            if not time_ok():
                break
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            target = random.choice(unc_idx)
            candidates = tsub_to_blocks[target]
            best_bi = -1
            best_gain = 0
            for bi in candidates:
                if bi in used:
                    continue
                cov = block_to_tsubsets[bi]
                gain = np.sum(uncovered[cov])
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    if gain > 60:
                        break
            if best_bi == -1:
                break
            selected.append(best_bi)
            used.add(best_bi)
            uncovered[block_to_tsubsets[best_bi]] = False
        # Repair: add any missing blocks
        if not np.all(~uncovered):
            for ti in np.where(uncovered)[0]:
                for bi in tsub_to_blocks[ti]:
                    if bi not in used:
                        selected.append(bi)
                        used.add(bi)
                        break
        return selected

    # --- Local search: remove redundant blocks, swap to improve ---
    def local_search(blocks_list, max_iter=2000):
        current = list(blocks_list)
        current_size = len(current)
        best = list(current)
        best_size = current_size
        cc = coverage_count(current)

        for _ in range(max_iter):
            if not time_ok():
                break
            idx = random.randint(0, len(current) - 1)
            bi = current[idx]
            cov = block_to_tsubsets[bi]
            would_break = False
            for ti in cov:
                if cc[ti] <= 1:
                    would_break = True
                    break
            if not would_break:
                cc[cov] -= 1
                current.pop(idx)
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best = list(current)
                continue
            # Try swap
            uncovered_by_removal = [ti for ti in cov if cc[ti] == 1]
            if not uncovered_by_removal:
                continue
            current_set = set(current)
            cands = set(tsub_to_blocks[uncovered_by_removal[0]])
            for ti in uncovered_by_removal[1:]:
                cands &= set(tsub_to_blocks[ti])
            cands -= current_set
            if not cands:
                continue
            # Pick candidate that covers most currently-single-covered t-subsets
            best_cand = -1
            best_score = -1
            for cand in list(cands)[:15]:
                new_cov = block_to_tsubsets[cand]
                score = np.sum(cc[new_cov] == 1)
                if score > best_score:
                    best_score = score
                    best_cand = cand
            if best_cand != -1:
                cc[block_to_tsubsets[bi]] -= 1
                cc[block_to_tsubsets[best_cand]] += 1
                current[idx] = best_cand

        return best, best_size

    # --- Genetic Algorithm ---
    def genetic_algorithm(initial_sol, pop_size=30, generations=50):
        start_time = time.time()
        # Initialize population
        population = []
        # First individual: greedy solution
        population.append(list(initial_sol))
        # Add more greedy solutions with different seeds
        for i in range(1, pop_size):
            if not time_ok():
                break
            gs = greedy_initial(seed=i * 13 + 7)
            if len(gs) < len(population[0]):
                population[0] = list(gs)
            population.append(list(gs))

        best_individual = list(population[0])
        best_size = len(population[0])

        for gen in range(generations):
            if not time_ok():
                break

            # Evaluate and sort population
            scored = []
            for ind in population:
                if not is_valid(ind):
                    # Repair invalid individuals
                    cc = coverage_count(ind)
                    unc = np.where(cc == 0)[0]
                    used = set(ind)
                    for ti in unc:
                        for bi in tsub_to_blocks[ti]:
                            if bi not in used:
                                ind.append(bi)
                                used.add(bi)
                                break
                    if not is_valid(ind):
                        scored.append((10000 + len(ind), ind))
                    else:
                        scored.append((len(ind), ind))
                else:
                    scored.append((len(ind), ind))
            scored.sort(key=lambda x: x[0])

            # Update best
            if scored[0][0] < best_size:
                best_size = scored[0][0]
                best_individual = list(scored[0][1])

            # Create next generation
            next_pop = []
            # Elitism: keep top half
            elite_count = max(2, pop_size // 2)
            for i in range(elite_count):
                next_pop.append(list(scored[i][1]))

            # Crossover + mutation for the rest
            while len(next_pop) < pop_size:
                if not time_ok():
                    break
                # Pick two parents (tournament selection)
                p1_idx = random.randint(0, min(5, len(scored) - 1))
                p2_idx = random.randint(0, min(5, len(scored) - 1))
                p1 = scored[p1_idx][1]
                p2 = scored[p2_idx][1]

                # Crossover: union with random selection
                child = list(p1)
                p1_set = set(child)
                for bi in p2:
                    if bi not in p1_set and random.random() < 0.5:
                        child.append(bi)
                        p1_set.add(bi)

                # Mutation: random swaps/adds/removes
                if random.random() < 0.7 and len(child) > 0:
                    # Remove a random block
                    idx = random.randint(0, len(child) - 1)
                    child.pop(idx)
                if random.random() < 0.5:
                    # Add a random block from pool
                    if pool:
                        new_bi = pool[random.randint(0, len(pool) - 1)]
                        if new_bi not in child:
                            child.append(new_bi)
                if random.random() < 0.3:
                    # Swap: remove one, add another
                    if len(child) > 0 and pool:
                        idx = random.randint(0, len(child) - 1)
                        old_bi = child[idx]
                        new_bi = pool[random.randint(0, len(pool) - 1)]
                        if new_bi != old_bi:
                            child[idx] = new_bi

                next_pop.append(child)

            population = next_pop

        # Final local search on best
        improved, improved_size = local_search(best_individual, max_iter=3000)
        if improved_size < best_size:
            best_size = improved_size
            best_individual = improved

        return best_individual

    # Phase 1: Greedy initial
    initial_sol = greedy_initial(seed=42)
    if not initial_sol:
        return [list(b) for b in all_blocks]

    # Phase 2: Genetic algorithm
    optimized_sol = genetic_algorithm(initial_sol, pop_size=25, generations=40)

    # Phase 3: Final local search polish
    final_sol, _ = local_search(optimized_sol, max_iter=5000)

    # Convert back to point lists
    final_blocks = [list(all_blocks[bi]) for bi in final_sol]

    return final_blocks
