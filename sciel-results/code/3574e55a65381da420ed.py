import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets and k-blocks
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_cover = []
    for blk in all_k_blocks:
        covers = []
        for ts in itertools.combinations(blk, t):
            covers.append(ts_index[ts])
        block_cover.append(np.array(covers, dtype=np.int32))

    # Inverse: for each t-subset, which blocks cover it
    ts_to_blocks = [[] for _ in range(num_t)]
    for j, cov in enumerate(block_cover):
        for ts_idx in cov:
            ts_to_blocks[ts_idx].append(j)
    ts_to_blocks = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    def uncovered_count(mask):
        """Return number of uncovered t-subsets given a boolean mask of selected blocks."""
        # mask: bool array of length num_k
        if not np.any(mask):
            return num_t
        # For speed, iterate over selected blocks and mark covered
        covered = np.zeros(num_t, dtype=bool)
        sel_idx = np.where(mask)[0]
        for j in sel_idx:
            covered[block_cover[j]] = True
        return num_t - np.count_nonzero(covered)

    def is_valid(mask):
        sel_idx = np.where(mask)[0]
        if len(sel_idx) == 0:
            return False
        covered = np.zeros(num_t, dtype=bool)
        for j in sel_idx:
            covered[block_cover[j]] = True
        return np.all(covered)

    def fitness(mask):
        """Lower is better: number of blocks + large penalty * uncovered_count."""
        sel_count = np.count_nonzero(mask)
        unc = uncovered_count(mask)
        return sel_count + 10000 * unc

    def repair(mask):
        """Add blocks to cover all uncovered t-subsets. Returns new mask."""
        sel_idx = np.where(mask)[0]
        if len(sel_idx) == 0:
            # Start with a greedy solution
            uncovered = np.ones(num_t, dtype=bool)
            new_mask = mask.copy()
            while np.any(uncovered):
                best_j = -1
                best_gain = 0
                for j in range(num_k):
                    if new_mask[j]:
                        continue
                    gain = np.count_nonzero(block_cover[j][uncovered])
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                if best_j < 0 or best_gain == 0:
                    break
                new_mask[best_j] = True
                uncovered[block_cover[best_j]] = False
            return new_mask

        covered = np.zeros(num_t, dtype=bool)
        for j in sel_idx:
            covered[block_cover[j]] = True
        uncovered = ~covered
        if not np.any(uncovered):
            return mask

        new_mask = mask.copy()
        # Greedily add blocks to cover uncovered
        while np.any(uncovered):
            best_j = -1
            best_gain = 0
            for j in range(num_k):
                if new_mask[j]:
                    continue
                gain = np.count_nonzero(block_cover[j][uncovered])
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j < 0 or best_gain == 0:
                break
            new_mask[best_j] = True
            uncovered[block_cover[best_j]] = False
        return new_mask

    def mask_to_blocks(mask):
        sel = np.where(mask)[0]
        return [list(all_k_blocks[j]) for j in sel]

    # --- Greedy initial solution ---
    def greedy_solution():
        uncovered = np.ones(num_t, dtype=bool)
        mask = np.zeros(num_k, dtype=bool)
        while np.any(uncovered):
            best_j = -1
            best_gain = 0
            for j in range(num_k):
                if mask[j]:
                    continue
                gain = np.count_nonzero(block_cover[j][uncovered])
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j < 0 or best_gain == 0:
                break
            mask[best_j] = True
            uncovered[block_cover[best_j]] = False
        return mask

    # --- Random initial solution ---
    def random_solution(num_blocks):
        mask = np.zeros(num_k, dtype=bool)
        idxs = random.sample(range(num_k), min(num_blocks, num_k))
        mask[idxs] = True
        return repair(mask)

    # --- Crossover: uniform ---
    def crossover(p1, p2):
        child = np.random.rand(num_k) < 0.5
        child = np.where(child, p1, p2)
        return child

    # --- Mutation: flip random bits ---
    def mutate(mask, rate=0.005):
        new_mask = mask.copy()
        flip_idx = np.random.choice(num_k, size=int(num_k * rate), replace=False)
        new_mask[flip_idx] = ~new_mask[flip_idx]
        return new_mask

    # --- Simulated annealing style local search on a single mask ---
    def anneal(mask, time_limit):
        """Try to reduce block count while maintaining coverage."""
        current = mask.copy()
        current_f = fitness(current)
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        best = current.copy()
        best_f = current_f

        sel_idx = np.where(current)[0]
        covered = np.zeros(num_t, dtype=bool)
        for j in sel_idx:
            covered[block_cover[j]] = True

        while time.time() < time_limit and T > T_min:
            # Propose: remove a random block or swap
            op = random.random()
            if op < 0.5 and len(sel_idx) > 1:
                # Remove a block
                rem = random.choice(sel_idx)
                new_covered = covered.copy()
                new_covered[block_cover[rem]] = False
                # Check if still valid (simplified: just check uncovered)
                unc = num_t - np.count_nonzero(new_covered)
                new_count = len(sel_idx) - 1
                new_f = new_count + 10000 * unc
                delta = new_f - current_f
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    sel_idx = sel_idx[sel_idx != rem]
                    covered = new_covered
                    current[rem] = False
                    current_f = new_f
                    if current_f < best_f:
                        best_f = current_f
                        best = current.copy()
            elif op < 0.8:
                # Swap: remove one, add another
                if len(sel_idx) == 0:
                    T *= cooling
                    continue
                rem = random.choice(sel_idx)
                add = random.randrange(num_k)
                if add == rem or current[add]:
                    T *= cooling
                    continue
                # Compute new state
                # Remove rem
                new_covered = covered.copy()
                new_covered[block_cover[rem]] = False
                # Add add
                new_covered[block_cover[add]] = True
                unc = num_t - np.count_nonzero(new_covered)
                new_count = len(sel_idx)  # same count
                new_f = new_count + 10000 * unc
                delta = new_f - current_f
                if delta < 0 or random.random() < np.exp(-delta / T):
                    sel_idx = list(set(list(sel_idx) - [rem] + [add]))
                    covered = new_covered
                    current[rem] = False
                    current[add] = True
                    current_f = new_f
                    if current_f < best_f:
                        best_f = current_f
                        best = current.copy()
            else:
                # Add a block (to help with coverage)
                add = random.randrange(num_k)
                if current[add]:
                    T *= cooling
                    continue
                new_covered = covered.copy()
                new_covered[block_cover[add]] = True
                unc = num_t - np.count_nonzero(new_covered)
                new_count = len(sel_idx) + 1
                new_f = new_count + 10000 * unc
                delta = new_f - current_f
                if delta < 0 or random.random() < np.exp(-delta / T):
                    sel_idx = list(set(list(sel_idx) + [add]))
                    covered = new_covered
                    current[add] = True
                    current_f = new_f
                    if current_f < best_f:
                        best_f = current_f
                        best = current.copy()

            T *= cooling

        return best, best_f

    # --- Main GA loop ---
    # Initialize population
    pop_size = 30
    population = []
    # Start with greedy + some random
    gmask = greedy_solution()
    population.append(gmask)
    for i in range(pop_size - 1):
        rm = random_solution(350 + random.randint(-50, 50))
        population.append(rm)

    best_mask = None
    best_score = float('inf')

    # Evaluate initial population
    for m in population:
        f = fitness(m)
        if f < best_score:
            best_score = f
            best_mask = m.copy()

    # GA iterations
    gen = 0
    while time.time() < deadline - 15.0:
        gen += 1
        # Sort by fitness
        scored = sorted(population, key=lambda m: fitness(m))
        # Keep top half
        elite = scored[:max(5, pop_size // 2)]
        # Generate new population
        new_pop = list(elite)
        while len(new_pop) < pop_size and time.time() < deadline - 15.0:
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            if random.random() < 0.7:
                child = crossover(p1, p2)
            else:
                child = p1.copy()
            child = mutate(child, rate=0.003)
            # Repair if invalid
            child = repair(child)
            new_pop.append(child)

        population = new_pop

        # Update best
        for m in population:
            f = fitness(m)
            if f < best_score:
                best_score = f
                best_mask = m.copy()

        # Occasionally anneal the best
        if gen % 5 == 0 and time.time() < deadline - 20.0:
            anneal_end = min(time.time() + 5.0, deadline - 15.0)
            annealed, af = anneal(best_mask, anneal_end)
            if af < best_score:
                best_score = af
                best_mask = annealed.copy()
                # Replace worst individual with annealed
                scored = sorted(population, key=lambda m: fitness(m))
                population[-1] = annealed.copy()

    # Final repair to ensure validity
    final_mask = repair(best_mask)
    final_blocks = mask_to_blocks(final_mask)

    # Verify and return
    return final_blocks
