import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # masks[i] is an integer bitmask
    masks = [0] * num_k_subsets
    # t_to_k[j] is list of k-subset indices that cover t-subset j
    t_to_k = [[] for _ in range(num_t_subsets)]
    
    # Fast initial gain calculation using numpy
    # Create a list of t-subset indices for each k-subset
    # This might be memory heavy but num_k_subsets is ~75k and each has C(8,5)=56 t-subsets.
    # 75k * 56 = 4.2M integers. Manageable.
    
    t_indices_per_k = []
    for i, ks in enumerate(k_subsets):
        mask = 0
        indices = []
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            t_to_k[idx].append(i)
            indices.append(idx)
        masks[i] = mask
        t_indices_per_k.append(indices)
        
    # Initial gains using bincount
    all_t_indices = np.concatenate(t_indices_per_k) if t_indices_per_k else np.array([])
    initial_gains = np.bincount(all_t_indices, minlength=num_t_subsets)
    # Wait, initial_gains[i] should be number of uncovered t-subsets covered by k-subset i.
    # Initially all are uncovered, so gain[i] = number of t-subsets in k-subset i.
    # This is just len(t_indices_per_k[i]).
    # The bincount above counts how many k-subsets cover each t-subset. That's not what we need for gain.
    # We need gain[i] = number of t-subsets covered by k-subset i.
    # Since initially all t-subsets are uncovered, gain[i] = C(k, t).
    # So initial gains are constant!
    initial_gain_value = len(t_indices_per_k[0]) if t_indices_per_k else 0
    gain = [initial_gain_value] * num_k_subsets

    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Greedy with randomization to collect diverse good solutions
    def greedy_construct(randomize=True, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        # gain[i] is dynamic
        gain = [initial_gain_value] * num_k_subsets
        
        while uncovered:
            if time.time() > deadline:
                return None, False
            
            # Find max gain
            max_gain = 0
            candidates = []
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                if gain[i] > max_gain:
                    max_gain = gain[i]
                    candidates = [i]
                elif gain[i] == max_gain and gain[i] > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                return current_blocks, False
            
            if randomize and candidates:
                chosen_idx = random.choice(candidates)
            else:
                chosen_idx = candidates[0] if candidates else -1
            
            if chosen_idx == -1:
                return current_blocks, False
                
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            # Update uncovered and gains
            newly_covered = uncovered & masks[chosen_idx]
            uncovered &= ~masks[chosen_idx]
            
            # Efficient gain update using t_to_k
            nc = newly_covered
            while nc:
                lsb = nc & -nc
                idx_j = lsb.bit_length() - 1
                nc ^= lsb
                for i in t_to_k[idx_j]:
                    if i not in used_blocks:
                        gain[i] -= 1
                        if gain[i] < 0: gain[i] = 0

        return current_blocks, True

    def prune_redundant(block_indices):
        if not block_indices:
            return []
        
        cover_counts = [0] * num_t_subsets
        for idx in block_indices:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        changed = True
        current_blocks = list(block_indices)
        while changed:
            changed = False
            new_blocks = []
            for idx in current_blocks:
                mask = masks[idx]
                m = mask
                redundant = True
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if cover_counts[j] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_counts[j] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_blocks.append(idx)
            current_blocks = new_blocks
        return current_blocks

    def local_search_improve(block_indices, max_iters=1000):
        if not block_indices:
            return []
        
        current_blocks = list(block_indices)
        
        cover_counts = [0] * num_t_subsets
        for idx in current_blocks:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        def is_removable(idx):
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if cover_counts[j] == 1:
                    return False
                m ^= lsb
            return True
        
        def remove_block(idx):
            nonlocal current_blocks
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] -= 1
                m ^= lsb
            current_blocks.remove(idx)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_blocks)
            
            for idx in current_blocks[:]:
                if time.time() > deadline:
                    break
                if is_removable(idx):
                    remove_block(idx)
                    improved = True
        
        return current_blocks

    # Phase 2: Genetic Algorithm
    def genetic_search(initial_solutions, time_limit):
        if not initial_solutions:
            return []
        
        # Convert to bitmasks
        pop_masks = []
        for sol in initial_solutions:
            mask = 0
            for idx in sol:
                mask |= (1 << idx)
            pop_masks.append(mask)
            
        best_mask = 0
        best_fitness = float('inf')
        
        # Calculate initial best
        all_ones = (1 << num_t_subsets) - 1
        
        def get_fitness(mask):
            union_mask = 0
            # We need to compute the union of masks of all set bits in 'mask'
            # This is expensive if done naively.
            # However, we can pre-compute nothing easily for arbitrary masks.
            # But note: 'mask' represents the set of blocks.
            # The coverage is OR of masks[i] for all i in mask.
            # To speed this up, we can iterate through set bits.
            m = mask
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                union_mask |= masks[i]
                m ^= lsb
            
            uncovered = all_ones & ~union_mask
            num_uncovered = uncovered.bit_count()
            num_blocks = mask.bit_count()
            return num_blocks + 100 * num_uncovered

        # Evaluate initial population
        for m in pop_masks:
            f = get_fitness(m)
            if f < best_fitness:
                best_fitness = f
                best_mask = m

        population = list(pop_masks)
        
        while time.time() < start_time + time_limit:
            new_population = []
            
            # Selection and Crossover
            for _ in range(len(population) // 2):
                # Tournament Selection
                p1 = random.choice(population)
                p2 = random.choice(population)
                
                # Uniform Crossover
                rand_mask = random.getrandbits(num_k_subsets)
                child = (p1 & rand_mask) | (p2 & ~rand_mask)
                
                # Mutation: Flip a few bits
                # Probability of mutation per bit is low
                # Let's flip ~5 bits on average
                num_mutations = random.randint(1, 10)
                for _ in range(num_mutations):
                    bit_pos = random.randint(0, num_k_subsets - 1)
                    child ^= (1 << bit_pos)
                
                new_population.append(child)
            
            # Elitism: Keep best individuals
            population.sort(key=get_fitness)
            new_population = population[:len(population)//2] + new_population
            
            # Update best
            for m in new_population:
                f = get_fitness(m)
                if f < best_fitness:
                    best_fitness = f
                    best_mask = m
            
            population = new_population

        # Convert best_mask back to list of indices
        best_indices = []
        m = best_mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            best_indices.append(i)
            m ^= lsb
        return best_indices

    # Main Loop
    # Collect top solutions from greedy restarts
    top_solutions = []
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0: # Reserve time for GA
            break
            
        blocks, valid = greedy_construct(randomize=True)
        if not valid:
            continue
            
        pruned = prune_redundant(blocks)
        ls_blocks = local_search_improve(pruned)
        
        if len(ls_blocks) < best_count:
            best_count = len(ls_blocks)
            best_blocks = [list(k_subsets[i]) for i in ls_blocks]
            
        # Add to top solutions if it's good
        if len(ls_blocks) <= best_count + 5: # Keep solutions close to best
            # Check if it's distinct (simple check on sorted tuple)
            key = tuple(sorted(ls_blocks))
            if key not in [tuple(sorted(s)) for s in top_solutions]:
                top_solutions.append(ls_blocks)
                if len(top_solutions) > 10:
                    top_solutions.pop()

    # Run GA if we have time and solutions
    if top_solutions and time.time() < deadline - 5.0:
        ga_start = time.time()
        ga_time_limit = min(10.0, deadline - 5.0 - ga_start)
        if ga_time_limit > 1.0:
            ga_result = genetic_search(top_solutions, ga_time_limit)
            if ga_result and len(ga_result) < best_count:
                # Verify validity just in case
                # (GA fitness ensures validity if penalty is high enough and we check uncovered==0)
                # But get_fitness returns penalty if uncovered > 0.
                # We should only accept if uncovered is 0.
                # Let's re-verify the GA result.
                cover_counts = [0] * num_t_subsets
                for idx in ga_result:
                    mask = masks[idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_counts[j] += 1
                        m ^= lsb
                if all(c > 0 for c in cover_counts):
                    best_count = len(ga_result)
                    best_blocks = [list(k_subsets[i]) for i in ga_result]

    if best_blocks is None:
        blocks, valid = greedy_construct(randomize=False)
        if valid:
            pruned = prune_redundant(blocks)
            best_blocks = [list(k_subsets[i]) for i in pruned]
        else:
            best_blocks = []

    return best_blocks
