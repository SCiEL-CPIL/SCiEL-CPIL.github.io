import itertools
import random
import time
import numpy as np
from collections import defaultdict

def _precompute(v, k, t):
    """Precompute t-subsets, k-blocks, and mappings efficiently."""
    tsubs = list(itertools.combinations(range(v), t))
    tsub_idx = {ts: i for i, ts in enumerate(tsubs)}
    n_tsubs = len(tsubs)
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    block_idx = {b: i for i, b in enumerate(blocks)}
    
    # For each block, list of t-subset indices it covers
    block_to_ts = [[] for _ in range(n_blocks)]
    # For each t-subset, list of block indices that cover it
    ts_to_blocks = [[] for _ in range(n_tsubs)]
    
    # Precompute all t-subsets of each block
    for bi, b in enumerate(blocks):
        for ts in itertools.combinations(b, t):
            tsi = tsub_idx[ts]
            block_to_ts[bi].append(tsi)
            ts_to_blocks[tsi].append(bi)
            
    # Convert to numpy arrays for faster access
    block_to_ts_arr = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]
    
    # Precompute the count of blocks covering each t-subset
    ts_cover_count = np.array([len(lst) for lst in ts_to_blocks], dtype=np.int32)
    
    return tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, ts_cover_count

def _evaluate_solution(sol, precomp):
    """Evaluate a solution: return (num_uncovered, num_blocks)."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, ts_cover_count = precomp
    if not sol:
        return n_tsubs, 0
    
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for bi in sol:
        cov[block_to_ts_arr[bi]] += 1
        
    uncovered = np.sum(cov == 0)
    return uncovered, len(sol)

def _greedy_complete(sol, precomp, deadline):
    """Greedily add blocks to cover any uncovered t-subsets."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, ts_cover_count = precomp
    
    if not sol:
        sol = []
        
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for bi in sol:
        cov[block_to_ts_arr[bi]] += 1
        
    uncovered = np.where(cov == 0)[0]
    
    # Track available blocks (all blocks are initially available for adding)
    # We just pick the one with max gain among all blocks
    while len(uncovered) > 0 and time.time() < deadline - 5.0:
        # Calculate gain for each block: number of currently uncovered t-subsets it covers
        gains = np.zeros(n_blocks, dtype=np.int32)
        for tsi in uncovered:
            for bi in ts_to_blocks_arr[tsi]:
                gains[bi] += 1
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break # No block can cover any remaining uncovered t-subset (should not happen)
            
        candidates = np.where(gains == max_gain)[0]
        best_bi = candidates[np.random.randint(len(candidates))]
        
        sol.append(best_bi)
        cov[block_to_ts_arr[best_bi]] += 1
        # Update uncovered list
        newly_covered = block_to_ts_arr[best_bi][cov[block_to_ts_arr[best_bi]] == 1]
        uncovered = np.setdiff1d(uncovered, newly_covered)
        
    return sol

def _local_search(sol, precomp, deadline):
    """Local search: redundancy removal and 1-swaps."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, ts_cover_count = precomp
    rng = random.Random(42)
    
    sol = list(sol)
    
    # Build coverage array
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for bi in sol:
        cov[block_to_ts_arr[bi]] += 1
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Remove redundant blocks
        indices = list(range(len(sol)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 2.0:
                return sol
            bi = sol[i]
            ts_idxs = block_to_ts_arr[bi]
            if np.all(cov[ts_idxs] > 1):
                cov[ts_idxs] -= 1
                sol.pop(i)
                improved = True
                break # Restart loop
                
        if improved:
            continue
            
        # Phase 2: Try 1-swaps
        num_swaps_tried = 0
        max_swaps = 50
        
        while num_swaps_tried < max_swaps and time.time() < deadline - 2.0:
            num_swaps_tried += 1
            if len(sol) == 0:
                break
                
            i = rng.randrange(len(sol))
            bi = sol[i]
            ts_idxs = block_to_ts_arr[bi]
            
            # Find critical t-subsets
            critical_mask = cov[ts_idxs] == 1
            critical_ts = ts_idxs[critical_mask]
            
            if len(critical_ts) == 0:
                cov[ts_idxs] -= 1
                sol.pop(i)
                improved = True
                break
                
            # Find a replacement
            crit_sorted = sorted(critical_ts, key=lambda x: len(ts_to_blocks_arr[x]))
            first_crit = crit_sorted[0]
            
            found_replacement = False
            candidates = list(ts_to_blocks_arr[first_crit])
            rng.shuffle(candidates)
            
            for bi2 in candidates:
                if bi2 == bi:
                    continue
                ts_idxs_bi2 = block_to_ts_arr[bi2]
                if set(critical_ts).issubset(set(ts_idxs_bi2)):
                    cov[ts_idxs] -= 1
                    cov[ts_idxs_bi2] += 1
                    sol[i] = bi2
                    improved = True
                    found_replacement = True
                    break
                    
            if found_replacement:
                break # Restart loop
                
    return sol

def _ga_search(v, k, t, precomp, deadline):
    """Genetic Algorithm for covering design."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, ts_cover_count = precomp
    
    pop_size = 50
    elite_size = 5
    mutation_rate = 0.3
    
    # Initialize population with greedy solutions
    population = []
    for i in range(pop_size):
        # Random initial solution: start empty or with random blocks
        sol = []
        # Start with a few random blocks to give it a head start
        for _ in range(random.randint(100, 300)):
            sol.append(random.randint(0, n_blocks - 1))
        # Complete greedily
        sol = _greedy_complete(sol, precomp, deadline)
        population.append(sol)
        
    best_sol = None
    best_fitness = (n_tsubs, float('inf')) # (uncovered, count)
    
    generation = 0
    while time.time() < deadline - 10.0: # Leave time for final polish
        generation += 1
        
        # Evaluate population
        for i, sol in enumerate(population):
            unc, cnt = _evaluate_solution(sol, precomp)
            population[i] = (unc, cnt, sol)
            
            # Update best
            if (unc, cnt) < best_fitness:
                best_fitness = (unc, cnt)
                best_sol = sol
                
        # Sort by fitness
        population.sort(key=lambda x: (x[0], x[1]))
        
        # Elitism
        new_population = [population[i] for i in range(min(elite_size, len(population)))]
        
        # Selection and Crossover
        while len(new_population) < pop_size:
            # Tournament selection
            p1 = population[random.randint(0, len(population)-1)]
            p2 = population[random.randint(0, len(population)-1)]
            if p1[0] < p2[0] or (p1[0] == p2[0] and p1[1] < p2[1]):
                parent1 = p1[2]
            else:
                parent1 = p2[2]
                
            p1 = population[random.randint(0, len(population)-1)]
            p2 = population[random.randint(0, len(population)-1)]
            if p1[0] < p2[0] or (p1[0] == p2[0] and p1[1] < p2[1]):
                parent2 = p1[2]
            else:
                parent2 = p2[2]
                
            # Crossover: Union of parents, then prune
            child_sol = list(set(parent1) | set(parent2))
            
            # Mutation
            if random.random() < mutation_rate:
                # Add random block
                if random.random() < 0.5:
                    child_sol.append(random.randint(0, n_blocks - 1))
                # Remove random block
                elif child_sol and random.random() < 0.5:
                    idx = random.randint(0, len(child_sol)-1)
                    child_sol.pop(idx)
                # Swap block
                elif len(child_sol) > 0:
                    idx = random.randint(0, len(child_sol)-1)
                    child_sol[idx] = random.randint(0, n_blocks - 1)
                    
            # Repair: Ensure coverage
            child_sol = _greedy_complete(child_sol, precomp, deadline)
            
            new_population.append(child_sol)
            
        population = new_population
        
    # Final polish on best solution
    if best_sol is not None:
        best_sol = _local_search(best_sol, precomp, deadline)
        
    return best_sol

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

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    precomp = _precompute(v, k, t)
    
    # Run GA
    best_blocks = _ga_search(v, k, t, precomp, deadline)
    
    if best_blocks is None:
        # Fallback: Greedy
        best_blocks = []
        # Simple greedy fallback
        uncovered = set(itertools.combinations(range(v), t))
        while uncovered and time.time() < deadline - 5.0:
            target = next(iter(uncovered))
            best_block = None
            best_gain = -1
            remaining = [x for x in range(v) if x not in target]
            need = k - t
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            if best_block is None:
                break
            best_blocks.append(best_block)
            uncovered.difference_update(itertools.combinations(best_block, t))
            
    # Convert indices to actual blocks
    tsubs, tsub_idx, blocks_list, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, ts_cover_count = precomp
    final_blocks = []
    seen = set()
    for bi in best_blocks:
        b = blocks_list[bi]
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            final_blocks.append(list(b))
            
    return final_blocks
