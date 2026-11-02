import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build incidence structure
    # block_to_tsubsets[block_idx] = array of t-subset indices covered by this block
    # tsub_to_blocks[tsub_idx] = list of block indices that cover this t-subset
    block_to_tsubsets = []
    tsub_to_blocks = [[] for _ in range(N)]

    for bi, b in enumerate(all_blocks):
        covers = []
        for s in itertools.combinations(b, t):
            ti = t_idx[s]
            covers.append(ti)
            tsub_to_blocks[ti].append(bi)
        block_to_tsubsets.append(np.array(covers, dtype=np.int32))

    # Convert tsub_to_blocks to arrays for faster access
    tsub_to_blocks_arrays = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]

    def greedy_with_frequency(seed=42):
        """
        Multi-start greedy set cover with frequency-based tie-breaking.
        Uses a 'regret' style approach: prioritize t-subsets that have fewer
        covering blocks (harder to cover), and among blocks covering them,
        prefer those with higher marginal coverage.
        """
        random.seed(seed)
        uncovered = np.ones(N, dtype=bool)
        selected = []
        used_blocks = set()
        
        # Precompute block sizes for tie-breaking
        block_sizes = np.array([len(c) for c in block_to_tsubsets], dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline - 20:
                break

            # Find the rarest uncovered t-subset (fewest covering blocks that aren't used)
            # This is expensive, so we sample
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            
            # Sample a few uncovered t-subsets and pick the one with fewest available blocks
            sample_size = min(10, len(unc_idx))
            if sample_size < len(unc_idx):
                sample = random.sample(list(unc_idx), sample_size)
            else:
                sample = list(unc_idx)
            
            best_target = -1
            min_available = float('inf')
            for ti in sample:
                available_count = 0
                for bi in tsub_to_blocks[ti]:
                    if bi not in used_blocks:
                        available_count += 1
                        if available_count > min_available:
                            break
                if available_count < min_available:
                    min_available = available_count
                    best_target = ti
                    if min_available == 1:
                        break

            if best_target == -1:
                best_target = random.choice(unc_idx)

            # Among blocks covering this target, pick the one with highest marginal gain
            candidates = tsub_to_blocks[best_target]
            best_bi = -1
            best_gain = -1
            
            # Shuffle candidates for diversity
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            for bi in cand_list:
                if bi in used_blocks:
                    continue
                cov = block_to_tsubsets[bi]
                gain = np.sum(uncovered[cov])
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    if gain > 80:  # Early exit
                        break

            if best_bi == -1:
                # Fallback: pick any available block
                for bi in candidates:
                    if bi not in used_blocks:
                        best_bi = bi
                        break
                if best_bi == -1:
                    break

            selected.append(best_bi)
            used_blocks.add(best_bi)
            uncovered[block_to_tsubsets[best_bi]] = False

        # Repair: add blocks for any remaining uncovered t-subsets
        if not np.all(~uncovered):
            for ti in np.where(uncovered)[0]:
                for bi in tsub_to_blocks[ti]:
                    if bi not in used_blocks:
                        selected.append(bi)
                        used_blocks.add(bi)
                        break

        return selected

    def local_search_remove(blocks, max_time=30.0):
        """
        Try to remove redundant blocks from the solution.
        A block is removable if no t-subset it covers is covered by exactly 1 block.
        """
        start_time = time.time()
        current = list(blocks)
        current_size = len(current)
        best = list(current)
        best_size = current_size

        # Build coverage count
        cover_count = np.zeros(N, dtype=np.int32)
        for bi in current:
            cover_count[block_to_tsubsets[bi]] += 1

        # Randomly order blocks to try removal
        indices = list(range(len(current)))
        random.shuffle(indices)

        removals = 0
        for idx in indices:
            if time.time() > start_time + max_time:
                break
            if idx >= len(current):
                continue

            bi_to_remove = current[idx]
            cov = block_to_tsubsets[bi_to_remove]

            # Check if any t-subset in cov has count == 1
            # Use vectorized check
            if np.any(cover_count[cov] == 1):
                continue

            # Safe to remove
            cover_count[cov] -= 1
            current.pop(idx)
            current_size -= 1
            removals += 1

            if current_size < best_size:
                best_size = current_size
                best = list(current)

        return best, best_size

    def simulated_annealing(blocks, max_time=60.0, seed=0):
        """
        Simulated annealing to minimize block count.
        Moves: 
        1. Remove a block (if valid)
        2. Swap a block for another
        3. Add a block (if it enables removals later)
        """
        random.seed(seed)
        start_time = time.time()
        current = list(blocks)
        current_size = len(current)
        best = list(current)
        best_size = current_size

        # Build coverage count
        cover_count = np.zeros(N, dtype=np.int32)
        for bi in current:
            cover_count[block_to_tsubsets[bi]] += 1

        current_set = set(current)
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.1
        n_iterations = 20000
        
        for it in range(n_iterations):
            if time.time() > start_time + max_time:
                break
            
            T = T_start * (T_end / T_start) ** (it / n_iterations)
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Try removal
                if not current:
                    continue
                idx = random.randint(0, len(current) - 1)
                bi = current[idx]
                cov = block_to_tsubsets[bi]
                
                if np.any(cover_count[cov] == 1):
                    continue
                
                # Accept removal always (improvement)
                cover_count[cov] -= 1
                current.pop(idx)
                current_set.remove(bi)
                current_size -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best = list(current)

            elif move_type < 0.7:
                # Try swap
                if not current:
                    continue
                idx = random.randint(0, len(current) - 1)
                bi_old = current[idx]
                cov_old = block_to_tsubsets[bi_old]
                
                # Find t-subsets that would become uncovered
                at_risk = [ti for ti in cov_old if cover_count[ti] == 1]
                
                if not at_risk:
                    # Just remove
                    cover_count[cov_old] -= 1
                    current.pop(idx)
                    current_set.remove(bi_old)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best = list(current)
                    continue
                
                # Find candidate blocks covering all at_risk t-subsets
                candidates = set(tsub_to_blocks[at_risk[0]])
                for ti in at_risk[1:]:
                    candidates &= set(tsub_to_blocks[ti])
                candidates -= current_set
                
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                if len(cand_list) > 30:
                    cand_list = random.sample(cand_list, 30)
                
                best_cand = -1
                best_score = -1
                for cand in cand_list:
                    new_cov = block_to_tsubsets[cand]
                    # Score: how many additional t-subsets would be covered
                    score = np.sum((cover_count[new_cov] == 0))
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                
                if best_cand == -1:
                    continue
                
                # Evaluate swap
                new_cover_count = cover_count.copy()
                new_cover_count[cov_old] -= 1
                new_cover_count[block_to_tsubsets[best_cand]] += 1
                
                if np.any(new_cover_count < 0):
                    continue
                
                # Compute delta in "wasted coverage"
                old_waste = np.sum(np.maximum(cover_count - 1, 0))
                new_waste = np.sum(np.maximum(new_cover_count - 1, 0))
                delta = new_waste - old_waste
                
                # Accept if delta <= 0 or with probability exp(-delta/T)
                if delta <= 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                    cover_count[cov_old] -= 1
                    cover_count[block_to_tsubsets[best_cand]] += 1
                    current[idx] = best_cand
                    current_set.remove(bi_old)
                    current_set.add(best_cand)
                    # Size stays same, but coverage quality improved
                    if len(current) < best_size:
                        best_size = len(current)
                        best = list(current)

            else:
                # Try add (less common, to escape local optima)
                if len(current) > 400:  # Don't add too many
                    # Pick a random t-subset with low coverage
                    low_cov = np.where(cover_count <= 1)[0]
                    if len(low_cov) == 0:
                        continue
                    ti = random.choice(low_cov)
                    
                    candidates = tsub_to_blocks[ti]
                    if not candidates:
                        continue
                    
                    cand_list = [c for c in candidates if c not in current_set]
                    if not cand_list:
                        continue
                    
                    bi_new = random.choice(cand_list)
                    new_cov = block_to_tsubsets[bi_new]
                    
                    new_cover_count = cover_count.copy()
                    new_cover_count[new_cov] += 1
                    
                    old_waste = np.sum(np.maximum(cover_count - 1, 0))
                    new_waste = np.sum(np.maximum(new_cover_count - 1, 0))
                    delta = 1 + (new_waste - old_waste)  # +1 for the added block
                    
                    if delta <= 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                        cover_count[new_cov] += 1
                        current.append(bi_new)
                        current_set.add(bi_new)
                        current_size += 1

        # Final cleanup
        final, final_size = local_search_remove(current, max_time=5.0)
        if final_size < best_size:
            best_size = final_size
            best = final

        return best, best_size

    # Main algorithm: multi-start greedy + SA refinement
    best_solution = None
    best_size = float('inf')
    
    # Phase 1: Multiple greedy starts
    seeds = [42, 137, 251, 389, 523, 671, 811, 967]
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        
        sol = greedy_with_frequency(seed=seed)
        if not sol:
            continue
        
        # Quick removal pass
        cleaned, cleaned_size = local_search_remove(sol, max_time=5.0)
        if cleaned_size < best_size:
            best_size = cleaned_size
            best_solution = cleaned
    
    if best_solution is None:
        # Fallback
        sol = greedy_with_frequency(seed=42)
        if sol:
            best_solution = sol
            best_size = len(sol)
        else:
            return [list(b) for b in all_blocks[:100]]

    # Phase 2: SA refinement on the best solution
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 10:
        # Run SA on the best solution
        sa_sol, sa_size = simulated_annealing(best_solution, max_time=min(remaining_time, 60.0), seed=123)
        if sa_size < best_size:
            best_size = sa_size
            best_solution = sa_sol
        
        # Run SA on a few other solutions if time permits
        remaining_time = deadline - 2 - time.time()
        if remaining_time > 15:
            for seed in [777, 888]:
                if time.time() > deadline - 2:
                    break
                sol = greedy_with_frequency(seed=seed)
                if sol and len(sol) < best_size + 10:
                    cleaned, _ = local_search_remove(sol, max_time=3.0)
                    sa_sol, sa_size = simulated_annealing(cleaned, max_time=min(remaining_time / 2, 30.0), seed=seed)
                    if sa_size < best_size:
                        best_size = sa_size
                        best_solution = sa_sol

    # Final cleanup
    if time.time() < deadline - 1:
        final, final_size = local_search_remove(best_solution, max_time=3.0)
        if final_size < best_size:
            best_size = final_size
            best_solution = final

    # Convert to point lists
    final_blocks = [list(all_blocks[bi]) for bi in best_solution]
    
    # Verify validity (quick check)
    covered = np.zeros(N, dtype=bool)
    for bi in best_solution:
        covered[block_to_tsubsets[bi]] = True
    
    if not np.all(covered):
        # Repair
        for ti in np.where(~covered)[0]:
            for bi in tsub_to_blocks[ti]:
                if bi not in set(best_solution):
                    best_solution.append(bi)
                    covered[block_to_tsubsets[bi]] = True
                    break
        final_blocks = [list(all_blocks[bi]) for bi in best_solution]

    return final_blocks
