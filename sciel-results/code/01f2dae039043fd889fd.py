import numpy as np
import itertools
import time
import random

def construct(v, k, t):
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
    
    # Precompute subset_to_blocks as sets for fast intersection in swap
    subset_to_blocks_sets = [set(blocks) for blocks in subset_to_blocks]
    
    # Precompute block size (number of t-subsets per block)
    block_sizes = np.array([len(s) for s in block_to_subsets], dtype=np.int32)
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = block_sizes.copy()
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            # Use a smoother scoring with slight noise
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            uncovered[newly_covered] = False
            # Vectorized gain update
            for s in newly_covered:
                blocks_covering = subset_block_arrays[s]
                gains[blocks_covering] -= 1
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            blocks_covering = subset_block_arrays[s_idx]
            gains[blocks_covering] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            current_uncovered[newly_covered] = False
            for s in newly_covered:
                blocks_covering = subset_block_arrays[s]
                gains[blocks_covering] -= 1
                    
        return selected_blocks

    def minimize_with_swaps(blocks_indices, coverage_counts, time_limit, random_seed=None):
        """Local search with dynamic clause weighting (Max-SAT style)."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks, coverage_counts.copy()
            
        # Work with a copy of coverage counts
        cover_count = coverage_counts.copy()
        
        # Use a set for O(1) membership testing
        current_blocks_set = set(current_blocks)
        
        # Dynamic clause weights: weight for each t-subset based on how tightly it's covered
        # Higher weight = more "critical" clause
        clause_weights = np.ones(num_subsets, dtype=np.float64)
        
        improved = True
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 30
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Reset weights and try again with fresh randomization
                clause_weights = np.ones(num_subsets, dtype=np.float64)
                no_improve_count = 0
            
            # Deletion pass with weighted scoring
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if i >= len(current_blocks):
                    continue
                if (time.time() - start_time) > time_limit:
                    break
                b_idx = current_blocks[i]
                b_subs = block_subset_arrays[b_idx]
                
                # Check if block is redundant AND compute weighted cost of removing it
                is_redundant = True
                removal_cost = 0.0
                for s_idx in b_subs:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                    else:
                        # This subset would lose one coverage; add its weight to cost
                        removal_cost += clause_weights[s_idx]
                
                if is_redundant:
                    for s_idx in b_subs:
                        cover_count[s_idx] -= 1
                        # Update clause weights: less covered subsets become more critical
                        if cover_count[s_idx] == 1:
                            clause_weights[s_idx] *= 1.2
                    current_blocks_set.discard(b_idx)
                    current_blocks.pop(i)
                    improved = True
                    no_improve_count = 0
                    break  # Restart scan after removal
            
            # Swap pass with dynamic weighting
            if (time.time() - start_time) > time_limit:
                break
                
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            swaps_done = 0
            max_swaps = 80
            
            for i in indices:
                if (time.time() - start_time) > time_limit:
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(current_blocks):
                    continue
                    
                old_block = current_blocks[i]
                old_subs = block_subset_arrays[old_block]
                
                # Find unique subsets covered by old_block
                unique_subs = []
                for s_idx in old_subs:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    # Block is not critical - try swapping with a random block
                    # weighted by how much it helps with tight clauses
                    new_block = rng.randint(0, num_blocks - 1)
                    if new_block == old_block:
                        continue
                    if new_block in current_blocks_set:
                        continue
                        
                    # Evaluate swap quality using clause weights
                    old_weight = sum(clause_weights[s_idx] for s_idx in old_subs)
                    new_weight = sum(clause_weights[s_idx] for s_idx in block_subset_arrays[new_block])
                    
                    # Accept if new block covers more weighted clauses
                    if new_weight > old_weight or rng.random() < 0.3:
                        for s_idx in old_subs:
                            cover_count[s_idx] -= 1
                            if cover_count[s_idx] == 1:
                                clause_weights[s_idx] *= 1.2
                        new_subs = block_subset_arrays[new_block]
                        for s_idx in new_subs:
                            cover_count[s_idx] += 1
                            if cover_count[s_idx] == 2:
                                clause_weights[s_idx] *= 0.8
                        current_blocks_set.discard(old_block)
                        current_blocks_set.add(new_block)
                        current_blocks[i] = new_block
                        swaps_done += 1
                        improved = True
                        no_improve_count = 0
                else:
                    # Block is critical - find candidates that cover all unique_subs
                    candidates = None
                    for s_idx in unique_subs:
                        if candidates is None:
                            candidates = subset_to_blocks_sets[s_idx].copy()
                        else:
                            candidates.intersection_update(subset_to_blocks_sets[s_idx])
                            if not candidates:
                                break
                    
                    if candidates:
                        candidates.discard(old_block)
                        if candidates:
                            # Fast pre-filter: remove candidates already in current set
                            candidates = [c for c in candidates if c not in current_blocks_set]
                            
                            if candidates:
                                # Score candidates by how well they cover tight clauses
                                scored_candidates = []
                                for cand in candidates:
                                    cand_subs = block_subset_arrays[cand]
                                    score = sum(clause_weights[s_idx] for s_idx in cand_subs)
                                    scored_candidates.append((score, cand))
                            
                            if scored_candidates:
                                scored_candidates.sort(reverse=True)
                                # Pick from top candidates with some randomness
                                top_k = min(5, len(scored_candidates))
                                new_block = scored_candidates[rng.randint(0, top_k - 1)][1]
                                
                                # Perform swap
                                for s_idx in old_subs:
                                    cover_count[s_idx] -= 1
                                    if cover_count[s_idx] == 1:
                                        clause_weights[s_idx] *= 1.2
                                new_subs = block_subset_arrays[new_block]
                                for s_idx in new_subs:
                                    cover_count[s_idx] += 1
                                    if cover_count[s_idx] == 2:
                                        clause_weights[s_idx] *= 0.8
                                current_blocks_set.discard(old_block)
                                current_blocks_set.add(new_block)
                                current_blocks[i] = new_block
                                swaps_done += 1
                                improved = True
                                no_improve_count = 0
        
        # Final weight decay to avoid over-concentration
        if np.max(clause_weights) > 0:
            clause_weights = clause_weights / np.max(clause_weights)
        
        return current_blocks, cover_count

    def compute_coverage(blocks_indices):
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            b_subs = block_subset_arrays[b_idx]
            cover_count[b_subs] += 1
        return cover_count

    def try_two_opt(blocks_indices, coverage_counts, time_limit, random_seed=None):
        """Try replacing pairs of blocks with single blocks that cover all their unique subsets."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if len(current_blocks) < 2:
            return current_blocks, coverage_counts.copy()
            
        cover_count = coverage_counts.copy()
        current_blocks_set = set(current_blocks)
        
        improved = True
        start_time = time.time()
        no_improve = 0
        max_no_improve = 10
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            no_improve += 1
            if no_improve > max_no_improve:
                break
            
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if (time.time() - start_time) > time_limit:
                    break
                if i >= len(current_blocks):
                    continue
                    
                # Find a partner j != i
                j_candidates = [j for j in range(len(current_blocks)) if j != i]
                if not j_candidates:
                    continue
                j = rng.choice(j_candidates)
                
                b1 = current_blocks[i]
                b2 = current_blocks[j]
                subs1 = block_subset_arrays[b1]
                subs2 = block_subset_arrays[b2]
                
                # Find unique subsets of b1 (only covered by b1 among current blocks)
                unique1 = []
                for s_idx in subs1:
                    if cover_count[s_idx] == 1:
                        unique1.append(s_idx)
                
                # Find unique subsets of b2
                unique2 = []
                for s_idx in subs2:
                    if cover_count[s_idx] == 1:
                        unique2.append(s_idx)
                
                if not unique1 and not unique2:
                    # Neither block is critical; try removing one
                    # Just remove b1 (cheaper check)
                    for s_idx in subs1:
                        if cover_count[s_idx] == 1:
                            break
                    else:
                        for s_idx in subs1:
                            cover_count[s_idx] -= 1
                        current_blocks_set.discard(b1)
                        # Remove from list carefully
                        current_blocks.pop(i)
                        if j > i:
                            j -= 1
                        improved = True
                        no_improve = 0
                        break
                    continue
                
                # Find candidates that cover all unique subsets from both blocks
                all_unique = set(unique1) | set(unique2)
                if not all_unique:
                    continue
                    
                candidates = None
                for s_idx in all_unique:
                    if candidates is None:
                        candidates = subset_to_blocks_sets[s_idx].copy()
                    else:
                        candidates.intersection_update(subset_to_blocks_sets[s_idx])
                        if not candidates:
                            break
                
                if not candidates:
                    continue
                    
                candidates.discard(b1)
                candidates.discard(b2)
                
                # Filter out already-selected blocks
                candidates = [c for c in candidates if c not in current_blocks_set]
                
                if not candidates:
                    continue
                
                # Score candidates
                scored = []
                for cand in candidates:
                    cand_subs = block_subset_arrays[cand]
                    # Count how many of the unique subsets it covers (should be all)
                    # Score by additional coverage
                    score = 0
                    for s_idx in cand_subs:
                        if cover_count[s_idx] >= 1:
                            score += 1
                    scored.append((score, cand))
                
                if not scored:
                    continue
                    
                scored.sort(reverse=True)
                best_score, new_block = scored[0]
                
                # Check: does new_block cover all unique subsets?
                new_subs = block_subset_arrays[new_block]
                new_subs_set = set(new_subs.tolist())
                if not all_unique.issubset(new_subs_set):
                    continue
                
                # Perform the replacement: remove b1 and b2, add new_block
                for s_idx in subs1:
                    cover_count[s_idx] -= 1
                for s_idx in subs2:
                    cover_count[s_idx] -= 1
                for s_idx in new_subs:
                    cover_count[s_idx] += 1
                
                current_blocks_set.discard(b1)
                current_blocks_set.discard(b2)
                current_blocks_set.add(new_block)
                
                # Remove from list (careful with indices)
                # Remove the higher index first
                if i > j:
                    i, j = j, i
                current_blocks.pop(i)
                current_blocks.pop(j)
                current_blocks.append(new_block)
                
                improved = True
                no_improve = 0
        
        return current_blocks, cover_count

    best_blocks = None
    best_len = float('inf')
    best_coverage = None
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    trial = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
            
        # Strategy 1: Full Greedy + Local Search (deletion + swap)
        greedy_res = run_greedy_full(trial)
        coverage = compute_coverage(greedy_res)
        
        # Adaptive time limit for local search
        ls_time = min(4.0, max(1.5, remaining * 0.18))
        min_res, min_cov = minimize_with_swaps(greedy_res, coverage, time_limit=ls_time, random_seed=trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            best_coverage = min_cov
            
        # Strategy 2: Perturb Best + Repair + Local Search
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 15:
            perturbed = list(best_blocks)
            # Vary the number of blocks removed based on trial - more aggressive
            num_remove = min(max(3, len(perturbed) // (8 + trial % 8)), len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    b_subs = block_subset_arrays[b_idx]
                    cover_count_temp[b_subs] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                # Compute coverage for new set
                new_coverage = compute_coverage(new_set)
                
                ls_time2 = min(4.0, max(1.5, (deadline - time.time()) * 0.18))
                min_res2, min_cov2 = minimize_with_swaps(new_set, new_coverage, time_limit=ls_time2, random_seed=trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2
                    best_coverage = min_cov2

        # Strategy 3: Pure swap-based diversification on best solution
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 10:
            ls_time3 = min(5.0, max(2.0, (deadline - time.time()) * 0.22))
            swap_res, swap_cov = minimize_with_swaps(best_blocks, best_coverage, time_limit=ls_time3, random_seed=trial + 5000)
            
            if len(swap_res) < best_len:
                best_len = len(swap_res)
                best_blocks = swap_res
                best_coverage = swap_cov
        
        # Strategy 4: 2-opt pair replacement
        if best_blocks is not None and len(best_blocks) > 30 and time.time() < deadline - 8:
            two_opt_time = min(3.0, max(1.0, (deadline - time.time()) * 0.15))
            two_opt_res, two_opt_cov = try_two_opt(best_blocks, best_coverage, time_limit=two_opt_time, random_seed=trial + 9000)
            
            if len(two_opt_res) < best_len:
                best_len = len(two_opt_res)
                best_blocks = two_opt_res
                best_coverage = two_opt_cov
                # Follow up with single-swap local search on the improved solution
                if time.time() < deadline - 5:
                    ls_time4 = min(3.0, max(1.0, (deadline - time.time()) * 0.2))
                    swap_res4, swap_cov4 = minimize_with_swaps(best_blocks, best_coverage, time_limit=ls_time4, random_seed=trial + 12000)
                    if len(swap_res4) < best_len:
                        best_len = len(swap_res4)
                        best_blocks = swap_res4
                        best_coverage = swap_cov4

        trial += 1
        
    # Final polish: aggressive deletion pass on best solution
    if best_blocks is not None and time.time() < deadline - 5:
        final_cov = best_coverage.copy()
        final_blocks = list(best_blocks)
        final_set = set(final_blocks)
        # Multiple deletion passes
        for _ in range(5):
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for i in range(len(final_blocks)):
                    b_idx = final_blocks[i]
                    b_subs = block_subset_arrays[b_idx]
                    redundant = True
                    for s_idx in b_subs:
                        if final_cov[s_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        for s_idx in b_subs:
                            final_cov[s_idx] -= 1
                        final_set.discard(b_idx)
                        final_blocks.pop(i)
                        changed = True
                        break
        if len(final_blocks) < best_len:
            best_len = len(final_blocks)
            best_blocks = final_blocks
            best_coverage = final_cov
        
        # Final 2-opt pass
        if len(best_blocks) > 20 and time.time() < deadline - 3:
            final_two_opt, final_two_opt_cov = try_two_opt(best_blocks, best_coverage, time_limit=min(2.0, deadline - time.time() - 2), random_seed=99999)
            if len(final_two_opt) < best_len:
                best_len = len(final_two_opt)
                best_blocks = final_two_opt
                best_coverage = final_two_opt_cov
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        coverage = compute_coverage(greedy_res)
        best_blocks, best_coverage = minimize_with_swaps(greedy_res, coverage, time_limit=5.0, random_seed=0)
        
    return [list(all_blocks[i]) for i in best_blocks]
