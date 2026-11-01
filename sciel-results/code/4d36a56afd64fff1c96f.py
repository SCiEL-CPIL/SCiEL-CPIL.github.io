import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, compute the set of t-subset indices it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = tuple(sorted(t_subset_to_idx[s] for s in itertools.combinations(block, t)))
        block_t_indices.append(indices)
    
    # For each t-subset, which blocks cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Precompute degrees for scarcity heuristic
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]

    # Precompute block masks as bitsets for fast coverage gain computation
    block_masks = [0] * num_k_subsets
    for b_idx, indices in enumerate(block_t_indices):
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        block_masks[b_idx] = mask

    # Helper: minimize a given set of blocks (remove redundant blocks)
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return active_blocks_set, cover_counts
    
    # Helper: Standard Greedy Set Cover
    def greedy_cover():
        uncovered_mask = (1 << num_t_subsets) - 1  # all bits set
        gain = [len(block_t_indices[i]) for i in range(num_k_subsets)]
        selected_blocks = []
        
        while uncovered_mask != 0:
            # Find max gain
            max_gain = max(gain)
            if max_gain == 0:
                break
            
            # Find all blocks with max gain, pick randomly
            candidates = [i for i in range(num_k_subsets) if gain[i] == max_gain]
            best_block = random.choice(candidates)
            
            selected_blocks.append(best_block)
            
            # Update gains using bitset operations
            new_covered = block_masks[best_block] & uncovered_mask
            uncovered_mask ^= new_covered  # clear newly covered bits
            
            # Update gains: for each t_idx in new_covered bits, decrement gain for blocks covering it
            newly_covered_indices = []
            temp = new_covered
            while temp:
                # Find lowest set bit
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                newly_covered_indices.append(idx)
                temp ^= lsb
            
            for t_idx in newly_covered_indices:
                for b_idx in t_subset_blocks[t_idx]:
                    gain[b_idx] -= 1
                        
        return selected_blocks

    # Helper: Greedy Packing Initialization (Inspiration)
    def greedy_packing_init():
        """
        Phase 1: Select a maximal independent set of blocks (no shared t-subsets)
        prioritizing blocks that cover rare t-subsets.
        Phase 2: Fill remaining gaps with standard greedy.
        """
        pool = set(range(num_k_subsets))
        core_blocks = []
        uncovered_mask = [1] * num_t_subsets  # 1 if uncovered
        
        # Precompute initial scarcity weight for each block
        # weight[b] = sum of 1/degree[t] for t in block
        # We recompute dynamic score in the loop, but this helps if we needed static weights.
        # Here we compute dynamic score: (uncovered_count * 100) + scarcity_sum
        
        while pool:
            best_block = -1
            max_score = -1.0
            
            # To speed up, we iterate through the pool.
            # 50k blocks is small enough for a few iterations, but this loop runs ~100-200 times.
            # 50k * 200 = 10M ops. Acceptable.
            for b_idx in pool:
                cov_uncovered = 0
                scarcity_sum = 0.0
                # Optimization: If a block is in the pool, it doesn't share t-subsets with core_blocks.
                # So all its t-subsets are potentially uncovered? No, other pool blocks might cover them.
                # But core_blocks are independent, so they don't cover these.
                # Wait, the definition of "independent" here is "do not share ANY t-subset".
                # So if B is in pool, it shares no t-subset with any block in core_blocks.
                # Therefore, all t-subsets of B are NOT covered by core_blocks.
                # They MIGHT be covered by other pool blocks, but we haven't selected them yet.
                # So for the purpose of "what does this block add to the core", it adds all its t-subsets
                # that are not yet covered by *other selected core blocks*. Since core is independent,
                # no core block covers any t-subset of B.
                # So cov_uncovered is just the number of t-subsets of B that are currently uncovered
                # by the *global* state? No, the goal is to build a packing.
                # The score should reflect how "good" this block is for the packing.
                # A good packing block covers many t-subsets that are "hard" to cover later.
                # Scarcity is a good proxy.
                
                # However, we also want to ensure we cover as much as possible.
                # Let's stick to the inspiration's logic:
                # score = cov_uncovered * 100 + scarcity_sum
                # But since core is independent, cov_uncovered relative to core is always len(block_t_indices[b_idx]).
                # The "uncovered" part in the inspiration likely referred to the global coverage state
                # if it was a standard set cover. But for packing, we are just selecting distinct blocks.
                # Let's use scarcity primarily, and size as a tie-breaker.
                
                for t_idx in block_t_indices[b_idx]:
                    if uncovered_mask[t_idx]:
                        cov_uncovered += 1
                        scarcity_sum += 1.0 / t_subset_degrees[t_idx]
                    else:
                        # If already covered by previous core blocks? 
                        # Impossible if core is independent.
                        pass
                
                # Actually, if we enforce independence, we don't need to check uncovered_mask 
                # against core blocks because they are disjoint.
                # We only care about scarcity.
                # Let's refine: The inspiration says "Prefer blocks that cover rare t-subsets".
                # Let's just use scarcity_sum * 100 + len(block_t_indices[b_idx])
                
                # Re-evaluating the inspiration code provided:
                # It checks `if uncovered_mask[t_idx]:` and accumulates.
                # It updates `uncovered_mask[t_idx] = 0` after selection.
                # This implies it's not strictly enforcing independence in the mask, 
                # but it *does* remove neighbors from the pool.
                # If neighbors are removed, then any t-subset of a pool block 
                # is NOT covered by any core block.
                # So `uncovered_mask` for pool blocks should always be 1 for all their t-subsets.
                # Unless... the "neighbors" removal is incomplete or I'm misinterpreting.
                # "Remove best_block and its conflicts from pool".
                # Conflicts = blocks sharing ANY t-subset.
                # So yes, if B is in pool, it shares no t-subset with core.
                # So all its t-subsets are uncovered by core.
                # So cov_uncovered is constant for a given block.
                # The variable part is scarcity_sum.
                
                score = scarcity_sum * 100.0 + len(block_t_indices[b_idx])
                
                if score > max_score:
                    max_score = score
                    best_block = b_idx
            
            if best_block == -1:
                break
            
            core_blocks.append(best_block)
            
            # Update uncovered mask (for the second phase greedy, though we can just compute it later)
            for t_idx in block_t_indices[best_block]:
                uncovered_mask[t_idx] = 0
            
            # Remove best_block and its conflicts from pool
            neighbors_to_remove = set()
            for t_idx in block_t_indices[best_block]:
                neighbors_to_remove.update(t_subset_blocks[t_idx])
            
            pool.difference_update(neighbors_to_remove)

        # Phase 2: Solve Remaining Set Cover
        active_blocks = set(core_blocks)
        cover_count = [0] * num_t_subsets
        
        for b_idx in active_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        still_uncovered = [i for i in range(num_t_subsets) if cover_count[i] == 0]
        
        if still_uncovered:
            residual_uncovered = set(still_uncovered)
            
            # Standard Greedy for the rest
            while residual_uncovered:
                best_block = -1
                max_gain = 0
                
                for b_idx in range(num_k_subsets):
                    if b_idx in active_blocks:
                        continue
                    gain = 0
                    for t_idx in block_t_indices[b_idx]:
                        if t_idx in residual_uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_block = b_idx
                        
                if best_block == -1 or max_gain == 0:
                    break
                    
                active_blocks.add(best_block)
                for t_idx in block_t_indices[best_block]:
                    if t_idx in residual_uncovered:
                        residual_uncovered.remove(t_idx)
                    cover_count[t_idx] += 1

        return list(active_blocks), cover_count

    # Helper: Local Search with swap + minimize
    def local_search(active_blocks, cover_count, time_limit):
        start_time_ls = time.time()
        
        while time.time() - start_time_ls < time_limit:
            improved = False
            current_blocks = list(active_blocks)
            random.shuffle(current_blocks)
            
            for B in current_blocks:
                if time.time() - start_time_ls >= time_limit:
                    break
                if B not in active_blocks:
                    continue
                
                # Find t-subsets uniquely covered by B
                uncovered_by_B = []
                for t_idx in block_t_indices[B]:
                    if cover_count[t_idx] == 1:
                        uncovered_by_B.append(t_idx)
                
                if not uncovered_by_B:
                    continue

                # Find candidate blocks that cover all of uncovered_by_B
                uncovered_by_B.sort(key=lambda x: len(t_subset_blocks[x]))
                
                candidates = None
                for t_idx in uncovered_by_B:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Filter: must not be B, must not already be in active_blocks
                valid_candidates = [c for c in candidates if c != B and c not in active_blocks]
                
                if not valid_candidates:
                    continue
                
                # Perform swap
                B_new = random.choice(valid_candidates)
                
                active_blocks.remove(B)
                active_blocks.add(B_new)
                
                for t_idx in block_t_indices[B]:
                    cover_count[t_idx] -= 1
                for t_idx in block_t_indices[B_new]:
                    cover_count[t_idx] += 1
                
                # Minimize after swap
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                improved = True
                
            if not improved:
                break
        
        return active_blocks, cover_count
    
    # Helper: Simulated Annealing for block replacement
    def simulated_annealing(active_blocks, cover_count, time_limit):
        start_time_sa = time.time()
        current_temp = 50.0  # Initial temperature
        cooling_rate = 0.95
        min_temp = 0.5
        
        while time.time() - start_time_sa < time_limit and current_temp > min_temp:
            # Pick a random block to potentially replace
            current_blocks = list(active_blocks)
            random.shuffle(current_blocks)
            
            improved_this_round = False
            
            for B in current_blocks:
                if time.time() - start_time_sa >= time_limit:
                    break
                if B not in active_blocks:
                    continue
                
                # Find t-subsets uniquely covered by B
                uncovered_by_B = []
                for t_idx in block_t_indices[B]:
                    if cover_count[t_idx] == 1:
                        uncovered_by_B.append(t_idx)
                
                if not uncovered_by_B:
                    # B is redundant, remove it
                    active_blocks.remove(B)
                    for t_idx in block_t_indices[B]:
                        cover_count[t_idx] -= 1
                    improved_this_round = True
                    continue

                # Find candidate blocks that cover all of uncovered_by_B
                uncovered_by_B.sort(key=lambda x: len(t_subset_blocks[x]))
                
                candidates = None
                for t_idx in uncovered_by_B:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Filter: must not be B, must not already be in active_blocks
                valid_candidates = [c for c in candidates if c != B and c not in active_blocks]
                
                if not valid_candidates:
                    continue
                
                # Evaluate candidates based on how many additional t-subsets they cover
                best_candidate = None
                best_score = -float('inf')
                
                # Compute current coverage of B
                B_coverage_mask = block_masks[B]
                
                for C in valid_candidates:
                    # Score: number of new t-subsets covered by C that aren't covered by B
                    C_mask = block_masks[C]
                    new_coverage = C_mask & ~B_coverage_mask
                    score = new_coverage.bit_count()
                    
                    # Add some randomness to break ties
                    score += random.uniform(-1, 1)
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = C
                
                if best_candidate is None:
                    continue
                
                # Compute the change in uncovered count
                C_mask = block_masks[best_candidate]
                
                # How many of uncovered_by_B does C cover?
                covered_by_C = 0
                for t_idx in uncovered_by_B:
                    if (C_mask >> t_idx) & 1:
                        covered_by_C += 1
                
                # Net change in uncovered: len(uncovered_by_B) - covered_by_C
                delta_uncovered = len(uncovered_by_B) - covered_by_C
                
                # Accept if improving or with probability exp(-delta/temp)
                if delta_uncovered <= 0 or random.random() < np.exp(-delta_uncovered / current_temp):
                    active_blocks.remove(B)
                    active_blocks.add(best_candidate)
                    
                    for t_idx in block_t_indices[B]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_t_indices[best_candidate]:
                        cover_count[t_idx] += 1
                    
                    improved_this_round = True
            
            # Cool down
            current_temp *= cooling_rate
        
        # Final minimize
        active_blocks, cover_count = minimize(active_blocks, cover_count)
        
        return active_blocks, cover_count
    
    # Main Loop: Multi-Start
    best_blocks = None
    best_size = float('inf')
    
    # Alternate between standard greedy and packing init
    use_packing = False
    
    while time.time() < deadline - 2.0:
        if use_packing:
            selected, initial_cover_count = greedy_packing_init()
            active_blocks = set(selected)
            cover_count = initial_cover_count
        else:
            selected = greedy_cover()
            active_blocks = set(selected)
            cover_count = [0] * num_t_subsets
            for b_idx in active_blocks:
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] += 1
        
        use_packing = not use_packing
        
        # Minimize
        active_blocks, cover_count = minimize(active_blocks, cover_count)
        
        # Local Search with remaining time budget
        remaining_time = deadline - 2.0 - time.time()
        if remaining_time > 0.5:
            # Give local search and simulated annealing time
            ls_time = max(0.2, remaining_time * 0.25)
            sa_time = max(0.2, remaining_time * 0.25)
            
            active_blocks, cover_count = local_search(active_blocks, cover_count, ls_time)
            active_blocks, cover_count = simulated_annealing(active_blocks, cover_count, sa_time)
        
        # Update Best
        current_size = len(active_blocks)
        if current_size < best_size:
            best_size = current_size
            best_blocks = list(active_blocks)
            
            # Lower bound is 131; if we hit it, stop
            if best_size <= 131:
                break
    
    if best_blocks is None:
        selected = greedy_cover()
        best_blocks = selected

    result = [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
    return result

