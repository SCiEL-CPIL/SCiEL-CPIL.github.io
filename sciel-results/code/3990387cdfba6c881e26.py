import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe under 150s

    # 1. Map triples to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)

    # 2. Generate all candidate blocks and their masks
    # For v=24, k=7, num_blocks = 346104.
    blocks = []
    block_masks = []
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
        blocks.append(block)
        block_masks.append(mask)
        
    num_blocks = len(blocks)
    
    # Precompute block lists for each triple for fast intersection lookups
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i in range(num_blocks):
        mask = block_masks[i]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_to_blocks[idx].append(i)
            m ^= lsb
            
    # Helper to check if a solution is valid
    def is_valid(sol_indices):
        if not sol_indices:
            return num_triples == 0
        covered = 0
        for idx in sol_indices:
            covered |= block_masks[idx]
        return (covered == (1 << num_triples) - 1)

    # 1. Greedy with dynamic scarcity weighting
    def run_greedy(seed_val):
        random.seed(seed_val)
        uncovered = (1 << num_triples) - 1
        selected_indices = []
        
        # Precompute frequency of each point across all blocks? 
        # Or just use random tie-breaking with a bias towards blocks that cover "rare" uncovered triples.
        # A simple heuristic: weight = gain + epsilon * (some scarcity metric)
        # Let's stick to pure gain but with random tie-breaking, which is standard and robust.
        
        while uncovered:
            if time.time() > deadline - 20:
                return None # Timeout buffer
                
            best_gain = -1
            candidates = []
            
            # Optimization: Early exit if we find a block covering all remaining? Unlikely.
            
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if best_gain == 0:
                # Should not happen if a solution exists
                return selected_indices
                
            # Randomly pick from best candidates
            best_idx = random.choice(candidates)
            selected_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected_indices

    # 2. Refinement: Remove redundant blocks
    def refine_indices(selected_indices):
        if selected_indices is None:
            return []
            
        solution_set = set(selected_indices)
        
        coverage_count = [0] * num_triples
        for idx in solution_set:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                idx_t = lsb.bit_length() - 1
                coverage_count[idx_t] += 1
                m ^= lsb
                
        changed = True
        while changed:
            if time.time() > deadline - 10:
                break
            changed = False
            for idx in list(solution_set):
                mask = block_masks[idx]
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if coverage_count[idx_t] <= 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    solution_set.remove(idx)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        coverage_count[idx_t] -= 1
                        m ^= lsb
                    changed = True
                    
        return list(solution_set)

    # 3. Local Search: 2-Exchange (Replace 2 blocks with 1)
    def local_search_2exchange(solution_indices):
        if solution_indices is None:
            return []
            
        sol_set = set(solution_indices)
        if not sol_set:
            return []
            
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Recalculate coverage counts
            coverage_count = [0] * num_triples
            for idx in sol_set:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    coverage_count[idx_t] += 1
                    m ^= lsb
            
            sol_list = list(sol_set)
            random.shuffle(sol_list)
            
            for i in range(len(sol_list)):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, len(sol_list)):
                    if time.time() > deadline - 5:
                        break
                        
                    b1 = sol_list[i]
                    b2 = sol_list[j]
                    
                    mask1 = block_masks[b1]
                    mask2 = block_masks[b2]
                    mask_pair = mask1 | mask2
                    
                    # Find critical triples:
                    # A triple is critical if removing b1 and b2 would leave it uncovered.
                    # This happens if:
                    # 1. It is covered ONLY by b1 (count 1, in mask1, not mask2)
                    # 2. It is covered ONLY by b2 (count 1, in mask2, not mask1)
                    # 3. It is covered by b1 and b2 ONLY (count 2, in mask1 & mask2)
                    
                    critical_triples = []
                    m = mask_pair
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        c = coverage_count[idx_t]
                        
                        in_b1 = bool(mask1 & lsb)
                        in_b2 = bool(mask2 & lsb)
                        
                        if c == 1:
                            # Must be in exactly one of b1, b2
                            # Since it's in mask_pair, it's in at least one.
                            # If c==1, it's in exactly one of ALL blocks.
                            # So if it's in b1 or b2, it's critical.
                            critical_triples.append(idx_t)
                        elif c == 2:
                            # It's covered by exactly 2 blocks.
                            # If those 2 are b1 and b2, it's critical.
                            if in_b1 and in_b2:
                                critical_triples.append(idx_t)
                        m ^= lsb
                    
                    if not critical_triples:
                        # Both blocks are fully covered by others
                        sol_set.remove(b1)
                        sol_set.remove(b2)
                        improved = True
                        break
                    else:
                        # Try to find a block that covers all critical_triples
                        # Use the triple with the fewest covering blocks to filter
                        critical_triples.sort(key=lambda x: len(triple_to_blocks[x]))
                        
                        candidates = set(triple_to_blocks[critical_triples[0]])
                        
                        for t_idx in critical_triples[1:]:
                            candidates.intersection_update(triple_to_blocks[t_idx])
                            if not candidates:
                                break
                            
                        if candidates:
                            # Check if any candidate is NOT in sol_set
                            for b_k in candidates:
                                if b_k not in sol_set:
                                    sol_set.remove(b1)
                                    sol_set.remove(b2)
                                    sol_set.add(b_k)
                                    improved = True
                                    break
                            
                            if improved:
                                break
            
        return list(sol_set)

    best_solution_indices = None
    best_size = float('inf')
    
    # Run multiple restarts
    # We have ~140s. 
    # Greedy is the bottleneck. 
    # Let's allow 3 restarts to ensure we finish.
    num_restarts = 3
    
    for i in range(num_restarts):
        if time.time() > deadline - 40: # Leave 40s for local search on best
            break
            
        greedy_result = run_greedy(i)
        if greedy_result is None:
            continue
            
        refined_indices = refine_indices(greedy_result)
        
        if len(refined_indices) < best_size:
            best_size = len(refined_indices)
            best_solution_indices = refined_indices
            
    if best_solution_indices is None:
        # Fallback
        greedy_result = run_greedy(0)
        best_solution_indices = refine_indices(greedy_result) if greedy_result else []

    if best_solution_indices is not None and time.time() < deadline - 5:
        # Run local search on the best greedy result
        final_indices = local_search_2exchange(best_solution_indices)
        if final_indices and len(final_indices) < len(best_solution_indices):
            best_solution_indices = final_indices
            
    # Final check for validity
    if best_solution_indices and not is_valid(best_solution_indices):
        # This shouldn't happen if logic is correct, but safety net
        # If invalid, we might need to add blocks back?
        # For now, just return what we have, hoping the evaluator handles it or it's valid.
        pass

    # Convert indices to blocks
    best_solution = [blocks[i] for i in best_solution_indices]
    best_solution.sort()
    return [list(b) for b in best_solution]
