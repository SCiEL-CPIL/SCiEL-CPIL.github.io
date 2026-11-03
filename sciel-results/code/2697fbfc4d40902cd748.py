import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Strategy:
    1. Precompute all blocks and their covered t-subsets.
    2. Use scarcity-weighted greedy: prioritize uncovered t-subsets with the fewest
       candidate blocks (hardest to cover), breaking ties randomly.
    3. Apply pairwise block-exchange local search:
       - Try removing block A and adding block B (1-for-1) if it doesn't break coverage
         and reduces total block count (or keeps it same but improves "slack").
       - Try removing block A and adding blocks B,C (1-for-2) if it reduces count.
    4. Multiple random restarts with different seeds.
    """
    deadline = time.time() + 148.0
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the list of t-subset indices it covers
    block_covered = []
    for block in all_blocks:
        covered = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered.append(idx)
        covered.sort()
        block_covered.append(covered)
    
    # Reverse index: for each t-subset, which blocks cover it
    t_subset_to_blocks = [[] for _ in range(n_t)]
    for b_idx, covered in enumerate(block_covered):
        for idx in covered:
            t_subset_to_blocks[idx].append(b_idx)
    
    # Point scarcity: for each point, how many blocks contain it (used for weighting)
    # Actually, let's use triple scarcity: number of blocks that cover each t-subset
    triple_block_count = np.array([len(t_subset_to_blocks[i]) for i in range(n_t)], dtype=np.int32)
    
    # Helper: check if a set of block indices covers all t-subsets
    def is_cover(block_indices):
        if not block_indices:
            return n_t == 0
        covered = np.zeros(n_t, dtype=bool)
        for idx in block_indices:
            for j in block_covered[idx]:
                covered[j] = True
        return covered.all()
    
    # Helper: get coverage counts
    def get_coverage_counts(block_indices):
        counts = np.zeros(n_t, dtype=np.int32)
        for idx in block_indices:
            for j in block_covered[idx]:
                counts[j] += 1
        return counts
    
    # Scarcity-weighted greedy: pick the uncovered t-subset with the fewest candidate blocks,
    # then among blocks covering it, pick the one with max marginal gain
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 35.0:
                break
            
            # Pick the uncovered t-subset with minimum block count (hardest to cover)
            # Sample up to 50 random uncovered to find the scarcest
            sample_size = min(50, len(uncovered))
            sample = rng.sample(uncovered, sample_size)
            best_triple = min(sample, key=lambda x: triple_block_count[x])
            
            candidate_blocks = t_subset_to_blocks[best_triple]
            
            # Choose the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            for b_idx in candidate_blocks:
                gain = 0
                for j in block_covered[b_idx]:
                    if j in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            for j in block_covered[best_block]:
                uncovered.discard(j)
        
        return blocks
    
    # Local search: pairwise block exchange
    def local_search(block_indices, seed=0, max_iters=200):
        rng = random.Random(seed)
        current = list(block_indices)
        if not current:
            return current
        
        # Precompute coverage counts
        counts = get_coverage_counts(current)
        
        improved = True
        iteration = 0
        while improved and iteration < max_iters:
            if time.time() > deadline - 30.0:
                break
            
            improved = False
            iteration += 1
            
            # Shuffle block order for randomization
            block_order = list(range(len(current)))
            rng.shuffle(block_order)
            
            for i in block_order:
                if time.time() > deadline - 30.0:
                    break
                
                b_idx = current[i]
                # Check if this block is redundant (all its t-subsets have count > 1)
                can_remove = True
                for j in block_covered[b_idx]:
                    if counts[j] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Try 1-for-0: just remove it
                    current.pop(i)
                    for j in block_covered[b_idx]:
                        counts[j] -= 1
                    improved = True
                    i -= 1  # Adjust index after removal
                    continue
                
                # Try 1-for-1: remove b_idx, add another block
                # Find blocks that cover at least one t-subset uniquely covered by b_idx
                unique_to_b = [j for j in block_covered[b_idx] if counts[j] == 1]
                
                if not unique_to_b:
                    continue
                
                # For each unique t-subset, find alternative blocks
                candidate_replacements = set()
                for j in unique_to_b:
                    for alt_b in t_subset_to_blocks[j]:
                        if alt_b != b_idx and alt_b not in current:
                            candidate_replacements.add(alt_b)
                
                if not candidate_replacements:
                    continue
                
                # Try a few random replacements
                candidates = list(candidate_replacements)
                rng.shuffle(candidates)
                tried = 0
                for alt_b in candidates:
                    if tried >= 10:
                        break
                    tried += 1
                    
                    # Check if replacing b_idx with alt_b maintains coverage
                    # alt_b must cover all unique_to_b t-subsets that b_idx covers
                    covers_all_unique = all(
                        any(alt_b in t_subset_to_blocks[j] for j in [j]) 
                        for j in unique_to_b
                    )
                    # More efficient: check if alt_b covers each unique t-subset
                    alt_covered = set(block_covered[alt_b])
                    if all(j in alt_covered for j in unique_to_b):
                        # Perform the swap
                        current[i] = alt_b
                        # Update counts
                        for j in block_covered[b_idx]:
                            counts[j] -= 1
                        for j in block_covered[alt_b]:
                            counts[j] += 1
                        improved = True
                        break
        
        return current
    
    # Get initial solutions with multiple seeds
    initial_solutions = []
    for seed in range(30):
        if time.time() > deadline - 50.0:
            break
        sol = greedy_cover(seed)
        if is_cover(sol):
            # Apply local search
            reduced = local_search(sol, seed=seed, max_iters=100)
            if is_cover(reduced):
                initial_solutions.append(reduced)
            else:
                initial_solutions.append(sol)
    
    if not initial_solutions:
        sol = greedy_cover(0)
        initial_solutions = [sol]
    
    # Start with the best initial solution
    best_indices = min(initial_solutions, key=len)
    best_len = len(best_indices)
    
    # Apply more local search iterations to the best solution
    for seed in range(20):
        if time.time() > deadline - 20.0:
            break
        
        refined = local_search(best_indices, seed=seed + 1000, max_iters=200)
        if is_cover(refined):
            if len(refined) < best_len:
                best_len = len(refined)
                best_indices = refined
    
    # Try local search on other initial solutions too
    for sol in initial_solutions:
        if time.time() > deadline - 15.0:
            break
        if len(sol) >= best_len:
            continue
        
        refined = local_search(sol, seed=random.randint(0, 10000), max_iters=150)
        if is_cover(refined):
            if len(refined) < best_len:
                best_len = len(refined)
                best_indices = refined
    
    # Final validation
    if not is_cover(best_indices):
        # Fallback
        for sol in initial_solutions:
            if is_cover(sol):
                best_indices = sol
                break
        else:
            best_indices = greedy_cover(0)
    
    # Convert to block tuples and ensure uniqueness
    result = [all_blocks[i] for i in best_indices]
    result = list(set(result))
    
    # Final validation
    final_covered = np.zeros(n_t, dtype=bool)
    for b in result:
        for ts in itertools.combinations(b, t):
            final_covered[t_subset_to_idx[ts]] = True
    
    if not final_covered.all():
        # Fix by adding missing blocks
        uncovered = np.where(~final_covered)[0]
        for ts_idx in uncovered:
            candidates = t_subset_to_blocks[ts_idx]
            for b_idx in candidates:
                b = all_blocks[b_idx]
                if b not in result:
                    result.append(b)
                    for ts in itertools.combinations(b, t):
                        final_covered[t_subset_to_idx[ts]] = True
                    break
    
    return [list(b) for b in result]
