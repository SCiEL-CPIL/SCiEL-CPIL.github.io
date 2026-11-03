import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a fundamentally different approach:
    1. Precompute all blocks and their covered t-subsets as sorted lists
    2. Use a fast greedy construction with multiple random seeds
    3. Apply a structural reduction: iteratively find and remove redundant blocks
       using a different criterion (max overlap with others)
    4. Use a 'rebuild' strategy: remove a random subset of blocks, then greedily
       re-add blocks to cover what was lost, potentially finding a smaller cover
    """
    deadline = time.time() + 145.0
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
    
    # Helper: check if a set of block indices covers all t-subsets
    def is_cover(block_indices):
        if not block_indices:
            return n_t == 0
        covered_set = set()
        for idx in block_indices:
            covered_set.update(block_covered[idx])
        return len(covered_set) == n_t
    
    # Helper: count how many times each t-subset is covered
    def get_coverage_counts(block_indices):
        counts = np.zeros(n_t, dtype=np.int32)
        for idx in block_indices:
            for j in block_covered[idx]:
                counts[j] += 1
        return counts
    
    # Greedy cover with the logarithmic heuristic (max marginal gain / log(current_uncovered+1))
    # But we use a simpler variant: max marginal gain with random tie-breaking
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 30.0:
                break
            
            # Pick a random uncovered t-subset to start
            start_idx = rng.choice(list(uncovered))
            candidate_blocks = t_subset_to_blocks[start_idx]
            
            # Choose the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            best_gain_set = None
            for b_idx in candidate_blocks:
                gain = 0
                gain_set = set()
                for j in block_covered[b_idx]:
                    if j in uncovered:
                        gain += 1
                        gain_set.add(j)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_gain_set = gain_set
                elif gain == best_gain and best_block is not None:
                    # Tie-breaking: prefer blocks that cover more total t-subsets
                    # (already encoded in gain, but we can add randomness)
                    if rng.random() < 0.3:
                        best_block = b_idx
                        best_gain_set = gain_set
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            if best_gain_set:
                uncovered -= best_gain_set
        
        return blocks
    
    # Reduction: iteratively remove blocks that are redundant
    # A block is redundant if all t-subsets it covers are also covered by other blocks
    def reduce_cover(block_indices, seed=0, max_passes=50):
        rng = random.Random(seed)
        current = list(block_indices)
        
        for _ in range(max_passes):
            if time.time() > deadline - 25.0:
                break
            
            if not current:
                break
            
            counts = get_coverage_counts(current)
            
            # Find removable blocks (all their t-subsets have count > 1)
            removable = []
            for i, idx in enumerate(current):
                can_remove = True
                for j in block_covered[idx]:
                    if counts[j] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    removable.append(i)
            
            if not removable:
                break
            
            # Remove a random removable block
            i = rng.choice(removable)
            current.pop(i)
        
        return current
    
    # Rebuild strategy: remove a fraction of blocks, then greedily re-add
    def rebuild_cover(block_indices, seed=0, removal_fraction=0.15):
        rng = random.Random(seed)
        current = list(block_indices)
        
        if not current:
            return current
        
        # Remove a random fraction of blocks
        n_remove = max(1, int(len(current) * removal_fraction))
        to_remove = rng.sample(range(len(current)), min(n_remove, len(current)))
        to_remove_set = set(to_remove)
        
        remaining = [idx for i, idx in enumerate(current) if i not in to_remove_set]
        
        # Now greedily add blocks to cover what's missing
        covered_set = set()
        for idx in remaining:
            covered_set.update(block_covered[idx])
        
        uncovered = set(range(n_t)) - covered_set
        blocks = list(remaining)
        
        while uncovered:
            if time.time() > deadline - 25.0:
                break
            
            # Pick a random uncovered t-subset
            start_idx = rng.choice(list(uncovered))
            candidate_blocks = t_subset_to_blocks[start_idx]
            
            # Choose the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            best_gain_set = None
            for b_idx in candidate_blocks:
                gain = 0
                gain_set = set()
                for j in block_covered[b_idx]:
                    if j in uncovered:
                        gain += 1
                        gain_set.add(j)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_gain_set = gain_set
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            if best_gain_set:
                uncovered -= best_gain_set
        
        return blocks
    
    # Get initial solutions with multiple seeds
    initial_solutions = []
    for seed in range(20):
        if time.time() > deadline - 40.0:
            break
        sol = greedy_cover(seed)
        if is_cover(sol):
            # Reduce it
            reduced = reduce_cover(sol, seed=seed)
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
    
    # Apply rebuild and reduction to each initial solution
    for sol in initial_solutions:
        if time.time() > deadline - 20.0:
            break
        
        # Try rebuild with different fractions
        for frac in [0.1, 0.2, 0.3]:
            if time.time() > deadline - 20.0:
                break
            rebuilt = rebuild_cover(sol, seed=random.randint(0, 10000), removal_fraction=frac)
            if is_cover(rebuilt):
                reduced = reduce_cover(rebuilt, seed=random.randint(0, 10000))
                if is_cover(reduced):
                    if len(reduced) < best_len:
                        best_len = len(reduced)
                        best_indices = reduced
    
    # Additional passes: rebuild the best solution multiple times
    for seed in range(10):
        if time.time() > deadline - 15.0:
            break
        
        for frac in [0.1, 0.15, 0.2, 0.25, 0.3]:
            if time.time() > deadline - 15.0:
                break
            rebuilt = rebuild_cover(best_indices, seed=seed * 100 + int(frac * 10), removal_fraction=frac)
            if is_cover(rebuilt):
                reduced = reduce_cover(rebuilt, seed=seed * 100 + int(frac * 10) + 1)
                if is_cover(reduced):
                    if len(reduced) < best_len:
                        best_len = len(reduced)
                        best_indices = reduced
    
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
                    # Update covered
                    for ts in itertools.combinations(b, t):
                        final_covered[t_subset_to_idx[ts]] = True
                    break
    
    return [list(b) for b in result]
