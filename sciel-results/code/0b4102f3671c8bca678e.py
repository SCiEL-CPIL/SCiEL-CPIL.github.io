import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (evaluator kills at 150s, but let's be safe)

    # 1. Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage for every possible k-block
    # block_cover[i] is a list of indices of t-subsets covered by block i
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    block_cover = []
    
    for block in all_blocks:
        covered_indices = []
        for t_sub in itertools.combinations(block, t):
            covered_indices.append(t_subset_to_idx[t_sub])
        block_cover.append(covered_indices)
        
    def greedy_cover(rng, deadline):
        """
        Greedy set cover with random tie-breaking.
        Returns list of block indices.
        """
        # uncovered is a set of indices of t-subsets not yet covered
        uncovered = set(range(num_t_subsets))
        chosen_indices = []
        used = [False] * num_blocks
        
        # To speed up gain calculation, we can't easily vectorize without numpy bitsets.
        # However, for v=19, num_blocks=50388, num_t_subsets=3876.
        # Iterating 50388 blocks * 35 items each = ~1.7M operations per step.
        # If we need ~150 steps, that's 250M operations. Python might be too slow for pure loops.
        # We need a faster approach.
        
        # Alternative: Use a list of counts? No, we need to know WHICH are uncovered.
        # Let's try to optimize the inner loop.
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            best_gain = 0
            best_candidates = []
            
            # Optimization: Instead of checking all blocks, we can just check all.
            # 50k blocks is manageable if the inner loop is tight.
            for i in range(num_blocks):
                if used[i]:
                    continue
                # Count how many of block_cover[i] are in uncovered
                # Using a set for uncovered makes lookup O(1)
                gain = sum(1 for idx in block_cover[i] if idx in uncovered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                # Should not happen if problem is solvable, but break to avoid infinite loop
                break
                
            # Pick randomly from best candidates
            chosen_idx = rng.choice(best_candidates)
            chosen_indices.append(chosen_idx)
            used[chosen_idx] = True
            
            # Remove covered items from uncovered
            for idx in block_cover[chosen_idx]:
                if idx in uncovered:
                    uncovered.remove(idx)
                    
        return chosen_indices

    def reduce_cover(chosen_indices, deadline):
        """
        Remove redundant blocks.
        A block is redundant if all its t-subsets are covered by other chosen blocks.
        """
        if not chosen_indices:
            return []
            
        # Count coverage of each t-subset by the current set of blocks
        # cover_count[j] = number of blocks in chosen_indices covering t-subset j
        cover_count = [0] * num_t_subsets
        
        for idx in chosen_indices:
            for t_idx in block_cover[idx]:
                cover_count[t_idx] += 1
                
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i in range(len(chosen_indices)):
                if time.time() > deadline:
                    break
                    
                idx = chosen_indices[i]
                # Check if any t-subset in this block has count 1
                is_redundant = True
                for t_idx in block_cover[idx]:
                    if cover_count[t_idx] == 1:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    # Remove block i
                    for t_idx in block_cover[idx]:
                        cover_count[t_idx] -= 1
                    chosen_indices.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                    
        return chosen_indices

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    # Estimate: One greedy run takes ~150 steps * 50k blocks * 35 checks = 262M ops.
    # In Python, this is likely ~10-20 seconds.
    # So we can afford maybe 5-10 restarts.
    
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        seed = restart * 12345 + 67890
        rng = random.Random(seed)
        
        # Greedy phase
        chosen_indices = greedy_cover(rng, deadline - 5.0)
        if chosen_indices is None:
            continue
            
        # Reduction phase
        reduced = reduce_cover(chosen_indices, deadline - 1.0)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_solution = reduced
            
    if best_solution is None:
        # Fallback: Return a valid cover even if large, or empty if failed completely
        # The problem guarantees a cover exists. If greedy failed, it's likely a bug or timeout.
        # Let's try one more deterministic greedy if time permits
        if time.time() < deadline - 1.0:
            rng = random.Random(0)
            chosen_indices = greedy_cover(rng, deadline - 0.5)
            if chosen_indices:
                best_solution = chosen_indices
                
    if best_solution is None:
        return []
        
    return [list(all_blocks[idx]) for idx in best_solution]
