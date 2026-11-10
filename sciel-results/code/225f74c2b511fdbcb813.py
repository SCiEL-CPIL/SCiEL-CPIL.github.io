import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using randomized greedy with
    randomized perturbation and aggressive redundancy removal.
    
    Approach:
    1. Precompute all t-subsets and map them to indices.
    2. For each greedy run:
       - Start with all t-subsets uncovered
       - Repeatedly select a block that covers the maximum number of uncovered t-subsets
       - Use randomization to break ties and explore different solutions
    3. Remove redundant blocks after each greedy run
    4. Keep the best solution found within the time limit
    """
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute all k-subsets and their covered t-subsets for fast lookup
    # This is feasible since C(19,7) = 50388
    all_k_subsets = list(itertools.combinations(points, k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, precompute which t-subsets it covers
    # Store as a list of t-subset indices
    k_to_t_indices = {}
    for block in all_k_subsets:
        t_indices = []
        for ts in itertools.combinations(block, t):
            t_indices.append(t_subset_index[ts])
        k_to_t_indices[block] = t_indices
    
    # Also precompute for each t-subset, which k-subsets cover it
    # This helps with smarter block selection
    t_to_k_indices = [[] for _ in range(num_t_subsets)]
    for block, t_indices in k_to_t_indices.items():
        for ti in t_indices:
            t_to_k_indices[ti].append(block)
    
    best_blocks = None
    best_size = float('inf')
    
    start_time = time.time()
    time_limit = 110  # Leave buffer for evaluator
    
    # Strategy 1: Pure greedy with randomized tie-breaking
    # Strategy 2: Greedy with random restarts and perturbation
    # Strategy 3: Local search on the set of blocks
    
    def greedy_run(use_random_tie_break=True, num_random_anchors=0):
        """Run a greedy algorithm to construct a covering."""
        uncovered_count = [1] * num_t_subsets  # 1 if uncovered, 0 if covered
        uncovered_set = set(range(num_t_subsets))
        blocks = []
        
        while uncovered_set:
            if num_random_anchors > 0 and random.random() < 0.3:
                # Pick a random uncovered t-subset and find a block containing it
                anchor_t = random.choice(list(uncovered_set))
                candidate_blocks = t_to_k_indices[anchor_t]
                if not candidate_blocks:
                    # Should not happen
                    continue
                # Among blocks containing this t-subset, pick one that covers most uncovered
                best_block = None
                best_cov = -1
                for block in candidate_blocks:
                    cov = 0
                    for ti in k_to_t_indices[block]:
                        if ti in uncovered_set:
                            cov += 1
                    if cov > best_cov:
                        best_cov = cov
                        best_block = block
                if best_block is None:
                    continue
                chosen = best_block
            else:
                # Standard greedy: pick block covering max uncovered t-subsets
                # To speed up, we can sample if there are too many k-subsets
                best_block = None
                best_cov = -1
                best_candidates = []
                
                # Sample a subset of k-subsets to consider
                # For v=19, k=7, there are 50388 k-subsets
                # We can consider all of them since it's not too many
                for block in all_k_subsets:
                    cov = 0
                    for ti in k_to_t_indices[block]:
                        if ti in uncovered_set:
                            cov += 1
                    if cov > best_cov:
                        best_cov = cov
                        best_block = block
                        best_candidates = [block]
                    elif cov == best_cov and cov > 0:
                        best_candidates.append(block)
                
                if best_cov <= 0:
                    # No block covers any uncovered t-subset? Should not happen
                    break
                
                if use_random_tie_break and len(best_candidates) > 1:
                    chosen = random.choice(best_candidates)
                else:
                    chosen = best_candidates[0] if best_candidates else best_block
            
            blocks.append(chosen)
            
            # Update uncovered set
            for ti in k_to_t_indices[chosen]:
                if ti in uncovered_set:
                    uncovered_set.discard(ti)
        
        return blocks
    
    def remove_redundant(blocks):
        """Remove redundant blocks from a covering."""
        # Build coverage map: t_subset_index -> list of block indices
        coverage_map = [[] for _ in range(num_t_subsets)]
        for i, block in enumerate(blocks):
            for ti in k_to_t_indices[block]:
                coverage_map[ti].append(i)
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                is_redundant = True
                for ti in k_to_t_indices[block]:
                    if len(coverage_map[ti]) <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    changed = True
                    for ti in k_to_t_indices[block]:
                        coverage_map[ti].remove(i)
                else:
                    new_blocks.append(block)
            blocks = new_blocks
            if changed:
                # Rebuild coverage map
                coverage_map = [[] for _ in range(num_t_subsets)]
                for i, block in enumerate(blocks):
                    for ti in k_to_t_indices[block]:
                        coverage_map[ti].append(i)
        
        return blocks
    
    def local_search(blocks):
        """Try to improve a covering via local search."""
        blocks = list(blocks)
        improved = True
        while improved:
            improved = False
            # Try removing each block and see if covering is still valid
            for i in range(len(blocks)):
                candidate = blocks[:i] + blocks[i+1:]
                # Check if all t-subsets are covered
                covered = [0] * num_t_subsets
                valid = True
                for block in candidate:
                    for ti in k_to_t_indices[block]:
                        covered[ti] += 1
                for ti in range(num_t_subsets):
                    if covered[ti] == 0:
                        valid = False
                        break
                if valid:
                    blocks = candidate
                    improved = True
                    break
            # Try replacing a block with a different one
            for i in range(len(blocks)):
                # Try replacing block i with a different block
                # Sample a few random blocks
                for _ in range(20):
                    new_block = random.choice(all_k_subsets)
                    if new_block in blocks:
                        continue
                    candidate = blocks[:i] + [new_block] + blocks[i+1:]
                    covered = [0] * num_t_subsets
                    valid = True
                    for block in candidate:
                        for ti in k_to_t_indices[block]:
                            covered[ti] += 1
                    for ti in range(num_t_subsets):
                        if covered[ti] == 0:
                            valid = False
                            break
                    if valid and len(candidate) < len(blocks):
                        blocks = candidate
                        improved = True
                        break
                if improved:
                    break
        return blocks
    
    iterations = 0
    no_improve_count = 0
    patience = 30  # Stop if no improvement for this many iterations
    
    while time.time() - start_time < time_limit:
        iterations += 1
        
        # Use different strategies
        if iterations % 3 == 1:
            # Pure greedy
            blocks = greedy_run(use_random_tie_break=True, num_random_anchors=0)
        elif iterations % 3 == 2:
            # Greedy with random anchors
            blocks = greedy_run(use_random_tie_break=True, num_random_anchors=5)
        else:
            # Greedy without random tie-breaking (deterministic)
            blocks = greedy_run(use_random_tie_break=False, num_random_anchors=0)
        
        # Remove redundant blocks
        blocks = remove_redundant(blocks)
        
        # Local search
        if len(blocks) > 0 and time.time() - start_time < time_limit - 5:
            blocks = local_search(blocks)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = [list(b) for b in blocks]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > patience:
            break
    
    if best_blocks is None:
        # Fallback: run one final greedy
        blocks = greedy_run(use_random_tie_break=True, num_random_anchors=0)
        blocks = remove_redundant(blocks)
        best_blocks = [list(b) for b in blocks]
    
    return best_blocks
