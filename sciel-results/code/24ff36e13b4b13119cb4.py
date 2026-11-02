import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Covers design C(v, k, t) using Iterated Local Search (ILS) with a variable-neighborhood 2-exchange.
    
    Strategy:
    1. Construct an initial valid covering using a deterministic greedy algorithm (max coverage).
    2. Use Local Search to minimize block count via deletions and swaps.
    3. Use Iterated Local Search with perturbations (random block replacements) to escape local minima.
    4. Focus on 2-exchange moves (remove 2 blocks, add 1-2 blocks) to find better overlaps.
    
    Key differences from current program:
    - No random pool sampling; works directly with the full space of possible blocks.
    - Uses a more systematic local search with 2-exchange moves.
    - Perturbation strategy is designed to escape local minima in the block-interaction graph.
    """
    deadline = time.time() + 145.0  # Safety margin below 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # For each t-subset, precompute the list of all k-blocks that contain it
    # This is crucial for efficient local search
    ts_to_blocks = {}
    for ts in all_t_subsets:
        ts_set = set(ts)
        remaining = [i for i in range(v) if i not in ts_set]
        blocks = []
        for combo in itertools.combinations(remaining, k - t):
            block = tuple(sorted(list(ts) + list(combo)))
            blocks.append(block)
        ts_to_blocks[ts] = blocks
    
    # For each block, precompute which t-subsets it covers
    # We'll compute this on the fly for blocks we use, but cache it
    block_cover_cache = {}
    
    def get_block_cover(block):
        if block not in block_cover_cache:
            covered = frozenset(t_subset_to_idx[ts] for ts in itertools.combinations(block, t))
            block_cover_cache[block] = covered
        return block_cover_cache
    
    def get_initial_covering():
        """
        Build a valid covering using a deterministic greedy approach that maximizes
        coverage of uncovered t-subsets.
        """
        uncovered = set(range(n_t))
        selected = []
        
        while uncovered:
            if time.time() > deadline - 10.0:
                break
            
            # Find the block that covers the most uncovered t-subsets
            best_block = None
            best_count = 0
            
            # Sample a subset of possible blocks to speed up
            # Use a random sample of size 5000
            all_blocks = list(itertools.combinations(range(v), k))
            sample_size = min(5000, len(all_blocks))
            rng = random.Random()
            sample = rng.sample(all_blocks, sample_size)
            
            for block in sample:
                cov = get_block_cover(block)
                count = len(cov & uncovered)
                if count > best_count:
                    best_count = count
                    best_block = block
            
            if best_block is None:
                break
            
            selected.append(best_block)
            uncovered -= get_block_cover(best_block)
        
        return selected
    
    def local_search(selected_blocks, rng):
        """
        Perform local search to minimize the number of blocks.
        Uses deletion and 2-exchange moves.
        """
        selected_set = list(selected_blocks)
        current_count = len(selected_set)
        
        # Build coverage count array
        cover_count = np.zeros(n_t, dtype=np.int32)
        for block in selected_set:
            for idx in get_block_cover(block):
                cover_count[idx] += 1
        
        # Verify all are covered
        if np.any(cover_count == 0):
            return None, float('inf')
        
        best_count = current_count
        best_set = list(selected_set)
        
        # Local search loop
        no_improve = 0
        max_no_improve = 50
        steps = 0
        max_steps = 2000
        
        while no_improve < max_no_improve and steps < max_steps:
            if time.time() > deadline - 5.0:
                break
            
            steps += 1
            improved = False
            
            # Try deletion moves
            for i in range(len(selected_set)):
                block = selected_set[i]
                cov = get_block_cover(block)
                
                # Check if removal is valid
                can_remove = True
                for idx in cov:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                
                if not can_remove:
                    # Apply deletion
                    for idx in cov:
                        cover_count[idx] -= 1
                    selected_set.pop(i)
                    current_count -= 1
                    improved = True
                    no_improve = 0
                    break
                else:
                    no_improve += 1
            
            if improved:
                if current_count < best_count:
                    best_count = current_count
                    best_set = list(selected_set)
                continue
            
            # Try 2-exchange moves: remove one block, add another
            if selected_set:
                i = rng.randrange(len(selected_set))
                block_to_remove = selected_set[i]
                cov_remove = get_block_cover(block_to_remove)
                
                # Get the t-subsets that would become uncovered if we remove this block
                critical = set()
                for idx in cov_remove:
                    if cover_count[idx] == 1:
                        critical.add(idx)
                
                if not critical:
                    # This is effectively a deletion, already handled
                    no_improve += 1
                    continue
                
                # Find a block that covers all critical t-subsets
                # Sample possible blocks
                all_blocks = list(itertools.combinations(range(v), k))
                sample_size = min(2000, len(all_blocks))
                sample = rng.sample(all_blocks, sample_size)
                
                found_j = None
                for block in sample:
                    if block in selected_set:
                        continue
                    cov_j = get_block_cover(block)
                    if critical.issubset(cov_j):
                        found_j = block
                        break
                
                if found_j is not None:
                    # Apply swap
                    for idx in cov_remove:
                        cover_count[idx] -= 1
                    for idx in get_block_cover(found_j):
                        cover_count[idx] += 1
                    selected_set[i] = found_j
                    # current_count unchanged
                    no_improve = 0
                    # Don't update best_count since count is unchanged
                else:
                    no_improve += 1
        
        return best_set, best_count
    
    best_final_blocks = None
    best_final_count = float('inf')
    
    # Get initial covering
    initial_blocks = get_initial_covering()
    
    if not initial_blocks:
        # Fallback: simple greedy
        rng = random.Random(42)
        all_blocks = list(itertools.combinations(range(v), k))
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 2.0:
                break
            sample_size = min(1000, len(all_blocks))
            sample = rng.sample(all_blocks, sample_size)
            best_block = None
            best_count = 0
            for block in sample:
                cov = get_block_cover(block)
                count = len(cov & uncovered)
                if count > best_count:
                    best_count = count
                    best_block = block
            if best_block is None:
                break
            blocks.append(best_block)
            uncovered -= get_block_cover(best_block)
        
        best_final_blocks = blocks
        best_final_count = len(blocks)
    else:
        best_final_blocks = initial_blocks
        best_final_count = len(initial_blocks)
    
    # Iterated Local Search
    num_iterations = 20
    for iteration in range(num_iterations):
        if time.time() > deadline - 10.0:
            break
        
        rng = random.Random(42 + iteration)
        
        # Perturbation: randomly replace some blocks
        perturbed = list(best_final_blocks)
        num_perturb = max(1, len(perturbed) // 10)
        for _ in range(num_perturb):
            if not perturbed:
                break
            i = rng.randrange(len(perturbed))
            all_blocks = list(itertools.combinations(range(v), k))
            sample_size = min(1000, len(all_blocks))
            sample = rng.sample(all_blocks, sample_size)
            perturbed[i] = rng.choice(sample)
        
        # Local search on perturbed solution
        ls_result, ls_count = local_search(perturbed, rng)
        
        if ls_result is not None and ls_count < best_final_count:
            best_final_count = ls_count
            best_final_blocks = ls_result
    
    # Final verification and cleanup
    if not best_final_blocks:
        return []
    
    # Verify coverage
    covered = set()
    for b in best_final_blocks:
        for ts in itertools.combinations(b, t):
            covered.add(t_subset_to_idx[ts])
    
    if covered != set(range(n_t)):
        # Try to fix by adding missing blocks
        missing = set(range(n_t)) - covered
        for ts_idx in missing:
            ts = all_t_subsets[ts_idx]
            for block in ts_to_blocks[ts]:
                if block not in [tuple(b) for b in best_final_blocks]:
                    best_final_blocks.append(list(block))
                    break
    
    # Remove duplicate blocks
    unique_blocks = []
    seen = set()
    for b in best_final_blocks:
        bt = tuple(b)
        if bt not in seen:
            seen.add(bt)
            unique_blocks.append(b)
    
    return unique_blocks

if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(f"Blocks: {len(blocks)}")
