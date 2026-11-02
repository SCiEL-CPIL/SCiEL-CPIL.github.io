import itertools
import random
import time

# Precomputation for v=19, k=8, t=5
# We will hardcode the precomputation for speed, or do it dynamically if v,k,t change.
# The task says "evaluator passes this task's parameters v, k and t".
# So we must handle general v, k, t, but optimize for the specific case if possible?
# The prompt says "Keep the function names...". It implies the function signature is fixed.
# The parameters v, k, t are passed in.
# We should probably precompute inside the function if it's not the specific case, 
# or just assume it's the specific case? 
# "The evaluator passes this task's parameters v, k and t."
# It's safer to write a general solver but optimize the inner loops.
# Given v=19 is small, general code with bitmasks is fine.

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # 1. Generate all t-subsets and map them to indices
    # For v=19, t=5, this is small.
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # 2. Generate all k-blocks and their covered t-subset indices
    # For v=19, k=8, num_blocks = 75582.
    # Each block covers C(8,5)=56 t-subsets.
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage: list of lists of t-subset indices
    # block_coverage[i] is a list of indices into t_subsets
    block_coverage = []
    for block in all_blocks:
        covered_indices = []
        for ts in itertools.combinations(block, t):
            covered_indices.append(t_subset_to_idx[ts])
        block_coverage.append(covered_indices)
        
    # 3. Greedy Initialization
    # counts[i] = number of blocks in current solution that cover t_subset i
    counts = [0] * num_t_subsets
    current_blocks = [] # List of block indices
    
    # To speed up greedy, we can just iterate all blocks each time.
    # With 75k blocks and ~300 steps, that's 22 million operations. Python might be slow.
    # Optimization: Only consider blocks that cover at least one uncovered subset.
    # We can maintain a list of uncovered subsets.
    
    uncovered = set(range(num_t_subsets))
    
    while uncovered:
        best_block_idx = -1
        best_gain = -1
        
        # Iterate all blocks
        for b_idx in range(num_blocks):
            # Check if this block covers any uncovered subset
            gain = 0
            for idx in block_coverage[b_idx]:
                if idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            break # Should not happen if a covering exists
            
        current_blocks.append(best_block_idx)
        # Update counts and uncovered
        for idx in block_coverage[best_block_idx]:
            counts[idx] += 1
            if counts[idx] == 1:
                uncovered.discard(idx)
                
    # 4. Local Search: Reduction and Swap
    # We want to minimize len(current_blocks)
    
    def try_reduce(blocks_list):
        """Try to remove redundant blocks. Returns new list of block indices."""
        # We need to recalculate counts for the current set to be sure?
        # Or we can maintain counts incrementally.
        # Let's maintain counts globally and update on add/remove.
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks_list)):
                b_idx = blocks_list[i]
                # Check if removing b_idx leaves any subset uncovered
                # Temporarily remove
                is_redundant = True
                for idx in block_coverage[b_idx]:
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for idx in block_coverage[b_idx]:
                        counts[idx] -= 1
                    blocks_list.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        return blocks_list

    # Run initial reduction
    current_blocks = try_reduce(current_blocks)
    
    # Time check
    if time.time() - start_time > time_limit * 0.5:
        return [list(all_blocks[i]) for i in current_blocks]
        
    # Local Search Loop
    best_size = len(current_blocks)
    
    while time.time() - start_time < time_limit:
        # If stuck (no reduction possible), try a swap
        # Pick a random block to replace
        if len(current_blocks) > 0:
            i = random.randint(0, len(current_blocks) - 1)
            old_b_idx = current_blocks[i]
            
            # Find a replacement block that covers the unique subsets of old_b_idx
            # Unique subsets of old_b_idx: those with count == 1
            unique_indices = []
            for idx in block_coverage[old_b_idx]:
                if counts[idx] == 1:
                    unique_indices.append(idx)
            
            # If no unique indices, it should have been removed by try_reduce.
            # So unique_indices is not empty.
            
            # We want a block that covers ALL unique_indices.
            # And ideally covers some others too.
            # Iterate through candidate blocks?
            # To be fast, we can sample a few random blocks that contain the intersection of unique subsets?
            # Or just pick a random block from all_blocks and check if it covers unique_indices.
            
            # Better: Generate a random block that contains the points of the "hardest" subset?
            # Let's just try a few random swaps.
            
            # Find a candidate
            candidate_idx = -1
            # Try a few random blocks
            for _ in range(20):
                cand = random.randint(0, num_blocks - 1)
                # Check if cand covers all unique_indices
                # This check is expensive if we do it for every cand.
                # Precompute: for each t-subset idx, which blocks cover it?
                # This inverse index might be useful.
                
            # Let's build the inverse index: t_subset_to_blocks
            # t_subset_to_blocks[idx] = list of block indices covering idx
            # This takes memory but speeds up swap search.
            
            if not hasattr(construct, 't_to_blocks'):
                construct.t_to_blocks = [[] for _ in range(num_t_subsets)]
                for b_idx, cov in enumerate(block_coverage):
                    for idx in cov:
                        construct.t_to_blocks[idx].append(b_idx)
                
            # Get candidates that cover the first unique subset
            if unique_indices:
                first_unique = unique_indices[0]
                candidates = construct.t_to_blocks[first_unique]
                
                # Filter candidates that cover ALL unique_indices
                valid_candidates = []
                for cand in candidates:
                    # Check if cand covers all unique_indices
                    # We can do this by checking if set(unique_indices).issubset(set(block_coverage[cand]))
                    # But sets are slow. 
                    # Since unique_indices is small (usually), we can loop.
                    is_valid = True
                    # Optimization: if cand is old_b_idx, skip
                    if cand == old_b_idx:
                        continue
                    for u_idx in unique_indices:
                        if u_idx not in block_coverage[cand]: # 'in' on list is O(N). N=56. OK.
                            is_valid = False
                            break
                    if is_valid:
                        valid_candidates.append(cand)
                        
                if valid_candidates:
                    new_b_idx = random.choice(valid_candidates)
                    
                    # Perform swap
                    # Remove old
                    for idx in block_coverage[old_b_idx]:
                        counts[idx] -= 1
                    # Add new
                    for idx in block_coverage[new_b_idx]:
                        counts[idx] += 1
                    current_blocks[i] = new_b_idx
                    
                    # Try to reduce
                    current_blocks = try_reduce(current_blocks)
                    
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        # We can also keep a copy of the best solution
                        # But we are modifying in place.
                        
        # Occasionally, if stuck for a long time, do a random restart or perturbation?
        # For now, just keep swapping.
        
    # Return the blocks as lists of points
    return [list(all_blocks[i]) for i in current_blocks]
