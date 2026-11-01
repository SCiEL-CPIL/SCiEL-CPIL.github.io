import itertools
import random
import time

# Precomputation
V = 19
K = 8
T = 5

# Map 5-subsets to indices
all_5_subs = list(itertools.combinations(range(V), T))
sub_index = {s: i for i, s in enumerate(all_5_subs)}
N_SUBS = len(all_5_subs) # 11628

# Precompute coverage for all 8-blocks
# block_id -> list of sub_indices
all_8_blocks = list(itertools.combinations(range(V), K))
block_coverage = []
for b in all_8_blocks:
    cov = [sub_index[s] for s in itertools.combinations(b, T)]
    block_coverage.append(cov)

# Map from 5-subset to list of block indices containing it
# sub_idx -> list of block_idxs
sub_to_blocks = [[] for _ in range(N_SUBS)]
for b_idx, b in enumerate(all_8_blocks):
    for s_idx in block_coverage[b_idx]:
        sub_to_blocks[s_idx].append(b_idx)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Greedy Construction
    def greedy_construct():
        uncovered = [True] * N_SUBS
        blocks = []
        # Randomize order of subsets to break ties? 
        # Standard greedy: pick first uncovered, pick best block containing it.
        while True:
            # Find an uncovered subset
            try:
                # To speed up, we can just iterate, but checking all is O(N_SUBS)
                # N_SUBS is 11628, so it's fine.
                # Optimization: keep a list of uncovered indices?
                # Simpler: just find the first True
                target_idx = -1
                for i in range(N_SUBS):
                    if uncovered[i]:
                        target_idx = i
                        break
                if target_idx == -1:
                    break
            except IndexError:
                break
            
            # Get all blocks covering this subset
            candidates = sub_to_blocks[target_idx]
            
            # Pick the block that covers the most uncovered subsets
            best_block = -1
            max_gain = -1
            for b_idx in candidates:
                gain = 0
                for s_idx in block_coverage[b_idx]:
                    if uncovered[s_idx]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
            
            # Add the block
            blocks.append(best_block)
            for s_idx in block_coverage[best_block]:
                uncovered[s_idx] = False
        
        return blocks

    # Initial Solution
    current_blocks = greedy_construct()
    
    # Local Search
    def count_coverage(blocks):
        counts = [0] * N_SUBS
        for b_idx in blocks:
            for s_idx in block_coverage[b_idx]:
                counts[s_idx] += 1
        return counts

    def is_redundant(b_idx, counts):
        for s_idx in block_coverage[b_idx]:
            if counts[s_idx] <= 1:
                return False
        return True

    def local_search(blocks):
        # Convert to set for faster membership? 
        # We need to track counts dynamically.
        counts = count_coverage(blocks)
        block_set = set(blocks)
        
        improved = True
        while improved:
            improved = False
            # Pass 1: Remove redundant blocks
            # Iterate in random order to avoid bias
            random.shuffle(blocks)
            for b_idx in blocks[:]: # iterate over a copy
                if is_redundant(b_idx, counts):
                    # Remove
                    for s_idx in block_coverage[b_idx]:
                        counts[s_idx] -= 1
                    blocks.remove(b_idx)
                    block_set.remove(b_idx)
                    improved = True
            
            # Pass 2: Try to replace a block with a better one
            # For each block, find its unique subsets (count == 1)
            # If a block has no unique subsets, it's redundant (handled above)
            # So for remaining blocks, they have at least one unique subset.
            # We try to find a replacement block B' such that B' covers all unique subsets of B.
            # If such B' exists and B' != B, we can swap B with B'.
            # This might make other blocks redundant.
            
            # Optimization: Only try this if time allows
            if time.time() > deadline - 10:
                break

            for b_idx in blocks[:]:
                # Find unique subsets
                unique_subs = []
                for s_idx in block_coverage[b_idx]:
                    if counts[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    continue
                
                # Find a block B' that covers all unique_subs
                # Candidates must contain all unique_subs
                # Intersection of sub_to_blocks for each unique_sub
                # Start with the smallest list
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(sub_to_blocks[s_idx])
                    else:
                        candidates &= set(sub_to_blocks[s_idx])
                
                if not candidates:
                    continue
                
                # Exclude current block
                candidates.discard(b_idx)
                
                if not candidates:
                    continue
                
                # Pick one candidate (random or best)
                # "Best" might be one that covers more *other* subsets to help future removals?
                # For now, just pick one randomly or the first one.
                # Let's try the one that covers the most total subsets (not just unique)
                # Actually, any valid replacement is a step towards potentially removing others.
                # Let's just pick one.
                new_b_idx = next(iter(candidates))
                
                # Perform Swap
                # Remove b_idx
                for s_idx in block_coverage[b_idx]:
                    counts[s_idx] -= 1
                # Add new_b_idx
                for s_idx in block_coverage[new_b_idx]:
                    counts[s_idx] += 1
                
                blocks.remove(b_idx)
                blocks.append(new_b_idx)
                block_set.discard(b_idx)
                block_set.add(new_b_idx)
                improved = True
                break # Restart loop after swap to re-evaluate redundancies
            
            if time.time() > deadline - 5:
                break

        return blocks

    # Run local search
    current_blocks = local_search(current_blocks)
    
    # Convert indices back to blocks
    # The evaluator expects list of lists of ints
    result = [list(all_8_blocks[b_idx]) for b_idx in current_blocks]
    
    return result
