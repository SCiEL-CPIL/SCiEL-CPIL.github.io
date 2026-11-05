import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their covered subset indices
all_blocks = []
block_to_subsets = []
subset_to_blocks = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    block_idx = len(all_blocks)
    all_blocks.append(block)
    covered_indices = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        covered_indices.append(idx)
        subset_to_blocks[idx].append(block_idx)
    block_to_subsets.append(covered_indices)

N_BLOCKS = len(all_blocks)

# Weights for greedy: rare subsets are more important
subset_frequencies = [len(lst) for lst in subset_to_blocks]
weights = np.array([1.0 / f for f in subset_frequencies], dtype=np.float64)

# Coverage matrix: N_BLOCKS x N_SUBSETS
# coverage_matrix[i, j] = 1 if block i covers subset j
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, indices in enumerate(block_to_subsets):
    for j in indices:
        coverage_matrix[i, j] = 1

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def run_greedy():
        """Fast vectorized weighted greedy."""
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        
        while True:
            if not uncovered.any():
                break
            
            # Calculate gain for each block: sum of weights of uncovered subsets it covers
            # effective_weights = weights * uncovered
            # gains = coverage_matrix @ effective_weights
            effective_weights = weights * uncovered
            gains = coverage_matrix.dot(effective_weights)
            
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
                
            chosen.append(best_idx)
            uncovered[coverage_matrix[best_idx] == 1] = 0
            
            # Time check
            if time.time() > deadline - 5:
                break

        return chosen

    def prune_redundant(blocks_list):
        """Remove blocks that are redundant (all their subsets are covered by other blocks)."""
        if not blocks_list:
            return blocks_list
        
        # Count coverage for each subset
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in blocks_list:
            counts[block_to_subsets[b_idx]] += 1
            
        new_blocks = []
        for b_idx in blocks_list:
            redundant = True
            for s_idx in block_to_subsets[b_idx]:
                if counts[s_idx] == 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(b_idx)
            else:
                # If we remove it, counts decrease, but we are just checking redundancy
                # Actually, if we remove it, we need to update counts for subsequent checks?
                # No, redundancy is defined relative to the *current* set.
                # If a block is redundant, removing it doesn't uncover anything.
                # So we can just skip it.
                pass
                
        return new_blocks

    def local_search(initial_blocks):
        """Local search with 1-for-1 swaps and perturbation."""
        current_blocks = list(initial_blocks)
        # Ensure no duplicates and prune
        current_blocks = list(set(current_blocks))
        current_blocks = prune_redundant(current_blocks)
        
        best_blocks = current_blocks.copy()
        best_len = len(current_blocks)
        
        # Maintain counts for fast redundancy check and swap logic
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in current_blocks:
            counts[block_to_subsets[b_idx]] += 1
        
        def update_counts_add(b_idx):
            counts[block_to_subsets[b_idx]] += 1
            
        def update_counts_remove(b_idx):
            counts[block_to_subsets[b_idx]] -= 1
            
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline - 10:
            # Phase 1: Prune
            pruned = prune_redundant(current_blocks)
            if len(pruned) < len(current_blocks):
                # Rebuild counts if pruned
                counts = np.zeros(N_SUBSETS, dtype=np.int32)
                for b_idx in pruned:
                    counts[block_to_subsets[b_idx]] += 1
                current_blocks = pruned
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve = 0
                continue

            # Phase 2: Try 1-for-1 swaps
            improved = False
            # Shuffle to avoid bias
            block_indices = list(range(len(current_blocks)))
            random.shuffle(block_indices)
            
            for i in block_indices:
                if time.time() > deadline - 10:
                    break
                    
                b_out = current_blocks[i]
                
                # Find subsets uniquely covered by b_out
                unique_subsets = []
                for s_idx in block_to_subsets[b_out]:
                    if counts[s_idx] == 1:
                        unique_subsets.append(s_idx)
                        
                if not unique_subsets:
                    # Redundant, should have been pruned, but handle just in case
                    update_counts_remove(b_out)
                    current_blocks.pop(i)
                    improved = True
                    no_improve = 0
                    break
                    
                # Find candidate blocks that cover all unique_subsets
                # Intersect candidate lists
                cands = None
                for s_idx in unique_subsets:
                    lst = subset_to_blocks[s_idx]
                    if cands is None:
                        cands = set(lst)
                    else:
                        cands.intersection_update(lst)
                    if not cands:
                        break
                        
                if cands:
                    cands.discard(b_out)
                    if cands:
                        # Pick a random candidate
                        b_in = random.choice(list(cands))
                        
                        # Check if b_in is already in current_blocks
                        if b_in in current_blocks:
                            # If it's already there, swapping doesn't change the set, 
                            # but might change "counts" logic? No, set is same.
                            # But we removed b_out and add b_in. If b_in was already there, 
                            # we are just removing b_out.
                            # Wait, if b_in is in current_blocks, and we remove b_out,
                            # the set becomes {current - b_out + b_in} = {current - b_out} (since b_in already in)
                            # This is effectively a deletion.
                            # Let's just do the swap.
                            pass

                        update_counts_remove(b_out)
                        current_blocks.pop(i)
                        update_counts_add(b_in)
                        current_blocks.append(b_in)
                        
                        improved = True
                        no_improve = 0
                        break # Restart loop with new state

            if improved:
                continue
                
            # Phase 3: Perturbation if stuck
            no_improve += 1
            if no_improve > max_no_improve:
                # Add 2-4 random blocks
                for _ in range(random.randint(2, 4)):
                    b_rand = random.randint(0, N_BLOCKS - 1)
                    if b_rand not in current_blocks:
                        current_blocks.append(b_rand)
                        update_counts_add(b_rand)
                no_improve = 0
                # Re-prune immediately
                pruned = prune_redundant(current_blocks)
                if len(pruned) < len(current_blocks):
                    counts = np.zeros(N_SUBSETS, dtype=np.int32)
                    for b_idx in pruned:
                        counts[block_to_subsets[b_idx]] += 1
                    current_blocks = pruned
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()

        return best_blocks

    best_overall = None
    best_len_overall = float('inf')

    # Run multiple restarts
    while time.time() < deadline - 20:
        # 1. Greedy
        greedy_blocks = run_greedy()
        
        # 2. Local Search
        ls_blocks = local_search(greedy_blocks)
        
        if len(ls_blocks) < best_len_overall:
            best_len_overall = len(ls_blocks)
            best_overall = ls_blocks
            
        # If we hit the lower bound or close, we might want to try harder, 
        # but for now, just keep restarting.

    if best_overall is None:
        # Fallback
        best_overall = run_greedy()

    return [list(all_blocks[i]) for i in best_overall]
