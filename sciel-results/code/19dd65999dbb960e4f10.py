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

# Precompute block overlap matrix for conflict graph
# overlap[i,j] = number of 4-subsets shared by blocks i and j
# This is expensive to compute fully, so we compute on-demand or use a faster method
# For the conflict graph, we need to know which blocks share coverage of deficit subsets

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
    deadline = start_time + 118  # Safety margin

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

    def run_packing_greedy():
        """
        Partition-Refinement Step 1: Build a packing that maximizes disjointly covered 4-subsets.
        We greedily select blocks that cover the most NEW (previously uncovered) 4-subsets,
        but with a bias toward blocks that have less overlap with already-selected blocks.
        This creates a more "spread out" covering that local search can refine better.
        """
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        # Track how many times each subset is covered (for overlap penalty)
        coverage_counts = np.zeros(N_SUBSETS, dtype=np.int32)
        
        while True:
            if not uncovered.any():
                break
            if time.time() > deadline - 15:
                break
            
            # For each block, compute:
            # 1. new_coverage = number of previously uncovered subsets it covers
            # 2. overlap_penalty = number of subsets it covers that are already covered
            # Score = new_coverage * 2.0 - overlap_penalty * 0.5
            # This biases toward blocks that add new coverage while minimizing redundancy
            
            # new_coverage for each block
            new_cov = (coverage_matrix * uncovered).sum(axis=1)
            
            # overlap_penalty: subsets already covered that this block also covers
            # = total subsets block covers - new_coverage
            # But we want to penalize blocks that overlap with existing coverage
            total_cov = coverage_matrix.sum(axis=1)
            overlap_pen = total_cov - new_cov
            
            # Weighted score
            score = new_cov * 2.0 - overlap_pen * 0.3
            
            best_idx = np.argmax(score)
            if new_cov[best_idx] == 0:
                break
                
            chosen.append(best_idx)
            uncovered[coverage_matrix[best_idx] == 1] = 0
            coverage_counts[coverage_matrix[best_idx] == 1] += 1

        return chosen

    def resolve_deficit(packing_blocks):
        """
        Partition-Refinement Step 2: Resolve the deficit using a conflict-graph-based
        minimum vertex cover heuristic.
        
        After the packing, some 4-subsets may still be uncovered. We build a conflict graph
        where vertices are candidate 7-blocks that can cover deficit subsets, and edges
        represent shared coverage. We use a greedy set cover on the deficit with the
        conflict graph to guide selection.
        """
        current = list(packing_blocks)
        current_set = set(current)
        
        # Find which subsets are still uncovered
        coverage = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in current:
            coverage[block_to_subsets[b_idx]] += 1
        
        deficit_subsets = np.where(coverage == 0)[0]
        if len(deficit_subsets) == 0:
            return current
        
        # Build candidate pool: blocks that cover at least one deficit subset
        # For each deficit subset, find blocks that cover it
        # We want to select a minimal set of blocks to cover all deficit subsets
        
        # Greedy set cover on deficit with conflict-aware scoring
        deficit_mask = np.zeros(N_SUBSETS, dtype=np.int8)
        deficit_mask[deficit_subsets] = 1
        
        while True:
            if time.time() > deadline - 15:
                break
            
            # Recompute deficit
            coverage = np.zeros(N_SUBSETS, dtype=np.int32)
            for b_idx in current:
                coverage[block_to_subsets[b_idx]] += 1
            
            deficit_subsets = np.where(coverage == 0)[0]
            if len(deficit_subsets) == 0:
                break
            
            deficit_mask = np.zeros(N_SUBSETS, dtype=np.int8)
            deficit_mask[deficit_subsets] = 1
            
            # For each candidate block (not already in current), compute:
            # 1. deficit_coverage = number of deficit subsets it covers
            # 2. conflict_score = how much it overlaps with existing blocks (lower is better)
            #    We approximate conflict by the number of subsets it shares with existing blocks
            
            # Get all blocks not in current
            all_block_indices = np.arange(N_BLOCKS)
            not_in_current = np.array([b not in current_set for b in all_block_indices])
            
            if not not_in_current.any():
                break
            
            # Deficit coverage for each block
            deficit_cov = (coverage_matrix * deficit_mask).sum(axis=1)
            
            # Only consider blocks that cover at least one deficit subset
            valid_mask = not_in_current & (deficit_cov > 0)
            if not valid_mask.any():
                break
            
            # Conflict score: for each valid block, count how many of its subsets
            # are already covered by existing blocks
            # This is expensive, so we approximate using the coverage counts
            # A block has high conflict if most of its subsets are already covered
            total_cov_per_block = coverage_matrix.sum(axis=1)
            # Approximate: conflict = total_cov_per_block - deficit_cov
            # (subsets already covered + subsets not in deficit)
            conflict_approx = total_cov_per_block - deficit_cov
            
            # Score: maximize deficit_cov, minimize conflict
            score = deficit_cov * 1.0 - conflict_approx * 0.2
            score[~valid_mask] = -np.inf
            
            best_idx = np.argmax(score)
            if deficit_cov[best_idx] == 0:
                break
            
            current.append(best_idx)
            current_set.add(best_idx)
        
        return current

    def prune_redundant(blocks_list):
        """Remove blocks that are redundant (all their subsets are covered by other blocks).
        Single forward pass with incremental count updates."""
        if not blocks_list:
            return blocks_list
        
        current = list(blocks_list)
        if not current:
            return current
        
        # Count coverage for each subset
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in current:
            counts[block_to_subsets[b_idx]] += 1
            
        new_blocks = []
        for b_idx in current:
            redundant = True
            for s_idx in block_to_subsets[b_idx]:
                if counts[s_idx] == 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(b_idx)
            else:
                # Remove this block and update counts
                counts[block_to_subsets[b_idx]] -= 1
        
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
            
        # Track which blocks are in current for O(1) membership
        current_set = set(current_blocks)
            
        no_improve = 0
        max_no_improve = 30
        
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
                    current_set.discard(b_out)
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
                        # Score each candidate by its coverage gain
                        # After removing b_out, a candidate b_in covers subsets where
                        # (counts[s] - 1 + 1) > 0, i.e., all subsets it covers.
                        # The "new" coverage from b_in (relative to state after removing b_out)
                        # is subsets s where counts[s] - 1 == 0 (i.e., counts[s] == 1) and b_in covers s.
                        # But we already know unique_subsets are those with counts==1.
                        # A better metric: how many subsets does b_in cover that have counts[s] <= 1?
                        # Actually, the simplest useful metric: total number of subsets b_in covers
                        # that are NOT already over-covered (counts[s] > 1).
                        # Let's just pick the candidate that covers the most "scarce" subsets.
                        # For speed, just pick randomly among top few by a quick score.
                        cand_list = list(cands)
                        if len(cand_list) <= 5:
                            b_in = random.choice(cand_list)
                        else:
                            # Quick score: count how many of b_in's subsets have counts[s] == 1
                            # (these would become uniquely covered by b_in after swap)
                            best_cand = cand_list[0]
                            best_score = -1
                            # Sample up to 10 candidates for speed
                            sample = random.sample(cand_list, min(10, len(cand_list)))
                            for b_c in sample:
                                score = 0
                                for s_idx in block_to_subsets[b_c]:
                                    if counts[s_idx] == 1:
                                        score += 1
                                if score > best_score:
                                    best_score = score
                                    best_cand = b_c
                            b_in = best_cand

                        update_counts_remove(b_out)
                        current_blocks.pop(i)
                        current_set.discard(b_out)
                        update_counts_add(b_in)
                        current_blocks.append(b_in)
                        current_set.add(b_in)
                        
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
                    if b_rand not in current_set:
                        current_blocks.append(b_rand)
                        current_set.add(b_rand)
                        update_counts_add(b_rand)
                no_improve = 0
                # Re-prune immediately
                pruned = prune_redundant(current_blocks)
                if len(pruned) < len(current_blocks):
                    counts = np.zeros(N_SUBSETS, dtype=np.int32)
                    for b_idx in pruned:
                        counts[block_to_subsets[b_idx]] += 1
                    current_blocks = pruned
                    current_set = set(current_blocks)
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()

        return best_blocks

    def polish(best_blocks):
        """Final high-precision polish: try all removals and 1-for-1 swaps exhaustively."""
        if not best_blocks:
            return best_blocks
        
        current = list(best_blocks)
        current = list(set(current))
        current = prune_redundant(current)
        
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in current:
            counts[block_to_subsets[b_idx]] += 1
        
        improved = True
        while improved:
            improved = False
            
            # Try removals
            for i in range(len(current)):
                b_out = current[i]
                can_remove = True
                for s_idx in block_to_subsets[b_out]:
                    if counts[s_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    counts[block_to_subsets[b_out]] -= 1
                    current.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-1 swaps
            for i in range(len(current)):
                b_out = current[i]
                unique_subsets = []
                for s_idx in block_to_subsets[b_out]:
                    if counts[s_idx] == 1:
                        unique_subsets.append(s_idx)
                
                if not unique_subsets:
                    # Shouldn't happen after removal pass, but handle it
                    counts[block_to_subsets[b_out]] -= 1
                    current.pop(i)
                    improved = True
                    break
                
                # Find candidates
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
                    # Try each candidate
                    for b_in in cands:
                        if b_in in current:
                            # Just remove b_out
                            counts[block_to_subsets[b_out]] -= 1
                            current.pop(i)
                            improved = True
                            break
                        else:
                            # Do the swap
                            counts[block_to_subsets[b_out]] -= 1
                            current.pop(i)
                            counts[block_to_subsets[b_in]] += 1
                            current.append(b_in)
                            improved = True
                            break
                    if improved:
                        break
            
            if time.time() > deadline - 3:
                break

        return current

    def run_random_greedy():
        """Random greedy: pick random blocks that cover at least one uncovered subset."""
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        
        while True:
            if not uncovered.any():
                break
            
            if time.time() > deadline - 5:
                break
            
            # Find all blocks that cover at least one uncovered subset
            effective_weights = uncovered  # binary
            gains = coverage_matrix.dot(effective_weights)
            
            valid_indices = np.where(gains > 0)[0]
            if len(valid_indices) == 0:
                break
            
            # Pick randomly from valid blocks
            best_idx = valid_indices[random.randint(0, len(valid_indices) - 1)]
            chosen.append(best_idx)
            uncovered[coverage_matrix[best_idx] == 1] = 0

        return chosen

    best_overall = None
    best_len_overall = float('inf')

    # Run multiple restarts with diverse starting points
    # Include packing-based initialization as a third strategy
    while time.time() < deadline - 15:
        r = random.random()
        if r < 0.3:
            initial_blocks = run_greedy()
        elif r < 0.5:
            initial_blocks = run_random_greedy()
        else:
            # Partition-Refinement: packing + deficit resolution
            packing = run_packing_greedy()
            initial_blocks = resolve_deficit(packing)
        
        # Local Search
        ls_blocks = local_search(initial_blocks)
        
        if len(ls_blocks) < best_len_overall:
            best_len_overall = len(ls_blocks)
            best_overall = ls_blocks

    # Final polish on the best solution found
    if best_overall is not None:
        polished = polish(best_overall)
        if len(polished) < best_len_overall:
            best_overall = polished
            best_len_overall = len(polished)
    else:
        # Fallback
        best_overall = run_greedy()

    return [list(all_blocks[i]) for i in best_overall]
