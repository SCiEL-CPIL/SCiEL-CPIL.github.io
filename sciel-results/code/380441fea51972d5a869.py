import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)
        
    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)
            
    def local_search(solution_blocks, solution_set, cover_count):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Pass 1: Try to remove any redundant block
            for i in range(len(solution_blocks)):
                b_idx = solution_blocks[i]
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_blocks.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break # Restart loop as indices changed
                    
            # Pass 2: If no removal, try to swap a block to create redundancy elsewhere
            if not improved and time.time() < deadline - 5:
                # We iterate through blocks to see if we can replace one with another
                # such that the new block covers the "critical" subsets of the old one.
                # This might allow removal of the old one or other blocks later.
                for i in range(len(solution_blocks)):
                    b_idx = solution_blocks[i]
                    # Find critical subsets (covered only by this block)
                    critical_subsets = []
                    for s in block_to_tsubs[b_idx]:
                        if cover_count[s] == 1:
                            critical_subsets.append(s)
                            
                    if not critical_subsets:
                        # Should have been removed in Pass 1, but just in case
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_blocks.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                        break
                        
                    # Find a candidate block that covers all critical subsets
                    # Intersect tsub_to_blocks for all critical subsets
                    # Start with the first critical subset's blocks
                    candidate_set = set(tsub_to_blocks[critical_subsets[0]])
                    for s in critical_subsets[1:]:
                        candidate_set.intersection_update(tsub_to_blocks[s])
                        if not candidate_set:
                            break
                            
                    if candidate_set:
                        # Pick a candidate not in current solution
                        # Prefer one that covers many other subsets? For now, just pick any valid one.
                        # To be more strategic, pick the one with highest 'potential' (e.g., covers most subsets)
                        best_candidate = -1
                        max_potential = -1
                        for c_idx in candidate_set:
                            if c_idx not in solution_set:
                                # Heuristic: prefer blocks that cover more subsets in general?
                                # Or just pick the first one. Let's pick the one with max index to diversify?
                                # Actually, let's just pick the first one not in solution.
                                best_candidate = c_idx
                                break
                                
                        if best_candidate != -1:
                            # Swap: Remove b_idx, Add best_candidate
                            # Update cover counts
                            # Decrement for b_idx
                            for s in block_to_tsubs[b_idx]:
                                cover_count[s] -= 1
                            # Increment for best_candidate
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1
                                
                            # Replace in solution
                            solution_blocks[i] = best_candidate
                            solution_set.remove(b_idx)
                            solution_set.add(best_candidate)
                            improved = True
                            break
                            
            # Pass 3: If still no improvement, try a 2-for-1 swap (replace 2 blocks with 1)
            if not improved and time.time() < deadline - 5:
                # This is more expensive but can escape deeper local minima
                # For each pair of blocks, check if their union of critical subsets can be covered by a single new block
                # Actually, simpler: for each block i, for each block j > i, 
                # check if there exists a block c that covers ALL critical subsets of i AND ALL critical subsets of j
                # If so, remove i and j, add c. This reduces size by 1.
                
                # To make this efficient, we only consider pairs where the union of critical subsets is small.
                # Let's just try a random sample of pairs if time permits.
                
                # Get list of critical subsets for each block
                criticals = []
                for i in range(len(solution_blocks)):
                    b_idx = solution_blocks[i]
                    crits = [s for s in block_to_tsubs[b_idx] if cover_count[s] == 1]
                    criticals.append(crits)
                    
                # Try a few random pairs
                n_pairs_to_try = 50
                for _ in range(n_pairs_to_try):
                    if time.time() > deadline - 5:
                        break
                    i, j = random.sample(range(len(solution_blocks)), 2)
                    b_i = solution_blocks[i]
                    b_j = solution_blocks[j]
                    
                    all_crits = criticals[i] + criticals[j]
                    if not all_crits:
                        continue # Should have been removed in Pass 1
                        
                    # Find candidate covering all all_crits
                    candidate_set = set(tsub_to_blocks[all_crits[0]])
                    for s in all_crits[1:]:
                        candidate_set.intersection_update(tsub_to_blocks[s])
                        if not candidate_set:
                            break
                    
                    if candidate_set:
                        # Check if candidate is not b_i or b_j
                        valid_candidates = [c for c in candidate_set if c not in solution_set]
                        if valid_candidates:
                            best_candidate = valid_candidates[0]
                            # Perform swap
                            # Remove b_i and b_j
                            for s in block_to_tsubs[b_i]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[b_j]:
                                cover_count[s] -= 1
                            # Add best_candidate
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1
                                
                            # Update solution lists
                            # Careful with indices if i < j
                            if i < j:
                                solution_blocks.pop(j)
                                solution_blocks.pop(i)
                            else:
                                solution_blocks.pop(i)
                                solution_blocks.pop(j)
                            solution_set.remove(b_i)
                            solution_set.remove(b_j)
                            solution_set.add(best_candidate)
                            solution_blocks.append(best_candidate)
                            
                            improved = True
                            break
                            
    best_blocks = []
    best_len = float('inf')
    
    # Run multiple rounds of randomized greedy + local search
    # Budget: aim for ~10 rounds if each takes ~10s, or fewer if they take longer
    # We will use the remaining time to run as many rounds as possible.
    
    while time.time() < deadline - 15:
        # --- Phase 1: Randomized Lagrangian Greedy Construction ---
        # Use random dual weights to break ties and explore different paths
        # This acts as a smooth surrogate for the hard selection, allowing diversity
        
        # Initialize dual weights with random noise to create diversity
        # Base weight 1.0, noise up to 0.5
        duals = np.random.uniform(1.0, 1.5, size=n_t_subs)
        uncovered = np.ones(n_t_subs, dtype=bool)
        total_uncovered = n_t_subs
        solution_blocks = []
        solution_set = set()
        
        while total_uncovered > 0:
            if time.time() > deadline - 15:
                break
                
            # Find block with max weighted uncovered coverage
            # gain[b] = sum(duals[s] for s in block_to_tsubs[b] if uncovered[s])
            
            max_gain = -1.0
            best_b_idx = -1
            
            # Optimization: Use Python loop for gain calculation
            for i in range(n_blocks):
                gain = 0.0
                for s in block_to_tsubs[i]:
                    if uncovered[s]:
                        gain += duals[s]
                
                if gain > max_gain:
                    max_gain = gain
                    best_b_idx = i
            
            if best_b_idx == -1 or max_gain <= 0:
                break
                
            solution_blocks.append(best_b_idx)
            solution_set.add(best_b_idx)
            
            # Update uncovered status
            for s in block_to_tsubs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = False
                    total_uncovered -= 1
        
        # Check if we have a valid cover
        if total_uncovered > 0:
            continue
            
        # --- Phase 2: Local Search ---
        # Initialize cover_count: number of blocks in solution covering each t-subset
        cover_count = [0] * n_t_subs
        for b_idx in solution_blocks:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
                
        local_search(solution_blocks, solution_set, cover_count)
        
        # Update best if improved
        if len(solution_blocks) < best_len:
            best_len = len(solution_blocks)
            best_blocks = list(solution_blocks)
            
        # If we hit the lower bound or a very good number, we might stop early
        # But for now, just keep going until time runs out.
        
    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return final_blocks
