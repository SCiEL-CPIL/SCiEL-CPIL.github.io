import itertools
import time
import random
import sys

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
                            
    best_blocks = []
    best_len = float('inf')
    
    # Run multiple rounds of randomized greedy + local search
    # Budget: aim for ~10 rounds if each takes ~10s, or fewer if they take longer
    # We will use the remaining time to run as many rounds as possible.
    
    while time.time() < deadline - 15:
        # --- Phase 1: Randomized Greedy Construction ---
        uncovered_count_per_block = [len(block_to_tsubs[b]) for b in range(n_blocks)]
        uncovered = [1] * n_t_subs
        total_uncovered = n_t_subs
        solution_blocks = []
        solution_set = set()
        
        while total_uncovered > 0:
            if time.time() > deadline - 15:
                break
                
            # Find block with max uncovered coverage
            # Add randomness to break ties and explore different paths
            max_count = -1
            candidates = []
            for i in range(n_blocks):
                c = uncovered_count_per_block[i]
                if c > max_count:
                    max_count = c
                    candidates = [i]
                elif c == max_count:
                    candidates.append(i)
            
            if not candidates or max_count == 0:
                break
                
            # Randomly pick from candidates
            best_b_idx = random.choice(candidates)
            
            solution_blocks.append(best_b_idx)
            solution_set.add(best_b_idx)
            
            # Update counts
            for s in block_to_tsubs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
                    # Update uncovered count for all blocks covering s
                    for b_idx in tsub_to_blocks[s]:
                        uncovered_count_per_block[b_idx] -= 1
                        
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
