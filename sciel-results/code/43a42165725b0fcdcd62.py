import itertools
import time
import numpy as np
import sys
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
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

    best_solution = None
    best_size = float('inf')
    
    # Multiple restarts with different random seeds for tie-breaking in greedy
    # We will run a few iterations to try to find a better cover
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20.0: # Leave time for final local search on best
            break
            
        random.seed(restart * 12345 + 42)
        
        # --- Phase 1: Greedy Construction with Random Tie-Breaking ---
        uncovered_count_per_block = [len(block_to_tsubs[b]) for b in range(n_blocks)]
        uncovered = [1] * n_t_subs
        total_uncovered = n_t_subs
        solution_blocks = []
        solution_set = set()
        
        while total_uncovered > 0:
            if time.time() > deadline:
                break
                
            # Find block with max uncovered coverage
            # To introduce diversity, we can pick from top-N candidates randomly?
            # Or just break ties randomly.
            max_count = -1
            best_candidates = []
            
            # Optimization: If we find a block covering all its subsets, it's optimal
            # But for diversity, maybe we don't break immediately?
            # Let's stick to strict max, but randomize if there are multiple maxes.
            
            # Efficient scan
            for i in range(n_blocks):
                c = uncovered_count_per_block[i]
                if c > max_count:
                    max_count = c
                    best_candidates = [i]
                    # If c == len(block_to_tsubs[i]), it's a perfect cover for remaining? No, just max possible.
                    # We can't break early if we want to collect all maxes for randomization,
                    # but that's slow. Let's just break if we hit the theoretical max?
                    # Theoretical max is len(block_to_tsubs[i]) which varies? No, all blocks have C(k,t) subsets.
                    # So max possible is C(k,t).
                    if c == len(block_to_tsubs[i]):
                        # This block covers all its subsets. Any other block with same count is equivalent.
                        # We can break to save time, but we lose diversity.
                        # For now, let's break to ensure speed, and rely on local search.
                        break
                elif c == max_count:
                    best_candidates.append(i)
            
            if best_candidates is None or max_count == 0:
                break
                
            # Pick one from best_candidates
            best_b_idx = random.choice(best_candidates)
            
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
                    
        # Check if solution is valid (should be, but just in case)
        if total_uncovered > 0:
            # Fallback: if greedy didn't finish, try to force completion? 
            # Usually greedy finishes.
            continue

        # --- Phase 2: Local Search ---
        # Initialize cover_count
        cover_count = [0] * n_t_subs
        for b_idx in solution_blocks:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
                
        improved = True
        local_search_start = time.time()
        
        while improved and (time.time() - local_search_start) < 20.0: # Limit local search per restart
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
                    
            # Pass 2: If no removal, try to swap a block
            if not improved:
                for i in range(len(solution_blocks)):
                    b_idx = solution_blocks[i]
                    # Find critical subsets (covered only by this block)
                    critical_subsets = []
                    for s in block_to_tsubs[b_idx]:
                        if cover_count[s] == 1:
                            critical_subsets.append(s)
                            
                    if not critical_subsets:
                        # Should have been removed in Pass 1
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_blocks.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                        break
                        
                    # Find a candidate block that covers all critical subsets
                    # Intersect tsub_to_blocks for all critical subsets
                    candidate_set = set(tsub_to_blocks[critical_subsets[0]])
                    for s in critical_subsets[1:]:
                        candidate_set.intersection_update(tsub_to_blocks[s])
                        if not candidate_set:
                            break
                            
                    if candidate_set:
                        # Filter out current solution blocks
                        valid_candidates = [c for c in candidate_set if c not in solution_set]
                        if valid_candidates:
                            # Pick the best candidate
                            # Heuristic: Pick the one that covers the most subsets overall?
                            # Or just pick one that maximizes the number of subsets it shares with others?
                            # Let's pick the one with the highest "potential" to enable removal.
                            # A simple metric: count how many subsets in the candidate are already covered by >=2 blocks.
                            # If a candidate covers many "safe" subsets, it might not help.
                            # We want a candidate that covers the critical ones (guaranteed) and maybe helps elsewhere.
                            # Actually, any valid swap maintains coverage. The hope is that the new block
                            # allows *other* blocks to become redundant.
                            
                            # Let's just pick the first one, or randomize to escape local minima.
                            best_candidate = random.choice(valid_candidates)
                            
                            # Swap: Remove b_idx, Add best_candidate
                            # Update cover counts
                            for s in block_to_tsubs[b_idx]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1
                                
                            # Replace in solution
                            solution_blocks[i] = best_candidate
                            solution_set.remove(b_idx)
                            solution_set.add(best_candidate)
                            improved = True
                            break
                        
        # Check if this solution is better
        if len(solution_blocks) < best_size:
            best_size = len(solution_blocks)
            best_solution = solution_blocks[:]
            
    # Convert best solution block indices back to lists of points
    if best_solution:
        final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution]
        return final_blocks
    else:
        # Fallback: return empty or a trivial valid cover? 
        # The task requires a valid cover. If we failed, we should return something valid.
        # But greedy should always succeed.
        return []
