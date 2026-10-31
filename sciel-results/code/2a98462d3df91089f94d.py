import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute blocks and their subsets
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # For each block, list of subset indices it covers
    block_to_subsets = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(subset_to_idx[s])
        block_to_subsets.append(subs)
        
    # For each subset, list of block indices that cover it
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)
            
    best_solution = None
    best_len = float('inf')
    
    def greedy_randomized():
        # Returns a list of block indices
        coverage = [0] * num_subsets
        solution = []
        uncovered_count = num_subsets
        
        # To speed up greedy, we can just pick the first uncovered subset 
        # and pick the best block for it, or iterate all uncovered to find global best.
        # Global best is better for quality.
        
        while uncovered_count > 0:
            if time.time() > deadline:
                return solution if len(solution) < best_len else None
                
            # Calculate score for each block: number of currently uncovered subsets it covers
            # We only care about blocks that cover at least one uncovered subset.
            scores = [0] * num_blocks
            # Optimization: Only iterate over uncovered subsets
            # This is O(num_uncovered * avg_blocks_per_subset)
            # avg_blocks_per_subset is C(19-4, 7-4) = C(15, 3) = 455.
            # num_uncovered starts at 3876.
            # 3876 * 455 ~ 1.7M operations per step.
            # Number of steps ~ 150.
            # Total ~ 250M operations. Might be too slow in Python.
            
            # Alternative: Iterate over all blocks, check if they cover any uncovered subset.
            # O(num_blocks * t) ? No, we need to check coverage.
            
            # Let's try the "Vote" method:
            # For each uncovered subset, increment score of its blocks.
            for s_idx in range(num_subsets):
                if coverage[s_idx] == 0:
                    for b_idx in subset_to_blocks[s_idx]:
                        scores[b_idx] += 1
            
            # Find max score
            max_score = -1
            best_b = -1
            # Randomize tie-breaking
            candidates = []
            for b_idx in range(num_blocks):
                if scores[b_idx] > max_score:
                    max_score = scores[b_idx]
                    candidates = [b_idx]
                elif scores[b_idx] == max_score and scores[b_idx] > 0:
                    candidates.append(b_idx)
            
            if not candidates:
                break # Should not happen if uncovered > 0
                
            # Random choice among best candidates
            best_b = random.choice(candidates)
            
            solution.append(best_b)
            
            # Update coverage
            for s_idx in block_to_subsets[best_b]:
                if coverage[s_idx] == 0:
                    uncovered_count -= 1
                coverage[s_idx] += 1
                
        return solution

    def local_search(solution):
        # solution is a list of block indices
        # Returns improved solution
        coverage = [0] * num_subsets
        for b in solution:
            for s in block_to_subsets[b]:
                coverage[s] += 1
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try to remove or swap each block
            # We iterate over a copy or modify in place carefully
            # To avoid index issues, we can use a list of blocks and try to remove/replace
            # Simpler: iterate through blocks, if redundant, remove. If critical, try swap.
            
            # We need to be careful with indices if we remove blocks.
            # Let's use a set of block indices for the solution to check membership quickly?
            # Or just a list.
            
            n = len(solution)
            for i in range(n):
                if time.time() > deadline:
                    break
                    
                b = solution[i]
                # Check if redundant
                critical_subs = []
                for s in block_to_subsets[b]:
                    if coverage[s] == 1:
                        critical_subs.append(s)
                        
                if not critical_subs:
                    # Redundant, remove
                    for s in block_to_subsets[b]:
                        coverage[s] -= 1
                    solution.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                    
                else:
                    # Critical, try to find a replacement
                    # Find blocks that cover ALL critical subs
                    # Intersection of subset_to_blocks[s] for s in critical_subs
                    if len(critical_subs) > 0:
                        # Start with the first critical sub's blocks
                        candidates = set(subset_to_blocks[critical_subs[0]])
                        for s in critical_subs[1:]:
                            candidates.intersection_update(subset_to_blocks[s])
                            if not candidates:
                                break
                                
                        if candidates:
                            # Filter out the current block and any block already in solution
                            # Actually, we can swap with any block not in solution.
                            # If we swap with a block already in solution, we just have a duplicate, which is bad.
                            # So we must pick a block NOT in solution.
                            sol_set = set(solution)
                            valid_candidates = [c for c in candidates if c not in sol_set]
                            
                            if valid_candidates:
                                # Pick one (random or best?)
                                # Best would be one that covers most other uncovered/subsets?
                                # For now, just pick random or first.
                                new_b = random.choice(valid_candidates)
                                
                                # Perform swap
                                # Remove b
                                for s in block_to_subsets[b]:
                                    coverage[s] -= 1
                                # Add new_b
                                for s in block_to_subsets[new_b]:
                                    coverage[s] += 1
                                    
                                solution[i] = new_b
                                improved = True
                                break # Restart loop? Or continue?
                                # Restarting is safer to catch new redundancies created by the swap.
                                # But it might be slow. Let's try continuing for now, but mark for restart?
                                # Actually, a swap can make other blocks redundant.
                                # Let's break and restart the while loop to be safe.
                                
        return solution

    # Run multiple times
    for run in range(20):
        if time.time() > deadline:
            break
            
        sol = greedy_randomized()
        if sol is None:
            continue
            
        sol = local_search(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol.copy()
            
    if best_solution is None:
        # Fallback: just return a valid covering, even if large
        # Greedy should always produce one.
        sol = greedy_randomized()
        return [list(all_blocks[b]) for b in sol]
        
    return [list(all_blocks[b]) for b in best_solution]

if __name__ == "__main__":
    # Test
    blocks = construct(19, 7, 4)
    print(len(blocks))
