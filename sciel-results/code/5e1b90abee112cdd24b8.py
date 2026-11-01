import itertools
import random
import time
import math

def construct(v, k, t):
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N = len(all_blocks)
    
    all_targets = list(itertools.combinations(points, t))
    M = len(all_targets)
    
    target_index = {target: i for i, target in enumerate(all_targets)}
    
    # Precompute subsets for each block
    block_subsets = []
    for block in all_blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(target_index[sub])
        block_subsets.append(subs)
        
    # Precompute target_to_blocks
    target_to_blocks = [[] for _ in range(M)]
    for i, subs in enumerate(block_subsets):
        for j in subs:
            target_to_blocks[j].append(i)
            
    random.seed(42)
    
    # State variables
    cov = [0] * M
    x = [0] * N  # 1 if selected
    selected = []
    
    # Helper: Add block i
    def add_block(i):
        if x[i]:
            return
        x[i] = 1
        selected.append(i)
        for s in block_subsets[i]:
            cov[s] += 1
            
    # Helper: Remove block i
    def remove_block(i):
        if not x[i]:
            return
        x[i] = 0
        # Remove from selected list efficiently
        # Since we don't need order, we can just swap-pop if we track index, 
        # but finding index is O(N). 
        # Given small N_selected, linear search in 'selected' is fine.
        try:
            idx = selected.index(i)
            selected[idx] = selected[-1]
            selected.pop()
        except ValueError:
            pass
        for s in block_subsets[i]:
            cov[s] -= 1
            
    # Helper: Get uncovered targets
    def get_uncovered():
        return [i for i in range(M) if cov[i] == 0]
        
    # Helper: Fast Greedy Step
    # Picks one block to add, targeting a specific uncovered subset
    def greedy_step(uncovered_list):
        if not uncovered_list:
            return False
        # Pick a random uncovered target
        t_idx = random.choice(uncovered_list)
        candidates = target_to_blocks[t_idx]
        
        best_block = -1
        max_new = -1
        
        for b in candidates:
            if x[b]:
                continue
            # Count how many subsets in b are currently uncovered
            new_cov_count = 0
            for s in block_subsets[b]:
                if cov[s] == 0:
                    new_cov_count += 1
            if new_cov_count > max_new:
                max_new = new_cov_count
                best_block = b
                
        if best_block != -1:
            add_block(best_block)
            return True
        return False
        
    # Helper: Remove Redundant Blocks
    def remove_redundant():
        removed = True
        while removed:
            removed = False
            # Iterate over a copy since we modify selected
            for i in selected[:]:
                # Check if block i is redundant
                # i.e., for all s in block_subsets[i], cov[s] > 1
                redundant = True
                for s in block_subsets[i]:
                    if cov[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(i)
                    removed = True
                    
    # 1. Initialization: Fast Greedy
    uncovered = list(range(M))
    while uncovered:
        greedy_step(uncovered)
        # Update uncovered list
        # To be efficient, we can just rebuild it, M is 11k
        uncovered = [i for i in uncovered if cov[i] == 0]
        
    remove_redundant()
    
    current_best = len(selected)
    best_selected = selected.copy()
    best_x = x.copy()
    best_cov = cov.copy()
    
    start_time = time.time()
    time_limit = 110.0
    
    # 2. Local Search: Perturb and Repair
    while time.time() - start_time < time_limit:
        # Check for improvement
        if len(selected) < current_best:
            current_best = len(selected)
            best_selected = selected.copy()
            best_x = x.copy()
            best_cov = cov.copy()
            
        # Perturb: Remove a random block
        if not selected:
            break
        victim = random.choice(selected)
        remove_block(victim)
        
        # Repair
        # We need to cover the subsets that victim covered
        # Greedy repair
        # Find uncovered
        uncovered = [i for i in range(M) if cov[i] == 0]
        while uncovered:
            if not greedy_step(uncovered):
                # Should not happen if target_to_blocks is complete
                break
            uncovered = [i for i in uncovered if cov[i] == 0]
            
        remove_redundant()
        
        # If we got stuck or something went wrong, revert?
        # The repair should always restore validity.
        
    # Final check and output
    # Use best found
    x = best_x
    selected = best_selected
    
    blocks = []
    for i in selected:
        blocks.append(list(all_blocks[i]))
        
    # Ensure distinct and sorted
    unique_blocks = []
    seen = set()
    for b in blocks:
        b.sort()
        t_b = tuple(b)
        if t_b not in seen:
            seen.add(t_b)
            unique_blocks.append(b)
            
    return unique_blocks
