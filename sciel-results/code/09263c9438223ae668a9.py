import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block to subsets mapping
    block_to_subsets = []
    for block in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_to_subsets.append(subs)
        
    # Precompute subset to blocks mapping
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s in subs:
            subset_to_blocks[s].append(b_idx)

    def get_greedy_solution():
        coverage = np.zeros(num_subsets, dtype=np.int32)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        selected = []
        
        for _ in range(num_blocks):
            if np.all(coverage > 0):
                break
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            for s in block_to_subsets[best_idx]:
                if coverage[s] == 0:
                    for b in subset_to_blocks[s]:
                        gains[b] -= 1
                coverage[s] += 1
                
        return selected, coverage

    start_time = time.time()
    time_limit = 115
    
    current_set, coverage = get_greedy_solution()
    best_set = list(current_set)
    best_len = len(current_set)
    current_set_set = set(current_set)
    
    all_block_indices = list(range(num_blocks))

    def is_redundant(block_idx):
        subs = block_to_subsets[block_idx]
        for s in subs:
            if coverage[s] == 1:
                return False
        return True

    def remove_block(block_idx):
        for s in block_to_subsets[block_idx]:
            coverage[s] -= 1
        current_set.remove(block_idx)
        current_set_set.remove(block_idx)

    def add_block(block_idx):
        for s in block_to_subsets[block_idx]:
            coverage[s] += 1
        current_set.append(block_idx)
        current_set_set.add(block_idx)

    def run_deletion_pass():
        changed = True
        while changed:
            changed = False
            random.shuffle(current_set)
            for b in current_set:
                if is_redundant(b):
                    remove_block(b)
                    changed = True
                    break

    # Local Search Loop
    while time.time() - start_time < time_limit:
        # 1. Deletion
        run_deletion_pass()
        
        if len(current_set) < best_len:
            best_len = len(current_set)
            best_set = list(current_set)
            
        # 2. Move
        move_type = random.random()
        
        if move_type < 0.6:
            # Swap
            if current_set:
                b_old = random.choice(current_set)
                # Find a good replacement
                # Calculate "value" of b_old: number of subsets it covers uniquely
                unique_count = 0
                for s in block_to_subsets[b_old]:
                    if coverage[s] == 1:
                        unique_count += 1
                
                if unique_count == 0:
                    # Can just remove? Deletion pass would have caught it.
                    # So unique_count > 0.
                    pass
                
                # Pick a random block to swap in
                # Bias towards blocks that cover the unique subsets of b_old
                candidates = []
                if unique_count > 0:
                    # Intersection of subset_to_blocks for unique subsets
                    # This is expensive if done naively.
                    # Let's just pick a random block and check.
                    # Or pick a few random blocks and score them.
                    for _ in range(10):
                        b_new = random.choice(all_block_indices)
                        if b_new in current_set_set:
                            continue
                        # Check if b_new covers all unique subsets of b_old
                        covers_all = True
                        for s in block_to_subsets[b_old]:
                            if coverage[s] == 1 and s not in block_to_subsets[b_new]:
                                covers_all = False
                                break
                        if covers_all:
                            candidates.append(b_new)
                
                if not candidates:
                    # Pick any random block
                    b_new = random.choice(all_block_indices)
                    if b_new in current_set_set:
                        continue
                    candidates = [b_new]
                
                b_new = random.choice(candidates)
                
                # Perform swap
                old_coverage = coverage.copy()
                old_set = list(current_set)
                
                remove_block(b_old)
                add_block(b_new)
                
                # Check validity
                if np.any(coverage == 0):
                    # Revert
                    coverage[:] = old_coverage
                    current_set.clear()
                    current_set_set.clear()
                    for b in old_set:
                        current_set.append(b)
                        current_set_set.add(b)
                else:
                    run_deletion_pass()
                    if len(current_set) < best_len:
                        best_len = len(current_set)
                        best_set = list(current_set)
                        
        elif move_type < 0.8:
            # Add
            b_new = random.choice(all_block_indices)
            if b_new in current_set_set:
                continue
            
            old_coverage = coverage.copy()
            old_set = list(current_set)
            
            add_block(b_new)
            run_deletion_pass()
            
            if len(current_set) < len(old_set):
                if len(current_set) < best_len:
                    best_len = len(current_set)
                    best_set = list(current_set)
            else:
                # Revert
                coverage[:] = old_coverage
                current_set.clear()
                current_set_set.clear()
                for b in old_set:
                    current_set.append(b)
                    current_set_set.add(b)

        else:
            # Remove (and repair)
            if current_set:
                b_old = random.choice(current_set)
                old_coverage = coverage.copy()
                old_set = list(current_set)
                
                remove_block(b_old)
                
                # Repair: Add blocks to cover uncovered subsets
                # Simple greedy repair
                uncovered = np.where(coverage == 0)[0]
                while len(uncovered) > 0:
                    s = random.choice(uncovered)
                    candidates = [b for b in subset_to_blocks[s] if b not in current_set_set]
                    if not candidates:
                        break
                    b_new = random.choice(candidates)
                    add_block(b_new)
                    uncovered = np.where(coverage == 0)[0]
                    
                if len(current_set) < len(old_set):
                    if len(current_set) < best_len:
                        best_len = len(current_set)
                        best_set = list(current_set)
                else:
                    # Revert
                    coverage[:] = old_coverage
                    current_set.clear()
                    current_set_set.clear()
                    for b in old_set:
                        current_set.append(b)
                        current_set_set.add(b)

    return [list(all_blocks[i]) for i in best_set]
