import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Pre-compute all t-subsets (targets)
    all_targets = list(itertools.combinations(range(v), t))
    num_targets = len(all_targets)
    
    # Map target to index
    target_to_idx = {tup: i for i, tup in enumerate(all_targets)}
    
    # Pre-compute for each block, the list of target indices it covers
    block_to_targets = []
    for block in all_blocks:
        targets = []
        for t_sub in itertools.combinations(block, t):
            targets.append(target_to_idx[t_sub])
        block_to_targets.append(targets)
    
    # Convert to numpy arrays for faster membership checks
    block_to_targets_np = [np.array(targets, dtype=np.int32) for targets in block_to_targets]
    
    # Initialize uncovered targets
    uncovered = set(range(num_targets))
    
    # Greedy Set Cover
    selected_indices = []
    selected_set = set()
    
    while uncovered:
        if time.time() > deadline - 5.0:
            break
            
        best_gain = 0
        best_block_idx = -1
        max_possible = len(uncovered)
        
        # Find the block that covers the most uncovered targets
        for i in range(num_blocks):
            if i in selected_set:
                continue
            
            # Calculate gain
            targets_arr = block_to_targets_np[i]
            gain = 0
            for t_idx in targets_arr:
                if int(t_idx) in uncovered:
                    gain += 1
                    if gain >= max_possible:
                        break
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                if best_gain >= max_possible:
                    break
        
        if best_block_idx == -1 or best_gain == 0:
            break
            
        selected_indices.append(best_block_idx)
        selected_set.add(best_block_idx)
        
        # Remove covered targets from uncovered
        for t_idx in block_to_targets_np[best_block_idx]:
            uncovered.discard(int(t_idx))
    
    # Post-processing: Try to remove redundant blocks using coverage counts
    if time.time() < deadline - 10.0 and len(selected_indices) > 0:
        # Maintain coverage count for each target
        coverage_count = np.zeros(num_targets, dtype=np.int32)
        for bi in selected_indices:
            for t_idx in block_to_targets_np[bi]:
                coverage_count[int(t_idx)] += 1
        
        # Iteratively remove redundant blocks
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_indices = []
            for bi in selected_indices:
                # Check if this block is redundant: all its targets have coverage >= 2
                targets_arr = block_to_targets_np[bi]
                redundant = True
                for t_idx in targets_arr:
                    if coverage_count[int(t_idx)] < 2:
                        redundant = False
                        break
                
                if redundant:
                    # Remove this block and update coverage counts
                    for t_idx in targets_arr:
                        coverage_count[int(t_idx)] -= 1
                    changed = True
                else:
                    new_indices.append(bi)
            
            if changed:
                selected_indices = new_indices
        
        # Final verification
        if selected_indices:
            covered_targets = set()
            for bi in selected_indices:
                for t_idx in block_to_targets_np[bi]:
                    covered_targets.add(int(t_idx))
            
            if len(covered_targets) == num_targets:
                selected_blocks = [all_blocks[i] for i in selected_indices]
            else:
                # Fallback: re-run greedy to ensure validity
                uncovered = set(range(num_targets))
                selected_indices = []
                selected_set = set()
                while uncovered:
                    if time.time() > deadline - 2.0:
                        break
                    best_gain = 0
                    best_block_idx = -1
                    for i in range(num_blocks):
                        if i in selected_set:
                            continue
                        targets_arr = block_to_targets_np[i]
                        gain = sum(1 for t_idx in targets_arr if int(t_idx) in uncovered)
                        if gain > best_gain:
                            best_gain = gain
                            best_block_idx = i
                    if best_block_idx == -1 or best_gain == 0:
                        break
                    selected_indices.append(best_block_idx)
                    selected_set.add(best_block_idx)
                    for t_idx in block_to_targets_np[best_block_idx]:
                        uncovered.discard(int(t_idx))
                selected_blocks = [all_blocks[i] for i in selected_indices]
        else:
            selected_blocks = []
    else:
        selected_blocks = [all_blocks[i] for i in selected_indices]
    
    return selected_blocks
