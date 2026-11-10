import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Precompute for each subset, the list of blocks that contain it
    subset_candidates = []
    for sub in all_subsets:
        sub_set = set(sub)
        remaining = [x for x in range(v) if x not in sub_set]
        num_add = k - t
        candidates = []
        for add_combo in itertools.combinations(remaining, num_add):
            block = tuple(sorted(sub + add_combo))
            candidates.append(block)
        subset_candidates.append(candidates)
    
    # Precompute for each block (as a frozenset or tuple), the list of t-subsets it covers
    block_to_subsets = {}
    def get_block_subsets(block):
        if block not in block_to_subsets:
            subs = []
            for sub in itertools.combinations(block, t):
                subs.append(subset_to_id[sub])
            block_to_subsets[block] = subs
        return block_to_subsets[block]
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    # Greedy construction
    def greedy_construct():
        uncovered = bytearray([1] * n_subsets)
        uncovered_ids = list(range(n_subsets))
        id_to_idx = {i: i for i in range(n_subsets)}
        
        blocks = []
        block_set = set()
        
        while uncovered_ids:
            # Pick random uncovered subset
            idx = random.randrange(len(uncovered_ids))
            t_id = uncovered_ids[idx]
            
            # Get precomputed candidates for this subset
            candidates = subset_candidates[t_id]
            
            # Calculate gain for each candidate
            best_gain = -1
            best_cands = []
            
            for block in candidates:
                if block in block_set:
                    continue
                # Calculate gain
                gain = 0
                for sub_id in get_block_subsets(block):
                    if uncovered[sub_id]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [block]
                elif gain == best_gain:
                    best_cands.append(block)
            
            if not best_cands:
                break
            
            chosen_block = random.choice(best_cands)
            
            blocks.append(list(chosen_block))
            block_set.add(chosen_block)
            
            # Update uncovered
            for sub_id in get_block_subsets(chosen_block):
                if uncovered[sub_id]:
                    uncovered[sub_id] = 0
                    sub_idx = id_to_idx[sub_id]
                    last_idx = len(uncovered_ids) - 1
                    temp_id = uncovered_ids[sub_idx]
                    uncovered_ids[sub_idx] = uncovered_ids[last_idx]
                    uncovered_ids[last_idx] = temp_id
                    if sub_idx != last_idx:
                        id_to_idx[uncovered_ids[sub_idx]] = sub_idx
                    uncovered_ids.pop()
                    del id_to_idx[sub_id]
                    
        return blocks
    
    # Simulated annealing
    def simulated_annealing(initial_blocks, max_time):
        blocks = [list(b) for b in initial_blocks]
        block_set = set(tuple(b) for b in blocks)
        
        # Build coverage counts
        coverage = np.zeros(n_subsets, dtype=np.int32)
        for block in blocks:
            for sub_id in get_block_subsets(tuple(block)):
                coverage[sub_id] += 1
        
        current_len = len(blocks)
        best_blocks = [list(b) for b in blocks]
        best_len = current_len
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.1
        cooling_rate = 0.995
        
        T = T_start
        
        while time.time() < max_time:
            if T < T_end:
                break
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Try to remove a block
                if len(blocks) == 0:
                    break
                rem_idx = random.randrange(len(blocks))
                rem_block = tuple(blocks[rem_idx])
                
                # Check if removing it keeps coverage
                can_remove = True
                for sub_id in get_block_subsets(rem_block):
                    if coverage[sub_id] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove the block
                    for sub_id in get_block_subsets(rem_block):
                        coverage[sub_id] -= 1
                    blocks.pop(rem_idx)
                    block_set.discard(rem_block)
                    
                    new_len = len(blocks)
                    if new_len < best_len:
                        best_len = new_len
                        best_blocks = [list(b) for b in blocks]
                    
                    T *= cooling_rate
                else:
                    # Accept with probability based on temperature
                    delta = 1  # Length increases by 1 if we don't remove
                    if random.random() < math.exp(-delta / T):
                        # Stay the same (no move made)
                        pass
                    T *= cooling_rate
                    
            elif move_type < 0.9:
                # Try to add a random block
                if len(blocks) >= 200:  # Don't add too many
                    T *= cooling_rate
                    continue
                    
                # Pick a random subset and a random candidate block
                t_id = random.randrange(n_subsets)
                candidates = subset_candidates[t_id]
                if not candidates:
                    T *= cooling_rate
                    continue
                    
                add_block = tuple(random.choice(candidates))
                
                if add_block in block_set:
                    T *= cooling_rate
                    continue
                
                # Add the block
                for sub_id in get_block_subsets(add_block):
                    coverage[sub_id] += 1
                blocks.append(list(add_block))
                block_set.add(add_block)
                
                # Now try to remove some redundant blocks
                improved = False
                for _ in range(5):
                    if not blocks:
                        break
                    rem_idx = random.randrange(len(blocks))
                    rem_block = tuple(blocks[rem_idx])
                    
                    can_remove = True
                    for sub_id in get_block_subsets(rem_block):
                        if coverage[sub_id] <= 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        for sub_id in get_block_subsets(rem_block):
                            coverage[sub_id] -= 1
                        blocks.pop(rem_idx)
                        block_set.discard(rem_block)
                        improved = True
                        
                        if len(blocks) < best_len:
                            best_len = len(blocks)
                            best_blocks = [list(b) for b in blocks]
                    
                    if time.time() > max_time - 2.0:
                        break
                
                T *= cooling_rate
                
            else:
                # Swap: remove one block, add another
                if len(blocks) < 2:
                    T *= cooling_rate
                    continue
                    
                rem_idx = random.randrange(len(blocks))
                rem_block = tuple(blocks[rem_idx])
                
                # Find a block to add that covers some of the same subsets
                t_id = random.randrange(n_subsets)
                candidates = subset_candidates[t_id]
                if not candidates:
                    T *= cooling_rate
                    continue
                    
                add_block = tuple(random.choice(candidates))
                
                if add_block in block_set:
                    T *= cooling_rate
                    continue
                
                # Check if swap is valid (after removal, all subsets still covered)
                # First, tentatively remove
                for sub_id in get_block_subsets(rem_block):
                    coverage[sub_id] -= 1
                
                # Check coverage
                valid = True
                for sub_id in range(n_subsets):
                    if coverage[sub_id] < 1:
                        valid = False
                        break
                
                if valid:
                    # Add the new block
                    for sub_id in get_block_subsets(add_block):
                        coverage[sub_id] += 1
                    
                    # Update blocks list
                    blocks[rem_idx] = list(add_block)
                    block_set.discard(rem_block)
                    block_set.add(add_block)
                    
                    # Try to remove redundant blocks
                    for _ in range(3):
                        if not blocks:
                            break
                        rem_idx2 = random.randrange(len(blocks))
                        rem_block2 = tuple(blocks[rem_idx2])
                        
                        can_remove = True
                        for sub_id in get_block_subsets(rem_block2):
                            if coverage[sub_id] <= 1:
                                can_remove = False
                                break
                        
                        if can_remove:
                            for sub_id in get_block_subsets(rem_block2):
                                coverage[sub_id] -= 1
                            blocks.pop(rem_idx2)
                            block_set.discard(rem_block2)
                            
                            if len(blocks) < best_len:
                                best_len = len(blocks)
                                best_blocks = [list(b) for b in blocks]
                else:
                    # Revert removal
                    for sub_id in get_block_subsets(rem_block):
                        coverage[sub_id] += 1
                
                T *= cooling_rate
        
        return best_blocks
    
    # Run multiple greedy constructions and then simulated annealing
    runs = 0
    while time.time() < deadline - 20.0:
        runs += 1
        blocks = greedy_construct()
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        # Run simulated annealing if we have time
        if time.time() < deadline - 10.0:
            sa_blocks = simulated_annealing(blocks, deadline - 5.0)
            if len(sa_blocks) < best_len:
                best_len = len(sa_blocks)
                best_blocks = [list(b) for b in sa_blocks]
    
    return best_blocks
