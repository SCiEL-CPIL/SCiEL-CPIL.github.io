import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Precompute: for each block, which t-subsets it covers
    block_to_tsubs = []
    for block in k_subsets:
        covered_tsubs = []
        for tsub in itertools.combinations(block, t):
            idx = t_subsets_set[tsub]
            covered_tsubs.append(idx)
        block_to_tsubs.append(covered_tsubs)
        
    # Precompute: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
            
    # Convert to numpy arrays
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Greedy Construction
    def greedy_construct():
        gains = np.full(num_blocks, len(block_to_tsubs[0]), dtype=np.int32)
        uncovered = np.ones(num_tsubs, dtype=bool)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
            b = np.argmax(gains)
            if gains[b] == 0:
                break
            selected.append(b)
            for s in block_to_tsubs[b]:
                if uncovered[s]:
                    uncovered[s] = False
                    gains[tsub_to_blocks_np[s]] -= 1
        return selected, uncovered
    
    # Local optimization
    def local_optimize(selected):
        if not selected:
            return selected
        
        # Build cover count
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        # Iteratively remove redundant blocks and try swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Remove redundant blocks
            new_selected = []
            for b in selected:
                # Check if block is redundant
                redundant = True
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_selected.append(b)
                else:
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                    improved = True
            
            selected = new_selected
            
            if not selected:
                break
            
            # Try swaps: find block with fewest unique t-subsets
            unique_counts = {}
            for b in selected:
                u_count = 0
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_count += 1
                unique_counts[b] = u_count
            
            if not unique_counts:
                break
                
            min_unique = min(unique_counts.values())
            
            if min_unique == 0:
                # A block has no unique t-subsets, it's redundant
                # But we should have caught that above
                continue
                
            # Try to swap a block with min unique
            candidates = [b for b, c in unique_counts.items() if c == min_unique]
            random.shuffle(candidates)
            
            for b_out in candidates:
                # Get unique t-subsets of b_out
                uniq_tsubs = [s for s in block_to_tsubs[b_out] if cover_count[s] == 1]
                
                # Find a block that covers all uniq_tsubs
                if not uniq_tsubs:
                    continue
                    
                # Get candidates from first unique t-subset
                first_s = uniq_tsubs[0]
                candidates_in = tsub_to_blocks_np[first_s]
                
                # Filter: must not be in selected, must cover all uniq_tsubs
                selected_set = set(selected)
                best_in = None
                best_gain = -1
                
                for c in candidates_in:
                    if c in selected_set:
                        continue
                    # Check if c covers all uniq_tsubs
                    covers_all = True
                    for s in uniq_tsubs:
                        if c not in tsub_to_blocks[s]:
                            covers_all = False
                            break
                    if covers_all:
                        # Calculate gain: how many new t-subsets does c cover?
                        new_cover = 0
                        for s in block_to_tsubs[c]:
                            if cover_count[s] == 0:
                                new_cover += 1
                        if new_cover > best_gain:
                            best_gain = new_cover
                            best_in = c
                
                if best_in is not None:
                    # Perform swap
                    selected.remove(b_out)
                    selected.append(best_in)
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[best_in]:
                        cover_count[s] += 1
                    improved = True
                    break  # Restart loop after one swap
        
        return selected
    
    # Multi-start with different random seeds for tie-breaking
    best_solution = None
    best_len = float('inf')
    
    # First, do a standard greedy
    selected, uncovered = greedy_construct()
    if not uncovered.any():
        optimized = local_optimize(selected)
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_solution = optimized
    
    # Try a few more runs with randomized tie-breaking
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Randomized greedy
        gains = np.full(num_blocks, len(block_to_tsubs[0]), dtype=np.int32)
        uncovered = np.ones(num_tsubs, dtype=bool)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
            # Find all blocks with max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            # Randomly pick from blocks with max gain
            max_blocks = np.where(gains == max_gain)[0]
            b = random.choice(max_blocks.tolist())
            selected.append(b)
            for s in block_to_tsubs[b]:
                if uncovered[s]:
                    uncovered[s] = False
                    gains[tsub_to_blocks_np[s]] -= 1
        
        if not uncovered.any():
            optimized = local_optimize(selected)
            if len(optimized) < best_len:
                best_len = len(optimized)
                best_solution = optimized
    
    # Return the best solution found
    if best_solution is None:
        # Fallback: return greedy result
        return [list(k_subsets[i]) for i in selected]
    
    return [list(k_subsets[i]) for i in best_solution]
