import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    block_to_tsubs = []
    for block in k_subsets:
        covered_tsubs = []
        for tsub in itertools.combinations(block, t):
            idx = t_subsets_set[tsub]
            covered_tsubs.append(idx)
        block_to_tsubs.append(covered_tsubs)
        
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
            
    # Convert to sets for faster membership? Or keep lists.
    # For "is in exclude_set", a set is better.
    # But tsub_to_blocks lists are small.
    
    def run_greedy(seed):
        random.seed(seed)
        gains = np.full(num_blocks, len(block_to_tsubs[0]), dtype=np.int32)
        uncovered = np.ones(num_tsubs, dtype=bool)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
            # Find max gain
            # To allow random tie-breaking, we can find all max gains and pick one randomly
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            b = random.choice(candidates.tolist())
            
            selected.append(b)
            for s in block_to_tsubs[b]:
                if uncovered[s]:
                    uncovered[s] = False
                    gains[tsub_to_blocks[s]] -= 1
        return selected

    def minimize(selected):
        # Returns minimized selected list
        if time.time() > deadline:
            return selected
            
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Delete Redundants
            to_remove = []
            for b in selected:
                redundant = True
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)
            
            if to_remove:
                for b in to_remove:
                    selected.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue # Restart loop
            
            # 2. 1-Swap
            # Identify unique t-subsets for each block
            unique_tsubs_per_block = {}
            for b in selected:
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_list.append(s)
                unique_tsubs_per_block[b] = u_list
            
            # Try swaps
            swap_made = False
            # Shuffle blocks to randomize order
            blocks_list = list(selected)
            random.shuffle(blocks_list)
            
            for b_out in blocks_list:
                if time.time() > deadline:
                    break
                u_list = unique_tsubs_per_block.get(b_out, [])
                if not u_list:
                    continue # No unique t-subsets, can't swap out directly without breaking cover
                
                # Try to find a b_in that covers all u_list
                # Candidates must cover u_list[0]
                candidates_in = tsub_to_blocks[u_list[0]]
                random.shuffle(candidates_in)
                
                for b_in in candidates_in:
                    if b_in in selected:
                        continue
                    
                    # Check if b_in covers all other u_list
                    valid = True
                    for s in u_list[1:]:
                        if b_in not in tsub_to_blocks[s]:
                            valid = False
                            break
                            
                    if valid:
                        # Perform swap
                        selected.remove(b_out)
                        selected.append(b_in)
                        
                        for s in block_to_tsubs[b_out]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[b_in]:
                            cover_count[s] += 1
                            
                        swap_made = True
                        improved = True
                        break # Restart minimize loop after one swap
                        
                if swap_made:
                    break
            
            if swap_made:
                continue
                
            # 3. Add-Minimize (Limited)
            # Find all unique t-subsets in the current solution
            all_unique_tsubs = []
            for b in selected:
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        all_unique_tsubs.append(s)
                        
            if not all_unique_tsubs:
                break # No unique t-subsets, cannot improve via add-minimize? 
                # Actually, if no unique t-subsets, all are covered >=2 times.
                # Adding a block won't make anything redundant unless it covers 
                # t-subsets that are covered exactly once? No, if all are >=2, 
                # adding a block makes them >=3. It won't free up any block 
                # unless the block we add allows us to remove a block that 
                # was covering something unique? But there are no unique ones.
                # So Add-Minimize is useless if no unique t-subsets.
                
            # Pick a random unique t-subset and try to add a block covering it
            random.shuffle(all_unique_tsubs)
            
            add_made = False
            for s_target in all_unique_tsubs:
                if time.time() > deadline:
                    break
                candidates_in = tsub_to_blocks[s_target]
                random.shuffle(candidates_in)
                
                for b_in in candidates_in:
                    if b_in in selected:
                        continue
                    
                    # Tentatively add b_in
                    selected.append(b_in)
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                        
                    # Delete redundants
                    to_remove = []
                    for b in selected:
                        redundant = True
                        for s in block_to_tsubs[b]:
                            if cover_count[s] == 1:
                                redundant = False
                                break
                        if redundant:
                            to_remove.append(b)
                            
                    # Check if we improved (size decreased)
                    # We added 1, removed len(to_remove)
                    if len(to_remove) > 1:
                        # Accept
                        for b in to_remove:
                            selected.remove(b)
                            for s in block_to_tsubs[b]:
                                cover_count[s] -= 1
                        add_made = True
                        improved = True
                        break # Restart minimize loop
                    else:
                        # Revert
                        selected.remove(b_in)
                        for s in block_to_tsubs[b_in]:
                            cover_count[s] -= 1
                            
                if add_made:
                    break
                    
        return selected

    best_solution = None
    best_size = float('inf')
    
    # Run multiple greedy restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 5: # Leave time for minimization
            break
            
        sol = run_greedy(i)
        if not sol:
            continue
            
        # Quick check: is it valid? (Greedy should be valid)
        # Minimize
        min_sol = minimize(sol)
        
        if len(min_sol) < best_size:
            best_size = len(min_sol)
            best_solution = min_sol
            
        # If we hit the lower bound or a very good size, we might stop early?
        # LB is 131. If we get 131, we can't do better (integer).
        if best_size <= 131:
            break

    if best_solution is None:
        # Fallback: single greedy
        best_solution = run_greedy(0)
        best_solution = minimize(best_solution)

    return [list(k_subsets[i]) for i in best_solution]
