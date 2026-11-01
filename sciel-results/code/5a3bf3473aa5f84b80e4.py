import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    block_to_tsubs = []
    block_to_tsubs_set = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))
        
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    # Convert to sets for faster intersection
    tsub_to_blocks_set = [set(lst) for lst in tsub_to_blocks]
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]

    def greedy_construction(randomize_ties=False, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = np.ones(num_tsubs, dtype=bool)
        gains = np.full(num_blocks, len(block_to_tsubs[0]), dtype=np.int32)
        selected_block_indices = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None
            
            if randomize_ties and random.random() < 0.2:
                top_k = 10
                if num_blocks > top_k:
                    top_indices = np.argpartition(gains, -top_k)[-top_k:]
                    b = random.choice(top_indices)
                else:
                    b = np.argmax(gains)
            else:
                b = np.argmax(gains)
            
            if gains[b] == 0:
                break
            
            selected_block_indices.append(b)
            
            for s_idx in block_to_tsubs[b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    gains[tsub_to_blocks_np[s_idx]] -= 1
                    
        if uncovered.any():
            return None
        return selected_block_indices

    def local_optimize(blocks):
        if not blocks:
            return []
            
        selected_set = set(blocks)
        selected_list = list(blocks)
        
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        def get_unique_tsubs(b_idx):
            return [s for s in block_to_tsubs[b_idx] if cover_count[s] == 1]
        
        def find_block_covering(tsubs, exclude_set):
            if not tsubs:
                return None
            # Intersect block sets for all tsubs
            candidates = tsub_to_blocks_set[tsubs[0]]
            for s in tsubs[1:]:
                candidates &= tsub_to_blocks_set[s]
                if not candidates:
                    return None
            candidates -= exclude_set
            if not candidates:
                return None
            return random.choice(list(candidates))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Redundancy Removal
            redundant = []
            for b in selected_list:
                if all(cover_count[s] > 1 for s in block_to_tsubs[b]):
                    redundant.append(b)
            
            if redundant:
                for b in redundant:
                    selected_list.remove(b)
                    selected_set.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # 2. 2-Delete, 1-Add
            # Try to replace two blocks with one
            n = len(selected_list)
            for i in range(n):
                if time.time() > deadline: break
                b1 = selected_list[i]
                u1 = get_unique_tsubs(b1)
                if not u1: continue # Redundant, handled above? No, if not redundant, u1 is not empty?
                # Actually if not redundant, u1 is not empty.
                
                for j in range(i+1, n):
                    b2 = selected_list[j]
                    u2 = get_unique_tsubs(b2)
                    
                    # Union of unique tsubs
                    u_union = list(set(u1) | set(u2))
                    
                    # Find a block covering u_union
                    # Exclude b1, b2
                    c = find_block_covering(u_union, {b1, b2})
                    if c is not None:
                        # Perform swap
                        selected_list.remove(b1)
                        selected_list.remove(b2)
                        selected_set.remove(b1)
                        selected_set.remove(b2)
                        
                        selected_list.append(c)
                        selected_set.add(c)
                        
                        for s in block_to_tsubs[b1]: cover_count[s] -= 1
                        for s in block_to_tsubs[b2]: cover_count[s] -= 1
                        for s in block_to_tsubs[c]: cover_count[s] += 1
                        
                        improved = True
                        break
                if improved: break
            
            if improved: continue
            
            # 3. 1-Delete, 1-Add
            # Sort by unique count to find easy swaps first
            unique_counts = {b: len(get_unique_tsubs(b)) for b in selected_list}
            sorted_blocks = sorted(selected_list, key=lambda x: unique_counts[x])
            
            for b_out in sorted_blocks:
                if time.time() > deadline: break
                u_out = get_unique_tsubs(b_out)
                if not u_out: continue
                
                c_in = find_block_covering(u_out, selected_set)
                if c_in is not None:
                    selected_list.remove(b_out)
                    selected_set.remove(b_out)
                    selected_list.append(c_in)
                    selected_set.add(c_in)
                    
                    for s in block_to_tsubs[b_out]: cover_count[s] -= 1
                    for s in block_to_tsubs[c_in]: cover_count[s] += 1
                    
                    improved = True
                    break
            
            if not improved:
                break
                
        return selected_list

    best_solution = None
    best_len = float('inf')
    
    # Multi-start with ILS
    for i in range(30):
        if time.time() > deadline:
            break
            
        seed = 42 + i
        blocks = greedy_construction(randomize_ties=(i > 0), random_seed=seed)
        
        if blocks is None:
            continue
            
        optimized_blocks = local_optimize(blocks)
        
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
            # If we hit a very good solution, we can try to polish it more
            # by doing more local search passes on the best solution found so far?
            # The local_optimize already loops until no improvement.
        
        # Optional: Perturb best solution and re-optimize
        if i > 5 and best_solution:
            # Perturb: remove 2 random blocks, add 2 random blocks
            perturbed = list(best_solution)
            if len(perturbed) > 2:
                to_remove = random.sample(perturbed, 2)
                for b in to_remove:
                    perturbed.remove(b)
                
                # Add random blocks
                for _ in range(2):
                    b_in = random.choice(range(num_blocks))
                    if b_in not in perturbed:
                        perturbed.append(b_in)
                
                # Fix coverage: Greedy repair
                # Check uncovered
                cover_count = np.zeros(num_tsubs, dtype=np.int32)
                for b in perturbed:
                    for s in block_to_tsubs[b]:
                        cover_count[s] += 1
                
                uncovered = np.where(cover_count == 0)[0]
                while len(uncovered) > 0 and time.time() < deadline:
                    # Greedy add
                    best_b = -1
                    best_gain = 0
                    for b in range(num_blocks):
                        if b in perturbed: continue
                        gain = sum(1 for s in block_to_tsubs[b] if cover_count[s] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                    if best_b == -1 or best_gain == 0: break
                    perturbed.append(best_b)
                    for s in block_to_tsubs[best_b]:
                        cover_count[s] += 1
                    uncovered = np.where(cover_count == 0)[0]
                
                if len(uncovered) == 0:
                    optimized_perturbed = local_optimize(perturbed)
                    if len(optimized_perturbed) < best_len:
                        best_len = len(optimized_perturbed)
                        best_solution = optimized_perturbed

    if best_solution is None:
        return []
        
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result
