"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_subsets = 1 << n
    
    # Precompute subset masks as numpy array
    all_masks = np.arange(total_subsets, dtype=np.int32)
    
    # Precompute comparability matrix
    # comp[i, j] is 1 if i and j are comparable (strict subset/superset), 0 otherwise
    comp = np.zeros((total_subsets, total_subsets), dtype=np.int8)
    
    # Efficiently compute comparability
    for i in range(total_subsets):
        is_subset = (all_masks & i) == i
        is_superset = (all_masks & i) == all_masks
        comparable = is_subset | is_superset
        comparable[i] = 0 # Exclude self
        comp[i] = comparable.astype(np.int8)

    best_result = None
    best_score = -1
    
    # Precompute masks by size for better initialization
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(total_subsets):
        size = bin(mask).count('1')
        masks_by_size[size].append(mask)

    # Try multiple random restarts
    num_restarts = 50
    for trial in range(num_restarts):
        random.seed(trial + 10000)
        selected_list = []
        selected_set = set()
        
        # Greedy Phase
        # Start with a random node, biased towards middle sizes
        start_size = random.choice([5, 6, 7])
        start = random.choice(masks_by_size[start_size])
        selected_list.append(start)
        selected_set.add(start)
        
        for _ in range(m - 1):
            scores = comp[:, selected_list].sum(axis=1)
            
            # Exclude selected
            for sel in selected_set:
                scores[sel] = -1
            
            best_idx = int(np.argmax(scores))
            selected_list.append(best_idx)
            selected_set.add(best_idx)
        
        # Calculate current score
        sub_matrix = comp[selected_list][:, selected_list]
        current_score = int(sub_matrix.sum() // 2)
        
        # Local Search Phase
        improved = True
        max_ls_iters = 50
        ls_iter = 0
        
        while improved and ls_iter < max_ls_iters:
            improved = False
            ls_iter += 1
            
            # Precompute external scores: connections from each node to selected set
            ext_scores = comp[:, selected_list].sum(axis=1)
            
            # Precompute internal degrees for selected nodes
            sub_matrix = comp[selected_list][:, selected_list]
            int_deg = sub_matrix.sum(axis=1)
            
            best_net_gain = 0
            best_swap_i = -1
            best_swap_j = -1
            
            # Create a mask for selected nodes to easily exclude them
            sel_mask = np.zeros(total_subsets, dtype=bool)
            for sel in selected_list:
                sel_mask[sel] = True
            
            # Iterate over each selected node to find the best swap
            for i in range(m):
                loss = int_deg[i]
                
                # Gains for all nodes j if we remove selected_list[i] and add j
                gains = ext_scores - comp[selected_list[i], :]
                gains[sel_mask] = -1
                
                best_j = int(np.argmax(gains))
                best_gain = int(gains[best_j])
                
                net_gain = best_gain - loss
                
                if net_gain > best_net_gain:
                    best_net_gain = net_gain
                    best_swap_i = i
                    best_swap_j = best_j
            
            if best_net_gain > 0:
                # Perform the swap
                old_node = selected_list[best_swap_i]
                selected_list[best_swap_i] = best_swap_j
                selected_set.remove(old_node)
                selected_set.add(best_swap_j)
                current_score += best_net_gain
                improved = True
        
        if current_score > best_score:
            best_score = current_score
            best_result = selected_list.copy()
            
    # Convert selected masks to lists of integers
    result = []
    for mask in best_result:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result

# EVOLVE-BLOCK-END
