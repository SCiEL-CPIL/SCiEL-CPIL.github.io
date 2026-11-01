import numpy as np
import itertools
import time
import random

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    """
    D = 11
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Baseline: All vectors with exactly two non-zero entries (+/-1).
    # Norm^2 = 2. Max norm = sqrt(2).
    # Min distance between distinct such vectors is sqrt(2).
    # So all 220 are valid.
    candidates_2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates_2.append(v)
    
    baseline = np.array(candidates_2, dtype=np.int64)
    best_set = baseline
    best_size = len(baseline)
    
    # Try to find a larger set from Norm^2=4 vectors.
    # If we use only Norm^2=4, max norm = 2. Need dist^2 >= 4.
    
    candidates_4 = []
    # Type A: One entry +/-2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates_4.append(v)
            
    # Type B: Four entries +/-1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates_4.append(v)
            
    candidates_4 = np.array(candidates_4, dtype=np.int64)
    n4 = len(candidates_4)
    
    if time.time() > deadline - 20:
        return best_set
    
    # Compute conflict matrix for candidates_4
    # dist^2 = n_i + n_j - 2 dot
    # All norms are 4. dist^2 = 8 - 2 dot.
    # Conflict if dist^2 < 4 => 8 - 2 dot < 4 => 2 dot > 4 => dot > 2.
    # dot is integer. So conflict if dot >= 3.
    
    dots = candidates_4 @ candidates_4.T
    conflict = (dots >= 3)
    np.fill_diagonal(conflict, False)
    
    # Precompute neighbor lists for efficiency
    # Using a list of lists of indices
    neighbor_lists = [np.where(conflict[i])[0] for i in range(n4)]
    
    # Heuristic Search for Maximum Independent Set
    best_4_set = []
    best_4_size = 0
    
    # Number of restarts limited by time
    max_restarts = 1000
    np.random.seed(42)
    
    for attempt in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # 1. Greedy Initialization
        # Randomly order nodes
        order = np.random.permutation(n4)
        remaining = np.ones(n4, dtype=bool)
        selected = []
        
        for node in order:
            if not remaining[node]:
                continue
            # Check if node conflicts with any already selected
            # Since we remove neighbors from 'remaining' when we select, 
            # if remaining[node] is True, it means it doesn't conflict with any selected.
            selected.append(node)
            # Remove node and its neighbors from remaining
            to_remove = conflict[node].copy()
            to_remove[node] = True
            remaining[to_remove] = False
            
        current_size = len(selected)
        
        # 2. Local Improvement (Hill Climbing)
        # Try to swap: remove one, add another
        improved = True
        max_iterations = 50
        iter_count = 0
        while improved and iter_count < max_iterations:
            if time.time() > deadline - 5:
                break
            improved = False
            iter_count += 1
            
            selected_set = set(selected)
            
            # Try removing each node
            for i in range(len(selected)):
                if time.time() > deadline - 5:
                    break
                node_to_remove = selected[i]
                
                # Find a candidate to add
                # A candidate must not conflict with any node in selected \ {node_to_remove}
                # It CAN conflict with node_to_remove (since we remove it)
                
                # Candidates are those not in selected_set
                # We can iterate through all nodes not in selected
                # Optimization: Only consider nodes that conflict with node_to_remove?
                # Not necessarily, but nodes that don't conflict with node_to_remove 
                # and don't conflict with others would have been added already?
                # No, greedy might have skipped them.
                
                # Let's iterate through all nodes not in selected
                # To speed up, we can precompute "compatible with selected minus node_to_remove"
                
                best_candidate = -1
                # We want to add a node that allows us to keep the size or increase it.
                # Since we remove 1 and add 1, size stays same.
                # To increase size, we need to remove 1 and add 2? 
                # Or remove 0 and add 1?
                # The "Add" step is: find a node not in selected that doesn't conflict with ANY selected.
                # If found, add it.
                
                # First, try to just add a node
                added = False
                for node in range(n4):
                    if node in selected_set:
                        continue
                    # Check if node conflicts with any in selected
                    # If conflict[node] & selected_mask is empty
                    if not np.any(conflict[node][selected]):
                        selected.append(node)
                        selected_set.add(node)
                        improved = True
                        added = True
                        break
                
                if added:
                    continue # Restart loop with new size
                
                # If no node could be added, try a swap
                # Remove node_to_remove, then try to add a node that conflicts with node_to_remove
                # but not with others.
                
                # Actually, a better swap:
                # Remove node_to_remove.
                # Now find a node X not in selected (excluding node_to_remove) 
                # such that X does not conflict with any node in (selected \ {node_to_remove}).
                
                temp_selected = [s for s in selected if s != node_to_remove]
                temp_selected_set = set(temp_selected)
                
                for node in range(n4):
                    if node in temp_selected_set:
                        continue
                    if node == node_to_remove:
                        continue
                    
                    # Check compatibility with temp_selected
                    if not np.any(conflict[node][temp_selected]):
                        # We can replace node_to_remove with node
                        # Size remains same. This doesn't increase size.
                        # But it might enable future additions.
                        # For now, let's just look for size increase.
                        pass
                        
                # If we can't increase size by 1, can we increase by more?
                # Remove node_to_remove.
                # Then greedily add as many as possible.
                # If count > 1, accept.
                
                # This is expensive. Let's stick to simple add/swap for now.
                # The "Add" phase above handles size +1.
                # The "Swap" phase above is size 0 change.
                
                # To get size +1 via swap:
                # Remove node_to_remove.
                # Add node X (compatible with rest).
                # Then try to add node Y (compatible with rest + X).
                # If Y exists, size increases by 1.
                
                # Let's implement this 2-step swap.
                temp_selected = [s for s in selected if s != node_to_remove]
                temp_selected_set = set(temp_selected)
                
                for node_x in range(n4):
                    if node_x in temp_selected_set or node_x == node_to_remove:
                        continue
                    if not np.any(conflict[node_x][temp_selected]):
                        # Node X is a valid replacement
                        # Now try to add Node Y
                        temp_selected_with_x = temp_selected + [node_x]
                        temp_selected_set_with_x = temp_selected_set | {node_x}
                        
                        for node_y in range(n4):
                            if node_y in temp_selected_set_with_x:
                                continue
                            if not np.any(conflict[node_y][temp_selected_with_x]):
                                # We can add Y!
                                # New set: temp_selected + [node_x, node_y]
                                # Size increases by 1
                                selected = temp_selected + [node_x, node_y]
                                selected_set = set(selected)
                                improved = True
                                break
                        if improved:
                            break
                            
                if improved:
                    break # Restart local search loop with new set

        if len(selected) > best_4_size:
            best_4_size = len(selected)
            best_4_set = selected.copy()
            
        # Periodic check
        if attempt % 100 == 0:
            pass

    if best_4_size > best_size:
        best_set = candidates_4[best_4_set]
        best_size = best_4_size
        
    return best_set
