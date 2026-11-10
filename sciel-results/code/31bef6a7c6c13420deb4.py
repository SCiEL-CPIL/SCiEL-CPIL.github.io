# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large square-difference-free set modulo 205.
    
    Uses Simulated Annealing with bitmask representations for fast conflict checking.
    The graph has 205 nodes. Edges exist if the difference is a non-zero square mod 205.
    We seek the Maximum Independent Set.
    """
    m = 205
    deadline = time.time() + 590.0  # Leave 10s buffer for 600s limit (evaluator kills at 630s)
    
    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add((x * x) % m)
    forbidden.discard(0)
    
    # 2. Precompute conflict masks
    # conflict_mask[i] is an integer bitmask where bit j is 1 if node i conflicts with node j
    # i conflicts with j if (i - j) % m in forbidden
    conflict_masks = [0] * m
    for i in range(m):
        mask_val = 0
        for j in range(m):
            if i == j:
                continue
            if (i - j) % m in forbidden:
                mask_val |= (1 << j)
        conflict_masks[i] = mask_val
    
    # Helper to get list of nodes in mask
    def mask_to_list(mask):
        nodes = []
        for i in range(m):
            if mask & (1 << i):
                nodes.append(i)
        return nodes

    # 3. Greedy Seed Generator
    def greedy_seed():
        # Order nodes by degree (number of conflicts) ascending? 
        # Or just random shuffle?
        # Let's try sorting by degree to pick "isolated" nodes first.
        degrees = [bin(conflict_masks[i]).count('1') for i in range(m)]
        order = list(range(m))
        # Randomize ties to ensure diversity
        random.shuffle(order)
        # Stable sort by degree
        order.sort(key=lambda x: degrees[x])
        
        mask = 0
        for node in order:
            if (mask & conflict_masks[node]) == 0:
                mask |= (1 << node)
        return mask

    # 4. Simulated Annealing with Bitmasks
    def simulated_annealing(start_mask, time_budget):
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        
        # Temperature parameters
        T_start = 2.0
        T_end = 0.01
        cooling_rate = 0.9995
        T = T_start
        
        end_time = time.time() + time_budget
        
        # Precompute list of all nodes for random selection
        all_nodes = list(range(m))
        
        while time.time() < end_time:
            # Generate a move
            # Types: 0: Add, 1: Remove, 2: Swap, 3: 2-Swap
            move_type = random.random()
            
            new_mask = current_mask
            new_size = current_size
            valid_move = False
            
            if move_type < 0.3:
                # Add a random node
                # Find nodes not in mask
                # To speed up, we can pick a random node and check if it's in mask
                # Or maintain a list of available nodes? Maintaining is O(N) update.
                # Let's just pick random node.
                node = random.choice(all_nodes)
                if not (current_mask & (1 << node)):
                    if (current_mask & conflict_masks[node]) == 0:
                        new_mask = current_mask | (1 << node)
                        new_size = current_size + 1
                        valid_move = True
            elif move_type < 0.5:
                # Remove a random node
                if current_size > 0:
                    # Pick a random set bit
                    # Efficient way: pick random int in [0, current_size), then find k-th set bit?
                    # Or just pick random node from 0..204 and check if set.
                    node = random.choice(all_nodes)
                    if (current_mask & (1 << node)):
                        new_mask = current_mask & ~(1 << node)
                        new_size = current_size - 1
                        valid_move = True
            elif move_type < 0.8:
                # Swap: Remove one, Add one
                # Pick random node to remove (must be in mask)
                rem_node = random.choice(all_nodes)
                if (current_mask & (1 << rem_node)):
                    # Pick random node to add (must not be in mask)
                    add_node = random.choice(all_nodes)
                    if not (current_mask & (1 << add_node)):
                        # Check if add_node is compatible with (mask - rem_node)
                        temp_mask = current_mask & ~(1 << rem_node)
                        if (temp_mask & conflict_masks[add_node]) == 0:
                            new_mask = temp_mask | (1 << add_node)
                            new_size = current_size # Size unchanged
                            valid_move = True
            else:
                # 2-Swap: Remove two, Add two
                # This is expensive to sample efficiently, so we do it less often
                # Pick two distinct nodes to remove
                rem_nodes = set()
                while len(rem_nodes) < 2 and current_size >= 2:
                    node = random.choice(all_nodes)
                    if (current_mask & (1 << node)):
                        rem_nodes.add(node)
                
                if len(rem_nodes) == 2:
                    rem_list = list(rem_nodes)
                    # Pick two distinct nodes to add
                    add_nodes = set()
                    while len(add_nodes) < 2:
                        node = random.choice(all_nodes)
                        # Must not be in current mask and not be the nodes we are removing
                        if not (current_mask & (1 << node)) and node not in rem_nodes:
                            add_nodes.add(node)
                    
                    if len(add_nodes) == 2:
                        add_list = list(add_nodes)
                        a1, a2 = add_list
                        r1, r2 = rem_list
                        
                        # New mask is current - r1 - r2 + a1 + a2
                        temp_mask = current_mask
                        temp_mask &= ~(1 << r1)
                        temp_mask &= ~(1 << r2)
                        
                        # Check a1 and a2 against temp_mask and each other
                        if (temp_mask & conflict_masks[a1]) == 0 and \
                           (temp_mask & conflict_masks[a2]) == 0 and \
                           (conflict_masks[a1] & (1 << a2)) == 0:
                            new_mask = temp_mask | (1 << a1) | (1 << a2)
                            new_size = current_size # Size unchanged
                            valid_move = True
            
            if not valid_move:
                # If the move was invalid, we could skip or just cool down.
                # Skipping is better to keep the state consistent.
                T *= cooling_rate
                if T < T_end:
                    T = T_end
                continue

            # Evaluate move
            delta = new_size - current_size
            
            if delta > 0:
                # Always accept improvement
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            else:
                # Accept with probability exp(delta / T)
                # delta is <= 0
                if T > 0:
                    prob = np.exp(delta / T)
                    if random.random() < prob:
                        current_mask = new_mask
                        current_size = new_size
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end

        return best_mask, best_size

    # 5. Main Search Loop
    best_overall_mask = 0
    best_overall_size = 0
    
    # Initial greedy seed
    init_mask = greedy_seed()
    best_overall_mask = init_mask
    best_overall_size = bin(init_mask).count('1')
    
    search_deadline = time.time() + 580.0
    
    # Run multiple SA restarts
    # We allocate time slices to each run
    # Shorter slices allow more diversity, but SA needs some time to converge.
    # Let's try slices of 5-10 seconds.
    
    while time.time() < search_deadline:
        # Generate start mask
        # 50% new greedy, 50% perturb best
        if random.random() < 0.5:
            start_mask = greedy_seed()
        else:
            start_mask = best_overall_mask
            # Remove 5-10 random elements to perturb
            nodes_in_best = mask_to_list(start_mask)
            if len(nodes_in_best) > 10:
                nodes_to_remove = random.sample(nodes_in_best, min(10, len(nodes_in_best)))
                for node in nodes_to_remove:
                    start_mask &= ~(1 << node)
        
        # Determine time slice
        slice_time = 10.0
        if time.time() + slice_time > search_deadline:
            slice_time = max(1.0, search_deadline - time.time())
        
        result_mask, result_size = simulated_annealing(start_mask, slice_time)
        
        if result_size > best_overall_size:
            best_overall_size = result_size
            best_overall_mask = result_mask
        
        # Check deadline
        if time.time() > search_deadline:
            break

    # Extract final set
    final_set = mask_to_list(best_overall_mask)
    
    # Final validation
    valid = True
    F_set = forbidden
    for i in range(len(final_set)):
        for j in range(i+1, len(final_set)):
            a = final_set[i]
            b = final_set[j]
            if (a - b) % m in F_set or (b - a) % m in F_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        return np.array([0], dtype=np.int64)

    return np.array(final_set, dtype=np.int64)

# EVOLVE-BLOCK-END
