# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large square-difference-free set modulo 205.
    
    Strategy:
    1. Exploit CRT structure: Z_205 = Z_5 x Z_41.
    2. A difference (a-b) is a square mod 205 iff (a1-b1) is square mod 5 AND (a2-b2) is square mod 41.
       Note: 0 is a square, but we exclude 0 from forbidden differences.
       So (a-b) is forbidden iff:
         (a1-b1) mod 5 is in S5_nonzero OR (a1-b1) mod 5 == 0 AND (a2-b2) mod 41 in S41_nonzero?
         No, (a-b) mod 205 is a square iff (a1-b1) mod 5 is a square in Z5 AND (a2-b2) mod 41 is a square in Z41.
         The set of squares mod 5 is {0, 1, 4}.
         The set of squares mod 41 is {0, 1, 4, 5, 9, 16, 20, 25, 33, 36, 39, 40}. (12 elements)
         Forbidden if (d1 in S5 AND d2 in S41) AND (d1, d2) != (0, 0).
         So (a, b) are incompatible if:
         d1 = (a1-b1)%5, d2 = (a2-b2)%41.
         Incompatible if (d1 in S5 and d2 in S41) and (d1 != 0 or d2 != 0).
         
    3. We can model this as a Maximum Independent Set on a graph of 205 nodes, but the structure allows
       smarter search. However, 205 is small enough that a well-tuned SA or Branch and Bound might work.
       Given the time limit (600s), and that the previous SA got 1.3333 (which implies ~133 elements? No, 
       score is value/seed. Seed is likely a greedy baseline. If seed is ~100, 1.3333 is ~133.
       Wait, max size is 205. 
       Let's check the density. 
       Squares mod 5: 0, 1, 4. (3/5)
       Squares mod 41: 12/41.
       Probability a random difference is forbidden:
       P(forbidden) = P(d1 in S5, d2 in S41) - P(d1=0, d2=0)
       = (3/5 * 12/41) - (1/5 * 1/41)
       = 36/205 - 1/205 = 35/205 = 7/41 ≈ 0.17.
       So the graph is relatively sparse. Average degree ~ 0.17 * 204 ≈ 35.
       Max Independent Set in a graph with 205 nodes and degree 35 is likely large, maybe 100-150?
       
    4. Let's use a hybrid approach:
       - Precompute the conflict graph.
       - Use a strong local search (SA) with multiple restarts.
       - Add a "repair" step: if a move creates a conflict, resolve it by removing the conflicting node? 
         No, standard SA on independent set usually moves between valid states.
       - To improve the previous SA:
         a) Better initialization: Greedy with random ordering, but also try "conflict-aware" greedy.
         b) Better moves: 
            - Add a random node (if valid).
            - Remove a random node.
            - Swap: Remove u, Add v (if valid after removing u).
            - 2-swap: Remove u1, u2; Add v1, v2.
         c) Use a "plateau" breaking mechanism.
         d) Crucially: The previous code had a bug or inefficiency? 
            "can_add" checks `np.any(conflict[node, :] & mask)`. This is O(N) per check.
            With N=205, this is fast.
            But `conflict` matrix is 205x205 bools.
            
    5. Let's try a different heuristic: 
       Since the graph is defined by CRT, we can try to construct the set by iterating over residues mod 5 and mod 41.
       However, the interaction is global.
       
    6. Let's stick to SA but optimize it:
       - Use bitsets for masks? Python ints are fast.
       - `mask` as an integer with 205 bits.
       - `conflict[i]` as an integer bitmask of nodes conflicting with i.
       - `can_add(node, mask)`: `(conflict[node] & mask) == 0`.
       - This is much faster than numpy array operations.
       
    7. Algorithm:
       - Precompute `conflict_mask[i]` for i in 0..204.
       - Represent set as an integer `S`.
       - `size = bin(S).count('1')`.
       - Moves:
         - Add: pick random i not in S. If `(conflict_mask[i] & S) == 0`, add.
         - Remove: pick random i in S. Remove.
         - Swap: pick random i in S, j not in S. If `(conflict_mask[j] & (S ^ (1<<i))) == 0`, swap.
       
    8. To beat 1.3333, we need a larger set.
       Let's run SA for a long time with many restarts.
       Also, try a "greedy repair" from a random set?
       
    9. Let's implement the bitset SA.
    """
    m = 205
    deadline = time.time() + 590.0  # Leave 10s buffer

    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add((x * x) % m)
    forbidden.discard(0)
    
    # 2. Precompute conflict masks
    # conflict_mask[i] is an integer where bit j is 1 if (i-j)%m is in forbidden
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if (i - j) % m in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    # Helper: check if node can be added to set S (integer bitmask)
    def can_add(node, S):
        return (conflict_mask[node] & S) == 0

    # Helper: get list of selected nodes from bitmask
    def get_selected(S):
        return [i for i in range(m) if (S >> i) & 1]
    
    # Helper: get list of unselected nodes from bitmask
    def get_unselected(S):
        return [i for i in range(m) if not ((S >> i) & 1)]

    # 3. Greedy Seed Generator
    def greedy_seed():
        order = list(range(m))
        random.shuffle(order)
        S = 0
        for node in order:
            if can_add(node, S):
                S |= (1 << node)
        return S

    # 4. Simulated Annealing with Bitsets
    def simulated_annealing(start_S, time_budget):
        current_S = start_S
        current_size = bin(current_S).count('1')
        best_S = current_S
        best_size = current_size
        
        T_start = 1.0
        T_end = 0.01
        cooling_rate = 0.999
        T = T_start
        
        end_time = time.time() + time_budget
        
        # Precompute lists for faster access? 
        # For 205 nodes, list comprehension is fine.
        
        while time.time() < end_time:
            move_type = random.random()
            new_S = current_S
            new_size = current_size
            
            if move_type < 0.4:
                # Add
                unselected = get_unselected(current_S)
                if unselected:
                    node = random.choice(unselected)
                    if can_add(node, current_S):
                        new_S |= (1 << node)
                        new_size += 1
                    else:
                        continue
            elif move_type < 0.7:
                # Remove
                selected = get_selected(current_S)
                if selected:
                    node = random.choice(selected)
                    new_S &= ~(1 << node)
                    new_size -= 1
            else:
                # Swap
                selected = get_selected(current_S)
                unselected = get_unselected(current_S)
                if selected and unselected:
                    rem_node = random.choice(selected)
                    add_node = random.choice(unselected)
                    
                    # Check if add_node is compatible with (current_S - rem_node)
                    temp_S = current_S & ~(1 << rem_node)
                    if can_add(add_node, temp_S):
                        new_S = temp_S | (1 << add_node)
                        # Size stays the same
                    else:
                        continue
            
            delta = new_size - current_size
            
            if delta > 0:
                current_S = new_S
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_S = current_S
            else:
                # Accept with probability exp(delta / T)
                # delta is 0 or negative
                if delta == 0:
                    prob = 1.0
                else:
                    prob = np.exp(delta / T)
                
                if random.random() < prob:
                    current_S = new_S
                    current_size = new_size
            
            T *= cooling_rate
            if T < T_end:
                T = T_end
            
        return best_S, best_size

    # 5. Main Search Loop
    best_overall_S = 0
    best_overall_size = 0
    
    # Initial greedy seed
    init_S = greedy_seed()
    best_overall_S = init_S
    best_overall_size = bin(init_S).count('1')
    
    search_deadline = time.time() + 580.0
    
    # We will run multiple SA instances.
    # To improve diversity, we can also try "reverse greedy" or other seeds.
    
    while time.time() < search_deadline:
        # Generate start state
        r = random.random()
        if r < 0.5:
            start_S = greedy_seed()
        elif r < 0.8:
            # Perturb best
            start_S = best_overall_S
            selected = get_selected(start_S)
            if len(selected) > 5:
                for _ in range(5):
                    node = random.choice(selected)
                    start_S &= ~(1 << node)
                    selected.remove(node)
        else:
            # Random sparse start
            start_S = 0
            for i in range(m):
                if random.random() < 0.3:
                    if can_add(i, start_S):
                        start_S |= (1 << i)
        
        # Time slice
        slice_time = 5.0
        if time.time() + slice_time > search_deadline:
            slice_time = max(0.5, search_deadline - time.time())
        
        result_S, result_size = simulated_annealing(start_S, slice_time)
        
        if result_size > best_overall_size:
            best_overall_size = result_size
            best_overall_S = result_S

    # Extract final set
    final_set = get_selected(best_overall_S)
    
    # Final validation
    valid = True
    for i in range(len(final_set)):
        for j in range(i+1, len(final_set)):
            a = final_set[i]
            b = final_set[j]
            if (a - b) % m in forbidden or (b - a) % m in forbidden:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return np.array([0], dtype=np.int64)

    return np.array(final_set, dtype=np.int64)

# EVOLVE-BLOCK-END
