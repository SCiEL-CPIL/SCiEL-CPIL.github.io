"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples for fast violation checking
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute 3-AP tuples for counting
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))
    
    # Precompute bitmask representations
    four_ap_bitmasks = []
    for (a, b, c, d_val) in four_ap_tuples:
        mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (d_val-1))
        four_ap_bitmasks.append(mask)
    
    three_ap_bitmasks = []
    for (a, b, c) in three_ap_tuples:
        mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
        three_ap_bitmasks.append(mask)
    
    # Precompute for each element x, which 3-APs and 4-APs it participates in
    elem_to_3aps = {x: [] for x in range(1, N + 1)}
    for i, (a, b, c) in enumerate(three_ap_tuples):
        elem_to_3aps[a].append(i)
        elem_to_3aps[b].append(i)
        elem_to_3aps[c].append(i)
    
    elem_to_4aps = {x: [] for x in range(1, N + 1)}
    for i, (a, b, c, d_val) in enumerate(four_ap_tuples):
        elem_to_4aps[a].append(i)
        elem_to_4aps[b].append(i)
        elem_to_4aps[c].append(i)
        elem_to_4aps[d_val].append(i)
    
    # Build conflict graph: two elements conflict if they are part of a 4-AP
    # Actually, we need: adding x creates a 4-AP if there exist three other selected elements
    # that with x form a 4-AP. So for each x, precompute the set of 4-APs containing x.
    
    def count_3aps_bitmask(selected_mask):
        """Count distinct 3-APs in selected_mask using bit operations."""
        count = 0
        for mask in three_ap_bitmasks:
            if (selected_mask & mask) == mask:
                count += 1
        return count
    
    def can_add_without_4ap_bitmask(selected_mask, x):
        """Check if adding x to selected_mask would create a 4-AP."""
        x_bit = 1 << (x - 1)
        new_mask = selected_mask | x_bit
        for i in elem_to_4aps[x]:
            mask = four_ap_bitmasks[i]
            if (new_mask & mask) == mask:
                return False
        return True
    
    def marginal_3ap_gain(selected_mask, x):
        """Count how many new 3-APs are created by adding x to selected_mask."""
        x_bit = 1 << (x - 1)
        new_3aps = 0
        for i in elem_to_3aps[x]:
            mask = three_ap_bitmasks[i]
            remaining = mask & ~x_bit
            if (selected_mask & remaining) == remaining:
                new_3aps += 1
        return new_3aps
    
    def greedy_build(seed_offset=0):
        """Run weighted greedy to build a solution."""
        rng = random.Random(seed_offset)
        sel_mask = 0
        available = set(range(1, N + 1))
        
        # Pre-compute potential 3-AP contributions for each element
        # An element x can participate in 3-APs where it's first, middle, or last
        # We'll use a dynamic scoring approach
        
        while available and time.time() < deadline - 20.0:
            best_elem = None
            best_score = -1
            
            for x in list(available):
                if not can_add_without_4ap_bitmask(sel_mask, x):
                    continue
                
                gain = marginal_3ap_gain(sel_mask, x)
                
                # Add a small random tiebreaker to diversify restarts
                score = gain * 100 + rng.random()
                
                if score > best_score:
                    best_score = score
                    best_elem = x
            
            if best_elem is None:
                break
            
            sel_mask |= (1 << (best_elem - 1))
            available.remove(best_elem)
        
        return sel_mask
    
    # Phase 1: Multiple greedy restarts with different random seeds
    best_mask = 0
    best_score = 0
    
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        
        mask = greedy_build(seed_offset=r * 1000 + 42)
        score = count_3aps_bitmask(mask)
        
        if score > best_score:
            best_score = score
            best_mask = mask
    
    # Convert to set for local search
    current_solution = set()
    for x in range(1, N + 1):
        if (best_mask >> (x - 1)) & 1:
            current_solution.add(x)
    
    current_score = best_score
    best_solution = set(current_solution)
    best_mask = best_mask
    
    # Phase 2: Local search with swap moves and simulated annealing
    if time.time() < deadline - 3.0:
        T = 30.0
        T_min = 0.05
        cooling_rate = 0.9995
        no_improve_count = 0
        max_no_improve = 3000
        
        while time.time() < deadline - 3.0 and T > T_min and no_improve_count < max_no_improve:
            move_type = random.random()
            
            if move_type < 0.3 and len(current_solution) > 0:
                # Remove a random element
                elem_to_remove = random.choice(list(current_solution))
                elem_bit = 1 << (elem_to_remove - 1)
                new_mask = best_mask & ~elem_bit
                new_score = count_3aps_bitmask(new_mask)
                
                if new_score >= current_score or (new_score - current_score) / T > -5 and random.random() < math.exp((new_score - current_score) / T):
                    best_mask = new_mask
                    current_solution.remove(elem_to_remove)
                    current_score = new_score
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = set(current_solution)
                        best_mask = best_mask
                else:
                    no_improve_count += 1
            elif move_type < 0.6:
                # Add a random element
                available = [x for x in range(1, N + 1) if x not in current_solution]
                if available:
                    elem_to_add = random.choice(available)
                    elem_bit = 1 << (elem_to_add - 1)
                    if can_add_without_4ap_bitmask(best_mask, elem_to_add):
                        new_mask = best_mask | elem_bit
                        new_score = count_3aps_bitmask(new_mask)
                        if new_score >= current_score or (new_score - current_score) / T > -5 and random.random() < math.exp((new_score - current_score) / T):
                            best_mask = new_mask
                            current_solution.add(elem_to_add)
                            current_score = new_score
                            no_improve_count = 0
                            if current_score > best_score:
                                best_score = current_score
                                best_solution = set(current_solution)
                                best_mask = best_mask
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
            else:
                # Swap: remove one, add another
                if len(current_solution) > 0:
                    elem_to_remove = random.choice(list(current_solution))
                    available = [x for x in range(1, N + 1) if x not in current_solution]
                    if available:
                        elem_to_add = random.choice(available)
                        remove_bit = 1 << (elem_to_remove - 1)
                        add_bit = 1 << (elem_to_add - 1)
                        
                        # Check if adding elem_to_add is valid after removing elem_to_remove
                        temp_mask = best_mask & ~remove_bit
                        if can_add_without_4ap_bitmask(temp_mask, elem_to_add):
                            new_mask = temp_mask | add_bit
                            new_score = count_3aps_bitmask(new_mask)
                            
                            if new_score >= current_score or (new_score - current_score) / T > -5 and random.random() < math.exp((new_score - current_score) / T):
                                best_mask = new_mask
                                current_solution.remove(elem_to_remove)
                                current_solution.add(elem_to_add)
                                current_score = new_score
                                no_improve_count = 0
                                if current_score > best_score:
                                    best_score = current_score
                                    best_solution = set(current_solution)
                                    best_mask = best_mask
                            else:
                                no_improve_count += 1
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            
            T *= cooling_rate
        
        # Final hill-climbing polish with multiple passes
        improved = True
        passes = 0
        while improved and time.time() < deadline - 1.0 and passes < 5:
            improved = False
            passes += 1
            
            # Try adding any element that increases score
            for x in range(1, N + 1):
                if x not in current_solution:
                    x_bit = 1 << (x - 1)
                    if can_add_without_4ap_bitmask(best_mask, x):
                        new_mask = best_mask | x_bit
                        new_score = count_3aps_bitmask(new_mask)
                        if new_score > current_score:
                            best_mask = new_mask
                            current_solution.add(x)
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_solution = set(current_solution)
                                best_mask = best_mask
            
            # Try removing any element that doesn't decrease score
            for x in list(current_solution):
                x_bit = 1 << (x - 1)
                new_mask = best_mask & ~x_bit
                new_score = count_3aps_bitmask(new_mask)
                if new_score >= current_score:
                    best_mask = new_mask
                    current_solution.remove(x)
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = set(current_solution)
                        best_mask = best_mask
            
            # Try swaps that improve score
            if time.time() < deadline - 1.0:
                for x_remove in list(current_solution):
                    if time.time() > deadline - 1.5:
                        break
                    remove_bit = 1 << (x_remove - 1)
                    temp_mask = best_mask & ~remove_bit
                    for x_add in range(1, N + 1):
                        if x_add in current_solution:
                            continue
                        add_bit = 1 << (x_add - 1)
                        if can_add_without_4ap_bitmask(temp_mask, x_add):
                            new_mask = temp_mask | add_bit
                            new_score = count_3aps_bitmask(new_mask)
                            if new_score > current_score:
                                best_mask = new_mask
                                current_solution.remove(x_remove)
                                current_solution.add(x_add)
                                current_score = new_score
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_solution = set(current_solution)
                                    best_mask = best_mask
                                break  # Restart outer loop with new state
    
    return sorted(best_solution)
# EVOLVE-BLOCK-END
