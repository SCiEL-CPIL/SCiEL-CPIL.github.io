"""
Strategy: Simulated Annealing with Conflict Tracking for 4-AP-free sets.
This approach uses a local search heuristic to maximize the cardinality of a subset of [1, n]
containing no 4-term arithmetic progression. It starts with a greedy construction and then
uses simulated annealing to swap elements, allowing temporary worsening of the objective
function (number of 4-APs) to escape local optima.
"""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    n = 512
    
    # --- Precompute 4-APs ---
    # A 4-AP is (a, a+d, a+2d, a+3d) with d > 0 and a+3d <= n
    # We store them as tuples of 0-indexed variables for faster array lookups
    ap_list = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            # Convert to 0-indexed
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(ap_list)
    
    # --- Helper: Count 4-APs in a set ---
    # in_set is a boolean list of length n
    def count_aps(in_set):
        count = 0
        for ap in ap_list:
            if in_set[ap[0]] and in_set[ap[1]] and in_set[ap[2]] and in_set[ap[3]]:
                count += 1
        return count

    # --- Helper: Check if adding x creates a new 4-AP ---
    # This is used for greedy construction
    def can_add_greedy(x, in_set):
        x_idx = x - 1
        # Check if x is part of any 4-AP where the other 3 are already in the set
        # We can iterate through all APs containing x, but that's slow.
        # Instead, we can check specific patterns.
        # x as 1st term: x, x+d, x+2d, x+3d
        # x as 2nd term: x-d, x, x+d, x+2d
        # x as 3rd term: x-2d, x-d, x, x+d
        # x as 4th term: x-3d, x-2d, x-d, x
        
        # 1st term
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            if in_set[x + d - 1] and in_set[x + 2 * d - 1] and in_set[x + 3 * d - 1]:
                return False
        # 2nd term
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            if in_set[x - d - 1] and in_set[x + d - 1] and in_set[x + 2 * d - 1]:
                return False
        # 3rd term
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            if in_set[x - 2 * d - 1] and in_set[x - d - 1] and in_set[x + d - 1]:
                return False
        # 4th term
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if in_set[x - d - 1] and in_set[x - 2 * d - 1] and in_set[x - 3 * d - 1]:
                return False
        
        return True

    # --- Initial Greedy Construction ---
    # Try to build a large set with no 4-APs
    in_set = [False] * n
    selected_count = 0
    
    # Randomize the order of selection to get different starting points
    # But for deterministic reproducibility of the "greedy base", we might just use natural order
    # However, randomization helps escape symmetric traps.
    # Let's use a randomized greedy first.
    random.seed(42) # Fixed seed for reproducibility in evaluation? Or maybe better to use system time?
    # The task says "Maximize cardinality". A single run is evaluated.
    # Using a fixed seed ensures consistent performance.
    
    candidates = list(range(n))
    random.shuffle(candidates)
    
    for idx in candidates:
        x = idx + 1
        if can_add_greedy(x, in_set):
            in_set[idx] = True
            selected_count += 1
            
    # Initial solution
    current_set = in_set[:]
    current_size = selected_count
    current_aps = 0 # Should be 0 by construction
    
    # --- Simulated Annealing ---
    # Objective: Maximize size, Minimize APs.
    # We want to reach a state with 0 APs and max size.
    # Penalty function: Score = Size - Lambda * APs
    # We want Score to be high.
    # If APs > 0, the set is invalid. We must drive APs to 0.
    
    # Strategy:
    # 1. Start with a valid set (0 APs).
    # 2. Try to add elements (increasing size) but this creates APs.
    # 3. Use swaps to remove APs while maintaining or increasing size.
    
    # Actually, a more robust strategy for this specific problem:
    # Start with a random subset of size K (e.g., 100).
    # Count APs.
    # Perform swaps: pick one element to remove, one to add.
    # Accept if APs decrease, or if APs same and size increases, or with prob exp(-delta/T).
    
    # Let's restart with a better initialization:
    # Use the greedy set as a starting point, but allow "breaking" it to fix APs?
    # No, greedy is 0 APs. We want to make it BIGGER.
    
    # Alternative SA approach:
    # State: in_set
    # Move: Swap a selected element with an unselected element.
    # Delta: Change in (Size - M * APs). Since Size is constant in a swap, Delta = -M * Delta_APs.
    # If we swap to reduce APs, we accept.
    # But how do we increase Size?
    # We need a move that adds an element and removes nothing? That increases Size but increases APs.
    # So we need a composite move or a different neighborhood.
    
    # Neighborhood:
    # 1. Add an element (Size +1, APs likely +).
    # 2. Remove an element (Size -1, APs likely -).
    # 3. Swap (Size 0, APs change).
    
    # Let's use a multi-start local search with swaps and add/remove.
    
    best_set = current_set[:]
    best_size = current_size
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.99
    
    T = T_start
    
    while T > T_end and time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add an element
            # Pick a random unselected element
            unselected_indices = [i for i in range(n) if not current_set[i]]
            if not unselected_indices:
                continue
            idx_add = random.choice(unselected_indices)
            if current_set[idx_add]:
                continue
            
            # Calculate delta APs
            # New APs created are those containing idx_add where other 3 are in set
            delta_aps = 0
            for ap in ap_list:
                if idx_add in ap:
                    others = [v for v in ap if v != idx_add]
                    if all(current_set[v] for v in others):
                        delta_aps += 1
            
            new_size = current_size + 1
            # Cost change: Size increases by 1 (good), APs increase by delta_aps (bad)
            # We use a penalty for APs. Let's say penalty weight is high.
            # Actually, we only accept if the resulting set is valid (0 APs) OR if we are in a "repair" phase?
            # No, standard SA on a constrained problem:
            # If new APs > 0, it's invalid. We can only accept invalid states if we plan to fix them.
            # Better: Only allow moves that keep APs at 0? That's just local search.
            # To escape local optima, we need to allow APs > 0 temporarily.
            
            # Let's define the objective function to minimize: -Size + Lambda * APs
            # We want to minimize this.
            # Delta = (-new_size + Lambda * (current_aps + delta_aps)) - (-current_size + Lambda * current_aps)
            # Delta = -1 + Lambda * delta_aps
            
            Lambda = 100.0 # High penalty for APs
            
            delta_cost = -1 + Lambda * delta_aps
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                current_set[idx_add] = True
                current_size = new_size
                current_aps += delta_aps
                
                # Update best if valid
                if current_aps == 0 and current_size > best_size:
                    best_set = current_set[:]
                    best_size = current_size
                    
        elif move_type < 0.8:
            # Remove an element
            selected_indices = [i for i in range(n) if current_set[i]]
            if not selected_indices:
                continue
            idx_rem = random.choice(selected_indices)
            
            # Calculate delta APs
            # APs removed are those containing idx_rem where other 3 are in set
            delta_aps = 0
            for ap in ap_list:
                if idx_rem in ap:
                    others = [v for v in ap if v != idx_rem]
                    if all(current_set[v] for v in others):
                        delta_aps += 1
            
            new_size = current_size - 1
            # Delta = -(-1) + Lambda * (-delta_aps) = 1 - Lambda * delta_aps
            # Wait, objective is Minimize(-Size + Lambda * APs)
            # Old: -Size + Lambda * APs
            # New: -(Size-1) + Lambda * (APs - delta_aps) = -Size + 1 + Lambda * APs - Lambda * delta_aps
            # Delta = 1 - Lambda * delta_aps
            
            Lambda = 100.0
            delta_cost = 1 - Lambda * delta_aps
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                current_set[idx_rem] = False
                current_size = new_size
                current_aps -= delta_aps
                
                if current_aps == 0 and current_size > best_size:
                    best_set = current_set[:]
                    best_size = current_size
                    
        else:
            # Swap
            selected_indices = [i for i in range(n) if current_set[i]]
            unselected_indices = [i for i in range(n) if not current_set[i]]
            
            if not selected_indices or not unselected_indices:
                continue
            
            idx_rem = random.choice(selected_indices)
            idx_add = random.choice(unselected_indices)
            
            if idx_rem == idx_add:
                continue
                
            # Calculate delta APs for removal and addition
            # This is tricky because the set changes.
            # APs affected are those containing idx_rem or idx_add.
            
            # Simulate the change
            # 1. Remove idx_rem
            # 2. Add idx_add
            
            # Let's just calculate the new AP count for the affected APs?
            # Or just recalculate total APs? Too slow.
            
            # Approximation:
            # delta_aps_rem = number of APs containing idx_rem where other 3 are in set (BEFORE change)
            # delta_aps_add = number of APs containing idx_add where other 3 are in set (AFTER removal of idx_rem, BEFORE addition of idx_add)
            
            # This is complex. Let's just do a full check for the swap?
            # No, let's stick to Add/Remove for simplicity and speed, or implement swap carefully.
            
            # Let's skip swap for now and rely on Add/Remove to change size and shape.
            # Actually, swap is crucial for maintaining size while changing structure.
            
            # Let's try a simpler swap logic:
            # Just check if the swap reduces APs.
            # We'll do a quick check.
            
            # Save state
            old_rem = current_set[idx_rem]
            old_add = current_set[idx_add]
            
            current_set[idx_rem] = False
            current_set[idx_add] = True
            
            # Count new APs
            new_aps = count_aps(current_set)
            delta_aps = new_aps - current_aps
            
            if delta_aps < 0 or random.random() < math.exp(-100.0 * delta_aps / T):
                current_aps = new_aps
                # Size remains same
                if current_aps == 0 and current_size > best_size:
                    best_set = current_set[:]
                    best_size = current_size
            else:
                # Revert
                current_set[idx_rem] = old_rem
                current_set[idx_add] = old_add
                
        T *= cooling_rate
        
        # Periodic restart?
        # If stuck (no improvement for a while), restart with a random valid set?
        # For now, just let SA run.
        
    # Final check
    if best_size > 0:
        # Convert back to 1-indexed sorted list
        result = [i + 1 for i in range(n) if best_set[i]]
        return result
    else:
        # Fallback: return a single element
        return [1]
