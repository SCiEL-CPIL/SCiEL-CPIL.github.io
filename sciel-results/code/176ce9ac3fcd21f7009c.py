# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses bitmask-based local search with perturbations for efficiency.
    """
    m = 205
    k = 2
    deadline = time.time() + 580.0  # 580s safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # For fast lookup, create a boolean array
    forbidden_arr = np.zeros(m, dtype=bool)
    for f in forbidden:
        forbidden_arr[f] = True
    
    # Precompute conflict bitmask for each residue
    # conflict_mask[i] is a bitmask where bit j is set if (i-j) % m is in forbidden
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for f in forbidden:
            j = (i - f) % m
            mask |= (1 << j)
            j2 = (i + f) % m
            mask |= (1 << j2)
        conflict_mask[i] = mask
    
    def is_valid_addition_bitmask(candidate, selected_mask):
        """Check if candidate can be added to selected_mask without creating a forbidden pair."""
        return (conflict_mask[candidate] & selected_mask) == 0
    
    def greedy_construct_bitmask(order, selected_mask):
        """Greedy construction using a specific order of candidates."""
        selected = []
        for candidate in order:
            if not (selected_mask & (1 << candidate)):
                if is_valid_addition_bitmask(candidate, selected_mask):
                    selected.append(candidate)
                    selected_mask |= (1 << candidate)
        return selected, selected_mask
    
    def local_search_bitmask(initial_mask, time_limit):
        """
        Bitmask-based local search to maximize the size of the independent set.
        Uses add/remove/swap moves with bitmask operations for speed.
        """
        S_mask = initial_mask
        best_mask = S_mask
        best_size = bin(S_mask).count('1')
        
        end_time = time.time() + time_limit
        no_improve_count = 0
        
        while time.time() < end_time:
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a random element
                candidate = random.randint(0, m - 1)
                if not (S_mask & (1 << candidate)):
                    if is_valid_addition_bitmask(candidate, S_mask):
                        S_mask |= (1 << candidate)
                        if bin(S_mask).count('1') > best_size:
                            best_size = bin(S_mask).count('1')
                            best_mask = S_mask
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                    
            elif move_type < 0.75:
                # Try to remove a random element
                if S_mask:
                    # Find a set bit
                    bits = [i for i in range(m) if (S_mask & (1 << i))]
                    if bits:
                        to_remove = random.choice(bits)
                        S_mask &= ~(1 << to_remove)
                        no_improve_count += 1
                        
            else:
                # Try a swap: remove one, add another
                if S_mask:
                    bits = [i for i in range(m) if (S_mask & (1 << i))]
                    if bits:
                        to_remove = random.choice(bits)
                        to_add = random.randint(0, m - 1)
                        if not (S_mask & (1 << to_add)):
                            # Check if to_add is valid with S \ {to_remove}
                            temp_mask = S_mask & ~(1 << to_remove)
                            if is_valid_addition_bitmask(to_add, temp_mask):
                                S_mask = temp_mask | (1 << to_add)
                                if bin(S_mask).count('1') > best_size:
                                    best_size = bin(S_mask).count('1')
                                    best_mask = S_mask
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1
                            else:
                                no_improve_count += 1
                        else:
                            no_improve_count += 1
            
            # Perturbation: if stuck for too long, do a random restart from best
            if no_improve_count > 100:
                # With some probability, perturb the current solution
                if random.random() < 0.3:
                    # Remove a random subset of elements to allow re-optimization
                    bits = [i for i in range(m) if (S_mask & (1 << i))]
                    num_remove = max(1, len(bits) // 5)
                    for _ in range(num_remove):
                        if bits:
                            to_remove = random.choice(bits)
                            S_mask &= ~(1 << to_remove)
                            bits.remove(to_remove)
                no_improve_count = 0
        
        return best_mask
    
    # Phase 1: Deterministic greedy with natural order
    best_result, best_mask = greedy_construct_bitmask(list(range(m)), 0)
    best_size = len(best_result)
    
    # Phase 2: Randomized greedy (more restarts)
    random.seed(42)
    for _ in range(200):
        if time.time() > deadline - 100:  # Leave time for local search
            break
        order = list(range(m))
        random.shuffle(order)
        result, mask = greedy_construct_bitmask(order, 0)
        if len(result) > best_size:
            best_result = result
            best_mask = mask
            best_size = len(result)
            
    # Phase 3: Bitmask-based local search from best result
    if time.time() < deadline - 10:
        # Use a significant portion of time for local search
        ls_time = min(deadline - time.time() - 5, 300)
        if ls_time > 10:
            improved_mask = local_search_bitmask(best_mask, ls_time)
            improved_size = bin(improved_mask).count('1')
            if improved_size > best_size:
                best_size = improved_size
                best_mask = improved_mask
                best_result = [i for i in range(m) if (best_mask & (1 << i))]
                
    # Phase 4: More restarts with local search if time permits
    random.seed(123)
    while time.time() < deadline - 10:
        order = list(range(m))
        random.shuffle(order)
        result, mask = greedy_construct_bitmask(order, 0)
        # Only run local search if this greedy result is promising
        if len(result) >= best_size - 3:
            ls_time = min(deadline - time.time() - 5, 30)
            if ls_time > 5:
                improved_mask = local_search_bitmask(mask, ls_time)
                improved_size = bin(improved_mask).count('1')
                if improved_size > best_size:
                    best_size = improved_size
                    best_mask = improved_mask
                    best_result = [i for i in range(m) if (best_mask & (1 << i))]
                    
    # Final validation
    for i, a in enumerate(best_result):
        for b in best_result[i+1:]:
            d1 = (a - b) % m
            d2 = (b - a) % m
            if forbidden_arr[d1] or forbidden_arr[d2]:
                return np.array([], dtype=np.int64)
    
    return np.array(best_result, dtype=np.int64)


# EVOLVE-BLOCK-END
