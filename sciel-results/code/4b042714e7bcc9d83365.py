"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

N = 256
MAX_D_4AP = (N - 1) // 3  # 85

def _count_3aps_bitmask(mask):
    """Count distinct 3-term APs in the set represented by bitmask."""
    count = 0
    elements = [i+1 for i in range(N) if (mask >> i) & 1]
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i+1, n):
            b = elements[j]
            c = 2*b - a
            if c > N:
                break
            if (mask >> (c-1)) & 1:
                count += 1
    return count

def _has_4ap_fast(mask):
    """Check if mask contains any 4-term AP. Returns True if it does."""
    for d in range(1, MAX_D_4AP + 1):
        inter = mask & (mask >> d) & (mask >> (2*d)) & (mask >> (3*d))
        valid_mask = (1 << (N - 3*d)) - 1
        if inter & valid_mask:
            return True
    return False

def _creates_4ap(mask, x):
    """
    Check if adding x to the set represented by mask creates a 4-AP.
    x is NOT in mask.
    """
    # x can be part of a 4-AP in 4 positions.
    # 1. x, x+d, x+2d, x+3d
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if ((mask >> (x+d-1)) & 1) and ((mask >> (x+2*d-1)) & 1) and ((mask >> (x+3*d-1)) & 1):
            return True
            
    # 2. x-d, x, x+d, x+2d
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if ((mask >> (x-d-1)) & 1) and ((mask >> (x+d-1)) & 1) and ((mask >> (x+2*d-1)) & 1):
            return True
            
    # 3. x-2d, x-d, x, x+d
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        if ((mask >> (x-2*d-1)) & 1) and ((mask >> (x-d-1)) & 1) and ((mask >> (x+d-1)) & 1):
            return True
            
    # 4. x-3d, x-2d, x-d, x
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if ((mask >> (x-3*d-1)) & 1) and ((mask >> (x-2*d-1)) & 1) and ((mask >> (x-d-1)) & 1):
            return True
            
    return False

def _delta_3ap(mask, x, is_add):
    """
    Calculate the change in the number of 3-APs if we add or remove x.
    mask: current bitmask
    x: element to add/remove (1-indexed)
    is_add: True if adding, False if removing
    """
    delta = 0
    
    # Case 1: x is the start (a)
    # (x, x+d, x+2d). Need x+d, x+2d in A.
    max_d_start = (N - x) // 2
    for d in range(1, max_d_start + 1):
        b = x + d
        c = x + 2*d
        if (mask >> (b-1)) & 1:
            if (mask >> (c-1)) & 1:
                delta += 1
                
    # Case 2: x is the middle (b)
    # (x-d, x, x+d). Need x-d, x+d in A.
    max_d_mid = min(x - 1, N - x)
    for d in range(1, max_d_mid + 1):
        a = x - d
        c = x + d
        if (mask >> (a-1)) & 1:
            if (mask >> (c-1)) & 1:
                delta += 1
                
    # Case 3: x is the end (c)
    # (x-2d, x-d, x). Need x-2d, x-d in A.
    max_d_end = (x - 1) // 2
    for d in range(1, max_d_end + 1):
        a = x - 2*d
        b = x - d
        if (mask >> (a-1)) & 1:
            if (mask >> (b-1)) & 1:
                delta += 1
                
    if is_add:
        return delta
    else:
        return -delta

def _construct_greedy(order=None):
    """Greedy construction of a set with no 4-APs."""
    mask = 0
    if order is None:
        order = list(range(1, N+1))
    for x in order:
        if (mask >> (x-1)) & 1:
            continue
        if not _creates_4ap(mask, x):
            mask = mask | (1 << (x-1))
    return mask

def _repair_mask(mask):
    """Remove elements involved in 4-APs until none remain."""
    while _has_4ap_fast(mask):
        found = False
        for d in range(1, MAX_D_4AP + 1):
            inter = mask & (mask >> d) & (mask >> (2*d)) & (mask >> (3*d))
            valid_mask = (1 << (N - 3*d)) - 1
            inter &= valid_mask
            if inter:
                # Find the lowest set bit
                k = (inter & -inter).bit_length() - 1
                a = k + 1
                mask &= ~(1 << (a-1))
                found = True
                break
        if not found:
            break
    return mask

def _local_search_sa(mask, time_limit):
    """Simulated Annealing local search."""
    start_time = time.time()
    best_mask = mask
    best_score = _count_3aps_bitmask(mask)
    current_mask = mask
    current_score = best_score
    
    # Initial temperature
    T = 15.0
    cooling = 0.99995
    min_T = 0.5
    
    elements = [i+1 for i in range(N) if (current_mask >> i) & 1]
    
    while time.time() < start_time + time_limit:
        if T < min_T:
            break
            
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            x = random.randint(1, N)
            if (current_mask >> (x-1)) & 1:
                continue
            if _creates_4ap(current_mask, x):
                continue
            new_mask = current_mask | (1 << (x-1))
            delta = _delta_3ap(current_mask, x, True)
            new_score = current_score + delta
            if new_score > current_score:
                accept = True
            else:
                # Accept worse moves with probability exp(delta/T)
                if T > 0:
                    prob = pow(2.718, (new_score - current_score) / T)
                else:
                    prob = 0
                accept = random.random() < prob
                
            if accept:
                current_mask = new_mask
                current_score = new_score
                elements.append(x)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    
        elif move_type < 0.6:
            # Remove
            if not elements:
                continue
            idx = random.randint(0, len(elements)-1)
            x = elements[idx]
            elements[idx] = elements[-1]
            elements.pop()
            new_mask = current_mask & ~(1 << (x-1))
            delta = _delta_3ap(current_mask, x, False)
            new_score = current_score + delta
            if new_score > current_score:
                accept = True
            else:
                if T > 0:
                    prob = pow(2.718, (new_score - current_score) / T)
                else:
                    prob = 0
                accept = random.random() < prob
                
            if accept:
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    
        else:
            # Swap
            if not elements:
                continue
            idx = random.randint(0, len(elements)-1)
            x_remove = elements[idx]
            x_add = random.randint(1, N)
            if x_add == x_remove:
                continue
            if (current_mask >> (x_add-1)) & 1:
                continue
            
            # Check validity of swap
            # 1. Remove x_remove, check if x_add creates 4-AP in resulting set
            temp_mask = current_mask & ~(1 << (x_remove-1))
            if _creates_4ap(temp_mask, x_add):
                continue
            
            new_mask = temp_mask | (1 << (x_add-1))
            
            # Calculate delta
            # Delta for removing x_remove from current_mask
            delta_remove = _delta_3ap(current_mask, x_remove, False)
            # Delta for adding x_add to temp_mask
            delta_add = _delta_3ap(temp_mask, x_add, True)
            # Note: _delta_3ap for add checks against temp_mask which is correct
            # But we must be careful: the APs formed by adding x_add might involve x_remove?
            # No, x_remove is not in temp_mask.
            # However, the APs destroyed by removing x_remove are those involving x_remove.
            # The APs created by adding x_add are those involving x_add.
            # Since x_remove != x_add, these sets of APs are disjoint?
            # Yes, an AP is a set of 3 distinct numbers. If an AP is destroyed by removing x, it contains x.
            # If an AP is created by adding y, it contains y.
            # If x != y, no AP can be both destroyed and created.
            # So delta = delta_remove + delta_add.
            
            total_delta = delta_remove + delta_add
            new_score = current_score + total_delta
            
            if new_score > current_score:
                accept = True
            else:
                if T > 0:
                    prob = pow(2.718, (new_score - current_score) / T)
                else:
                    prob = 0
                accept = random.random() < prob
                
            if accept:
                # Update elements list
                elements[idx] = x_add
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        
        T *= cooling
        
    return best_mask

def _iterated_local_search(mask, time_limit):
    """
    Iterated Local Search:
    1. Local Search (SA) to find local optimum.
    2. Perturb (swap k random elements).
    3. Repair.
    4. Repeat.
    """
    start_time = time.time()
    best_mask = mask
    best_score = _count_3aps_bitmask(mask)
    current_mask = mask
    
    while time.time() < start_time + time_limit:
        # Local Search with a sub-limit
        remaining = start_time + time_limit - time.time()
        if remaining < 1.0:
            break
        sub_limit = min(5.0, remaining * 0.8) # Limit SA to 5s or 80% of remaining
        
        ls_mask = _local_search_sa(current_mask, sub_limit)
        ls_score = _count_3aps_bitmask(ls_mask)
        
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask
            current_mask = ls_mask
        
        # Perturb
        # Swap 3-5 random elements
        elements = [i+1 for i in range(N) if (current_mask >> i) & 1]
        non_elements = [i for i in range(1, N+1) if i not in set(elements)]
        
        if not elements or not non_elements:
            break
            
        num_swaps = random.randint(3, 5)
        for _ in range(num_swaps):
            if not elements or not non_elements:
                break
            x_remove = random.choice(elements)
            x_add = random.choice(non_elements)
            
            # Apply swap blindly for perturbation
            current_mask = (current_mask & ~(1 << (x_remove-1))) | (1 << (x_add-1))
            
            # Update lists
            elements.remove(x_remove)
            non_elements.remove(x_add)
            elements.append(x_add)
            non_elements.append(x_remove)
        
        # Repair
        current_mask = _repair_mask(current_mask)
        
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 80 # 10s safety margin
    
    # Initial construction
    # Try a few different greedy strategies to get a good start
    best_initial_mask = 0
    best_initial_score = 0
    
    # Strategy 1: Standard increasing greedy
    mask1 = _construct_greedy()
    score1 = _count_3aps_bitmask(mask1)
    best_initial_mask = mask1
    best_initial_score = score1
    
    # Strategy 2: Random order greedy (multiple times)
    for _ in range(10):
        if time.time() > deadline - 20: break
        random_order = list(range(1, N+1))
        random.shuffle(random_order)
        mask = _construct_greedy(random_order)
        score = _count_3aps_bitmask(mask)
        if score > best_initial_score:
            best_initial_score = score
            best_initial_mask = mask
            
    # Local Search / ILS
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_mask = _iterated_local_search(best_initial_mask, remaining_time - 5)
    else:
        best_mask = best_initial_mask
        
    # Final check and convert to list
    if _has_4ap_fast(best_mask):
        best_mask = _repair_mask(best_mask)

    result = [i+1 for i in range(N) if (best_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
