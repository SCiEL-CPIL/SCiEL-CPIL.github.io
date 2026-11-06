"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflict_masks():
    """Precompute conflict masks for each element.
    conflict_masks[x] is a list of bitmasks representing pairs {y, z} 
    such that {x, y, z} is a bad triple."""
    # We use 1 << (x-1) to represent element x
    conflict_masks = [[] for _ in range(129)]
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            mask_bc = (1 << (b-1)) | (1 << (c-1))
            mask_ac = (1 << (a-1)) | (1 << (c-1))
            mask_ab = (1 << (a-1)) | (1 << (b-1))
            
            conflict_masks[a].append(mask_bc)
            conflict_masks[b].append(mask_ac)
            conflict_masks[c].append(mask_ab)
    return conflict_masks

def _can_add(x, selected_mask, conflict_masks):
    """Check if x can be added to selected_mask without creating a bad triple."""
    bit_x = 1 << (x-1)
    if selected_mask & bit_x:
        return False
    for m in conflict_masks[x]:
        if (selected_mask & m) == m:
            return False
    return True

def _greedy(conflict_masks, order):
    """Greedy: add elements in given order."""
    selected_mask = 0
    for x in order:
        if _can_add(x, selected_mask, conflict_masks):
            selected_mask |= (1 << (x-1))
    return selected_mask

def _mask_to_list(mask):
    """Convert bitmask to list of elements."""
    return [i+1 for i in range(128) if (mask >> i) & 1]

def _local_search(selected_mask, conflict_masks, time_limit=10):
    """Local search: try removing elements and adding others to improve size."""
    best_mask = selected_mask
    best_size = bin(selected_mask).count('1')
    start = time.time()
    
    # Precompute list of all elements
    all_elems = list(range(1, 129))
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try single removals + greedy fill
        for rem in all_elems:
            if not (selected_mask & (1 << (rem-1))):
                continue
            candidate_mask = selected_mask & ~(1 << (rem-1))
            
            # Greedily add back
            for x in all_elems:
                if not (candidate_mask & (1 << (x-1))) and _can_add(x, candidate_mask, conflict_masks):
                    candidate_mask |= (1 << (x-1))
            
            size = bin(candidate_mask).count('1')
            if size > best_size:
                best_mask = candidate_mask
                best_size = size
                selected_mask = candidate_mask
                improved = True
        
        # Try double removals + greedy fill (only if time permits)
        if not improved and time.time() - start < time_limit - 2:
            elems_in_set = _mask_to_list(selected_mask)
            random.shuffle(elems_in_set)
            # Limit number of double removals to try to avoid timeout
            max_double_checks = 100
            checks = 0
            for i in range(len(elems_in_set)):
                for j in range(i+1, len(elems_in_set)):
                    if checks >= max_double_checks:
                        break
                    checks += 1
                    rem1, rem2 = elems_in_set[i], elems_in_set[j]
                    candidate_mask = selected_mask & ~(1 << (rem1-1)) & ~(1 << (rem2-1))
                    
                    for x in all_elems:
                        if not (candidate_mask & (1 << (x-1))) and _can_add(x, candidate_mask, conflict_masks):
                            candidate_mask |= (1 << (x-1))
                    
                    size = bin(candidate_mask).count('1')
                    if size > best_size:
                        best_mask = candidate_mask
                        best_size = size
                        selected_mask = candidate_mask
                        improved = True
                        break
                if improved or checks >= max_double_checks:
                    break
        
        if not improved:
            break
            
    return best_mask

def _anneal(selected_mask, conflict_masks, time_limit=15):
    """Simulated annealing with varied move sizes using bitmasks."""
    current_mask = selected_mask
    best_mask = selected_mask
    best_size = bin(selected_mask).count('1')
    start = time.time()
    current_temp = 5.0
    cooling = 0.97
    
    all_elems = list(range(1, 129))
    
    while time.time() - start < time_limit and current_temp > 0.1:
        current_temp *= cooling
        
        # Random move: remove 1-2 random elements, try to add 1-2 others
        elems_in_set = _mask_to_list(current_mask)
        if not elems_in_set:
            for x in all_elems:
                if _can_add(x, current_mask, conflict_masks):
                    current_mask |= (1 << (x-1))
                    break
            continue
            
        n_remove = random.randint(1, 2)
        to_remove = random.sample(elems_in_set, min(n_remove, len(elems_in_set)))
        candidate_mask = current_mask
        for x in to_remove:
            candidate_mask &= ~(1 << (x-1))
        
        # Try to add random elements
        available = [x for x in all_elems if not (candidate_mask & (1 << (x-1)))]
        if available:
            n_add = random.randint(1, 2)
            to_add = random.sample(available, min(n_add, len(available)))
            for x in to_add:
                if _can_add(x, candidate_mask, conflict_masks):
                    candidate_mask |= (1 << (x-1))
        
        # Try to greedily add more
        for y in all_elems:
            if not (candidate_mask & (1 << (y-1))) and _can_add(y, candidate_mask, conflict_masks):
                candidate_mask |= (1 << (y-1))
        
        new_size = bin(candidate_mask).count('1')
        
        if new_size > best_size:
            best_mask = candidate_mask
            best_size = new_size
            current_mask = candidate_mask
        elif random.random() < current_temp / 10.0:
            current_mask = candidate_mask
            
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 88  # Safety margin before 90s kill
    
    # Precompute conflict structure
    conflict_masks = _build_conflict_masks()
    
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy with natural order
    if time.time() < deadline - 30:
        sel_mask = _greedy(conflict_masks, list(range(1, 129)))
        best_mask = sel_mask
        best_size = bin(sel_mask).count('1')
    
    # Strategy 2: Greedy with reverse order
    if time.time() < deadline - 30:
        sel_mask = _greedy(conflict_masks, list(range(128, 0, -1)))
        if bin(sel_mask).count('1') > best_size:
            best_mask = sel_mask
            best_size = bin(sel_mask).count('1')
    
    # Strategy 3: Local search on best
    if time.time() < deadline - 25:
        improved_mask = _local_search(best_mask, conflict_masks, time_limit=8)
        if bin(improved_mask).count('1') > best_size:
            best_mask = improved_mask
            best_size = bin(improved_mask).count('1')
    
    # Strategy 4: Multiple random restarts
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 20 and restarts < 200:
        restarts += 1
        order = list(range(1, 129))
        random.shuffle(order)
        
        sel_mask = _greedy(conflict_masks, order)
        sel_size = bin(sel_mask).count('1')
        
        if sel_size > best_size:
            best_mask = sel_mask
            best_size = sel_size
            
        # Improve if close to best
        if sel_size >= best_size - 3 and time.time() < deadline - 25:
            improved_mask = _local_search(sel_mask, conflict_masks, time_limit=3)
            if bin(improved_mask).count('1') > best_size:
                best_mask = improved_mask
                best_size = bin(improved_mask).count('1')
    
    # Strategy 5: Simulated annealing on best solution
    if time.time() < deadline - 10 and best_size > 0:
        improved_mask = _anneal(best_mask, conflict_masks, time_limit=10)
        if bin(improved_mask).count('1') > best_size:
            best_mask = improved_mask
            best_size = bin(improved_mask).count('1')
    
    # Strategy 6: Final local search
    if time.time() < deadline - 3 and best_size > 0:
        improved_mask = _local_search(best_mask, conflict_masks, time_limit=3)
        if bin(improved_mask).count('1') > best_size:
            best_mask = improved_mask
            best_size = bin(improved_mask).count('1')
    
    if best_size == 0:
        return [1]
    
    return _mask_to_list(best_mask)
# EVOLVE-BLOCK-END
