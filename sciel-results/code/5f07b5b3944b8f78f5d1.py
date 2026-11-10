"""
Optimized construction of a 4-AP-free subset of [1,256] maximizing 3-AP count.
Uses simulated annealing with smart move generation.
"""
import time
import random
import math

N = 256

def _count_3aps(selected):
    """Count distinct 3-term APs (a, a+d, a+2d) with d>0 in selected."""
    count = 0
    s = set(selected)
    for a in selected:
        for d in range(1, (N - a) // 2 + 1):
            if (a + d) in s and (a + 2 * d) in s:
                count += 1
    return count

def _has_4ap(selected):
    """Check if selected contains a nonconstant 4-term AP."""
    s = set(selected)
    for a in selected:
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            if (a + d) in s and (a + 2 * d) in s and (a + 3 * d) in s:
                return True
    return False

def _greedy_init():
    """Greedy construction: add elements that maximize marginal 3-AP gain while staying 4-AP-free."""
    selected = set()
    remaining = list(range(1, N + 1))
    
    while remaining:
        best_elem = None
        best_gain = -1
        best_new_count = 0
        
        for elem in remaining:
            # Check if adding elem creates a 4-AP
            new_selected = selected | {elem}
            if _has_4ap(new_selected):
                continue
            
            # Count 3-APs involving elem
            gain = 0
            s = selected
            # elem as middle: a, elem, a+2d => a = elem - d, a+2d = elem + d
            for d in range(1, N):
                if (elem - d) in s and (elem + d) in s:
                    gain += 1
            # elem as first: elem, elem+d, elem+2d
            for d in range(1, (N - elem) // 2 + 1):
                if (elem + d) in s and (elem + 2 * d) in s:
                    gain += 1
            # elem as last: elem-2d, elem-d, elem
            for d in range(1, elem // 2 + 1):
                if (elem - 2 * d) in s and (elem - d) in s:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_elem = elem
                best_new_count = _count_3aps(new_selected)
        
        if best_elem is not None:
            selected.add(best_elem)
            remaining.remove(best_elem)
        else:
            break
    
    return sorted(selected)

def _count_3aps_fast(selected_set):
    """Faster 3-AP count using set membership."""
    count = 0
    for a in selected_set:
        for d in range(1, (N - a) // 2 + 1):
            if (a + d) in selected_set and (a + 2 * d) in selected_set:
                count += 1
    return count

def _check_4ap_after_add(selected_set, elem):
    """Check if adding elem to selected_set creates a 4-AP."""
    s = selected_set
    # elem as first: elem, elem+d, elem+2d, elem+3d
    for d in range(1, (N - elem) // 3 + 1):
        if (elem + d) in s and (elem + 2 * d) in s and (elem + 3 * d) in s:
            return True
    # elem as second: elem-d, elem, elem+d, elem+2d
    for d in range(1, min(elem - 1, (N - elem) // 2) + 1):
        if (elem - d) in s and (elem + d) in s and (elem + 2 * d) in s:
            return True
    # elem as third: elem-2d, elem-d, elem, elem+d
    for d in range(1, min(elem // 2, N - elem) + 1):
        if (elem - 2 * d) in s and (elem - d) in s and (elem + d) in s:
            return True
    # elem as fourth: elem-3d, elem-2d, elem-d, elem
    for d in range(1, (elem - 1) // 3 + 1):
        if (elem - 3 * d) in s and (elem - 2 * d) in s and (elem - d) in s:
            return True
    return False

def _check_4ap_after_remove(selected_set, elem):
    """Removing an element cannot create a new 4-AP, so always valid."""
    return True

def _count_3aps_delta_add(selected_set, elem):
    """Count new 3-APs created by adding elem."""
    s = selected_set
    gain = 0
    # elem as middle: (elem-d, elem, elem+d)
    for d in range(1, N):
        if (elem - d) in s and (elem + d) in s:
            gain += 1
    # elem as first: (elem, elem+d, elem+2d)
    for d in range(1, (N - elem) // 2 + 1):
        if (elem + d) in s and (elem + 2 * d) in s:
            gain += 1
    # elem as last: (elem-2d, elem-d, elem)
    for d in range(1, elem // 2 + 1):
        if (elem - 2 * d) in s and (elem - d) in s:
            gain += 1
    return gain

def _count_3aps_delta_remove(selected_set, elem):
    """Count 3-APs lost by removing elem."""
    s = selected_set
    loss = 0
    # elem as middle
    for d in range(1, N):
        if (elem - d) in s and (elem + d) in s:
            loss += 1
    # elem as first
    for d in range(1, (N - elem) // 2 + 1):
        if (elem + d) in s and (elem + 2 * d) in s:
            loss += 1
    # elem as last
    for d in range(1, elem // 2 + 1):
        if (elem - 2 * d) in s and (elem - d) in s:
            loss += 1
    return loss

def _swap_check_4ap(selected_set, remove_elem, add_elem):
    """Check if swapping remove_elem for add_elem keeps the set 4-AP-free."""
    # Remove remove_elem, then check if adding add_elem creates a 4-AP
    s = selected_set - {remove_elem}
    return _check_4ap_after_add(s, add_elem)

def _swap_3ap_delta(selected_set, remove_elem, add_elem):
    """Net change in 3-AP count when swapping remove_elem for add_elem."""
    s = selected_set
    loss = _count_3aps_delta_remove(s, remove_elem)
    # For gain, we need to compute on s - {remove_elem} + {add_elem}
    s2 = s - {remove_elem}
    gain = _count_3aps_delta_add(s2, add_elem)
    return gain - loss

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    random.seed(42)
    
    # Phase 1: Greedy initialization
    best = _greedy_init()
    best_set = set(best)
    best_score = _count_3aps(best_set)
    
    # Phase 2: Simulated annealing with add/remove/swap moves
    current = list(best)
    current_set = set(current)
    current_score = best_score
    
    T = 50.0
    T_min = 0.01
    cooling = 0.9995
    
    iterations = 0
    max_iters = 2000000
    
    while time.time() < deadline and iterations < max_iters and T > T_min:
        iterations += 1
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.35:
            # Add move
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                continue
            elem = random.choice(candidates)
            
            if _check_4ap_after_add(current_set, elem):
                continue
            
            gain = _count_3aps_delta_add(current_set, elem)
            delta = gain
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_set.add(elem)
                current.append(elem)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best = sorted(current_set)
        
        elif move_type < 0.55:
            # Remove move
            if not current_set:
                continue
            elem = random.choice(list(current_set))
            
            loss = _count_3aps_delta_remove(current_set, elem)
            delta = -loss
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_set.remove(elem)
                current.remove(elem)
                current_score += delta
        
        else:
            # Swap move
            if not current_set:
                continue
            remove_elem = random.choice(list(current_set))
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                continue
            add_elem = random.choice(candidates)
            
            if _swap_check_4ap(current_set, remove_elem, add_elem):
                continue
            
            delta = _swap_3ap_delta(current_set, remove_elem, add_elem)
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_set.remove(remove_elem)
                current_set.add(add_elem)
                current.remove(remove_elem)
                current.append(add_elem)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best = sorted(current_set)
        
        T *= cooling
    
    # Phase 3: Local refinement - try all possible swaps
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            best_swap_delta = 0
            best_swap = None
            
            for remove_elem in list(best_set):
                for add_elem in range(1, N + 1):
                    if add_elem in best_set:
                        continue
                    if _swap_check_4ap(best_set, remove_elem, add_elem):
                        continue
                    delta = _swap_3ap_delta(best_set, remove_elem, add_elem)
                    if delta > best_swap_delta:
                        best_swap_delta = delta
                        best_swap = (remove_elem, add_elem)
            
            if best_swap and best_swap_delta > 0:
                rm, ad = best_swap
                best_set.remove(rm)
                best_set.add(ad)
                best_score += best_swap_delta
                best = sorted(best_set)
                improved = True
            
            # Also try single adds
            if time.time() < deadline - 2:
                best_add = None
                best_add_gain = 0
                for elem in range(1, N + 1):
                    if elem in best_set:
                        continue
                    if _check_4ap_after_add(best_set, elem):
                        continue
                    gain = _count_3aps_delta_add(best_set, elem)
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = elem
                if best_add and best_add_gain > 0:
                    best_set.add(best_add)
                    best_score += best_add_gain
                    best = sorted(best_set)
                    improved = True
        
        # Try single removals if they help (unlikely but check)
        if time.time() < deadline - 1:
            for elem in list(best_set):
                loss = _count_3aps_delta_remove(best_set, elem)
                # Only remove if it allows adding many more elements later
                # For now, skip pure removals as they reduce count
    
    return sorted(best_set)
