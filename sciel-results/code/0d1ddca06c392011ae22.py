"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

MAX_SUM = 2000  # max possible sum of two elements from 1..1000

def _count_sums_array(chosen):
    """Count all pairwise sums a+b where a<=b from chosen. Returns list of length MAX_SUM+1."""
    counts = [0] * (MAX_SUM + 1)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            s = chosen[i] + chosen[j]
            counts[s] += 1
    return counts

def _can_add_array(chosen, counts, x):
    """Check if x can be added to chosen without violating B2[2] using array counts."""
    for y in chosen:
        s = x + y
        if counts[s] >= 2:
            return False
    if counts[2 * x] >= 2:
        return False
    return True

def _add_to_counts(counts, chosen, x):
    """Update counts in-place after adding x to chosen."""
    for y in chosen:
        s = x + y
        counts[s] += 1
    counts[2 * x] += 1

def _remove_from_counts(counts, chosen, x):
    """Update counts in-place after removing x from chosen."""
    for y in chosen:
        if y != x:
            s = x + y
            counts[s] -= 1
    counts[2 * x] -= 1

def _greedy_construct(order):
    """Greedy construction following a specific order of candidates."""
    chosen = []
    counts = [0] * (MAX_SUM + 1)
    for x in order:
        if _can_add_array(chosen, counts, x):
            chosen.append(x)
            _add_to_counts(counts, chosen, x)
    return chosen

def _greedy_density_aware():
    """Greedy that at each step picks the candidate whose addition creates the fewest new sum collisions."""
    chosen = []
    counts = [0] * (MAX_SUM + 1)
    remaining = set(range(1, 1001))
    
    while remaining:
        best_x = None
        best_score = float('inf')
        
        for x in remaining:
            # Count how many sums would exceed limit if we add x
            collision_count = 0
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    collision_count += 100  # hard violation
                elif counts[s] == 1:
                    collision_count += 1    # soft: would make it 2
            s2 = 2 * x
            if counts[s2] >= 2:
                collision_count += 100
            elif counts[s2] == 1:
                collision_count += 1
            
            if collision_count == 0:
                # No collisions, prefer smaller x to leave room
                if best_x is None or x < best_x:
                    best_x = x
                    best_score = 0
            elif best_x is None or collision_count < best_score:
                best_x = x
                best_score = collision_count
        
        if best_x is None:
            break
        if best_score > 0 and best_score >= 100:
            # Hard violation, try next candidate with less collision
            # Actually we should only pick ones with no hard violations
            # Let's filter: only consider candidates with no hard violations
            valid_candidates = []
            for x in remaining:
                hard_violation = False
                for y in chosen:
                    s = x + y
                    if counts[s] >= 2:
                        hard_violation = True
                        break
                if not hard_violation and counts[2 * x] >= 2:
                    hard_violation = True
                if not hard_violation:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Pick the one with fewest soft collisions
            best_x = None
            best_soft = float('inf')
            for x in valid_candidates:
                soft = 0
                for y in chosen:
                    s = x + y
                    if counts[s] == 1:
                        soft += 1
                if counts[2 * x] == 1:
                    soft += 1
                if best_x is None or soft < best_soft:
                    best_x = x
                    best_soft = soft
        
        if best_x is None:
            break
        
        remaining.discard(best_x)
        chosen.append(best_x)
        _add_to_counts(counts, chosen, best_x)
    
    return sorted(chosen)

def _local_improve_fast(chosen, deadline):
    """Fast local improvement using incremental count updates."""
    chosen = sorted(chosen)
    counts = _count_sums_array(chosen)
    chosen_set = set(chosen)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(chosen)):
            if time.time() > deadline:
                break
            x_remove = chosen[i]
            # Remove x_remove
            _remove_from_counts(counts, chosen, x_remove)
            chosen_set.discard(x_remove)
            temp_chosen = chosen[:i] + chosen[i+1:]
            
            # Find a candidate to add
            best_add = None
            for x in range(1, 1001):
                if x in chosen_set:
                    continue
                if _can_add_array(temp_chosen, counts, x):
                    # Check if this helps: prefer elements that leave more flexibility
                    # Simple heuristic: accept first valid one, or prefer middle-range
                    best_add = x
                    break
            
            if best_add is not None:
                # Accept the swap
                temp_chosen.append(best_add)
                _add_to_counts(counts, temp_chosen, best_add)
                chosen_set.add(best_add)
                chosen = sorted(temp_chosen)
                improved = True
            else:
                # Put x_remove back
                chosen_set.add(x_remove)
                chosen = sorted(temp_chosen + [x_remove])
                _add_to_counts(counts, chosen, x_remove)
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Density-aware greedy
    set1 = _greedy_density_aware()
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1
    
    if time.time() > deadline - 60:
        return best_set
    
    # Strategy 2: Ascending greedy
    order2 = list(range(1, 1001))
    set2 = _greedy_construct(order2)
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2
    
    if time.time() > deadline - 45:
        return best_set
    
    # Strategy 3: Descending greedy
    order3 = list(range(1000, 0, -1))
    set3 = _greedy_construct(order3)
    if len(set3) > best_size:
        best_size = len(set3)
        best_set = set3
    
    if time.time() > deadline - 30:
        return best_set
    
    # Strategy 4: Random shuffles
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 20:
            break
        order = list(range(1, 1001))
        random.shuffle(order)
        s = _greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local improvement on the best set found
    if time.time() < deadline - 5:
        improved = _local_improve_fast(best_set, deadline - 2)
        if len(improved) > len(best_set):
            best_set = improved
    
    return best_set
# EVOLVE-BLOCK-END
