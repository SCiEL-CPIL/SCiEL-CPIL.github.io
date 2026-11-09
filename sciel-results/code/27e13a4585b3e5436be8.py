import time
import random
import sys

START_TIME = time.time()
DEADLINE = 85.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def get_subset_sum_bitsets(A):
    """
    Returns a list of bitsets S where S[j] has bit s set if sum s is achievable with cardinality j.
    Returns None if invalid (cross-cardinality collision).
    """
    n = len(A)
    if n == 0:
        return [1]
    S = [0] * (n + 1)
    S[0] = 1
    
    for i in range(n):
        x = A[i]
        for j in range(i, -1, -1):
            if S[j] == 0:
                continue
            new_part = S[j] << x
            # Check collision with other cardinalities (0..j, and j+1..i which already exist)
            # Actually we need to check against ALL other cardinalities
            # But we can be smarter: check against 0..j and j+2..i (j+1 is being updated)
            for k in range(0, j + 1):
                if new_part & S[k]:
                    return None
            for k in range(j + 2, i + 1):
                if new_part & S[k]:
                    return None
            S[j + 1] |= new_part
            
    # Final check: verify no cross-cardinality collisions
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return None
    return S

def check_addition_valid(S, current_k, x):
    """
    Check if adding x to a set with current cardinality-indexed bitsets S is valid.
    S has length current_k + 1. Returns (new_S, True) or (None, False).
    This is the fast incremental check.
    """
    n = current_k
    # New sums at cardinality j+1: S[j] << x for j in 0..n
    # We need to check that for each j, (S[j] << x) doesn't collide with S[k] for k != j+1
    
    # Build new bitsets
    new_S = [0] * (n + 2)
    new_S[0] = 1  # empty set
    
    for j in range(n + 1):
        if S[j] == 0:
            continue
        new_part = S[j] << x
        
        # Check collision with all other cardinalities
        # Existing cardinalities: 0..n
        # New cardinality j+1: will contain new_part OR existing S[j+1] (if j+1 <= n)
        
        # First check against existing sums at cardinalities other than j+1
        for k in range(n + 1):
            if k == j + 1:
                continue
            if new_part & S[k]:
                return None, False
        
        # Also need to check: new_part combined with existing S[j+1] (if any)
        # The new S[j+1] will be S[j+1] | new_part
        # We already checked new_part against S[k] for k != j+1
        # But S[j+1] itself might have bits that... no, S[j+1] is already valid with others
        
        new_S[j + 1] |= new_part
    
    # Copy existing sums
    for j in range(n + 1):
        new_S[j] |= S[j]
    
    # Final validation: check no cross-cardinality collisions
    for j in range(n + 2):
        for k in range(j + 1, n + 2):
            if new_S[j] & new_S[k]:
                return None, False
    
    return new_S, True

def greedy_build():
    """
    Greedy construction: try to build a valid set by adding elements one by one.
    Uses multiple random orderings and heuristics.
    """
    best_set = []
    best_S = None
    
    all_values = list(range(1, 257))
    
    for attempt in range(50):
        if is_done():
            break
        
        # Different ordering strategies
        if attempt % 3 == 0:
            order = list(range(1, 257))
            random.shuffle(order)
        elif attempt % 3 == 1:
            order = list(range(1, 257))
            order.sort()  # smallest first
        else:
            order = list(range(1, 257))
            order.sort(reverse=True)  # largest first
            random.shuffle(order[:50])  # shuffle top 50
        
        current_set = []
        S = [1]  # S[0] = 1 (bit 0 set)
        
        for x in order:
            if len(current_set) >= 60:
                break
            if is_done():
                break
            
            new_S, valid = check_addition_valid(S, len(current_set), x)
            if valid:
                current_set.append(x)
                S = new_S
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            best_S = S[:]
    
    return best_set

def local_search(initial, time_budget=40.0):
    """
    Local search to improve set size.
    """
    start_time = time.time()
    best_set = list(initial)
    best_size = len(best_set)
    best_S = get_subset_sum_bitsets(best_set)
    
    if best_S is None:
        return best_set
    
    all_values = set(range(1, 257))
    current_set = set(best_set)
    current_S = best_S[:]
    current_size = best_size
    
    # Temperature for simulated annealing
    temperature = 5.0
    cooling_rate = 0.99
    
    while not is_done() and (time.time() - start_time) < time_budget:
        improved = False
        
        # Try adding random elements
        candidates = list(all_values - current_set)
        if candidates:
            random.shuffle(candidates)
            for x in candidates[:20]:  # try up to 20 random additions
                if len(current_set) >= 60:
                    break
                if is_done():
                    break
                
                new_S, valid = check_addition_valid(current_S, len(current_set), x)
                if valid:
                    current_set.add(x)
                    current_S = new_S
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = list(current_set)
                        best_S = current_S[:]
                    temperature = 5.0  # reheat
        
        # If no improvement, try swaps
        if not improved and current_size < 60 and current_size > 0:
            # Try remove one, add one or two
            items = list(current_set)
            random.shuffle(items)
            
            for idx, removed in enumerate(items[:10]):  # try up to 10 removals
                if is_done():
                    break
                
                temp_set = current_set - {removed}
                temp_list = list(temp_set)
                
                # Rebuild S for temp_list
                temp_S = get_subset_sum_bitsets(temp_list)
                if temp_S is None:
                    continue
                
                # Try adding 1-2 elements
                add_candidates = list(all_values - current_set)
                random.shuffle(add_candidates)
                
                for x in add_candidates[:15]:
                    if len(temp_set) + 1 >= 60:
                        break
                    
                    new_S, valid = check_addition_valid(temp_S, len(temp_list), x)
                    if valid:
                        current_set = temp_set | {x}
                        current_S = new_S
                        current_size = len(current_set)
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_set = list(current_set)
                            best_S = current_S[:]
                        temperature = 5.0
                        break
                    
                    # Try adding 2 elements
                    if len(temp_set) + 2 < 60:
                        for y in add_candidates[:10]:
                            if y == x:
                                continue
                            # First add x
                            new_S1, valid1 = check_addition_valid(temp_S, len(temp_list), x)
                            if not valid1:
                                continue
                            # Then add y
                            new_S2, valid2 = check_addition_valid(new_S1, len(temp_list) + 1, y)
                            if valid2:
                                current_set = temp_set | {x, y}
                                current_S = new_S2
                                current_size = len(current_set)
                                improved = True
                                if current_size > best_size:
                                    best_size = current_size
                                    best_set = list(current_set)
                                    best_S = current_S[:]
                                temperature = 5.0
                                break
                        if improved:
                            break
                
                if improved:
                    break
        
        if not improved:
            temperature *= cooling_rate
        
        # Periodically try a fresh greedy from scratch
        if not improved and temperature < 0.1 and (time.time() - start_time) < time_budget - 10:
            fresh = greedy_build()
            if len(fresh) > best_size:
                best_set = fresh
                best_size = len(fresh)
                best_S = get_subset_sum_bitsets(best_set)
                if best_S is not None:
                    current_set = set(best_set)
                    current_S = best_S[:]
                    current_size = best_size
                    temperature = 5.0
    
    return best_set

def construct():
    # Start with greedy build
    greedy_result = greedy_build()
    
    if is_done():
        if greedy_result:
            return list(greedy_result)
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # If greedy got a decent set, try to improve with local search
    if len(greedy_result) >= 5:
        remaining_time = DEADLINE - elapsed() - 2
        if remaining_time > 5:
            improved = local_search(greedy_result, time_budget=remaining_time)
            if improved and len(improved) > len(greedy_result):
                return list(improved)
            return list(greedy_result)
    
    # Fallback: try local search from a small base
    if not is_done():
        base = [1, 2, 4, 8, 16, 32]
        remaining_time = DEADLINE - elapsed() - 2
        if remaining_time > 5:
            improved = local_search(base, time_budget=remaining_time)
            if improved and len(improved) > len(greedy_result):
                return list(improved)
    
    if greedy_result:
        return list(greedy_result)
    return [1, 2, 4, 8, 16, 32, 64, 128, 256]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
    # Validate
    S = get_subset_sum_bitsets(res)
    print(f"Valid: {S is not None}")
