import time
import random

L = 5354228880  # lcm(1..24)

def subset_sums_bounded(weights, target):
    """
    Returns a set of all possible subset sums of weights that are <= target.
    Weights are positive integers.
    """
    sums = {0}
    for w in weights:
        if w > target:
            continue
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        sums.update(new_sums)
    return sums

def is_valid_set(S, weights):
    """
    Check if S is a valid reciprocal-subsum-free set.
    For each x in S, target = weights[x] must NOT be in subset_sums of {weights[z] for z in S, z != x}.
    """
    for x in S:
        target = weights[x]
        other_weights = [weights[z] for z in S if z != x]
        sums = subset_sums_bounded(other_weights, target)
        if target in sums:
            return False
    return True

def greedy_construct(order, weights, deadline):
    """
    Greedy construction following a specific order.
    Returns the set S.
    """
    S = set()
    
    for x in order:
        if time.time() > deadline:
            break
            
        target = weights[x]
        
        # Get weights of current S
        current_weights = [weights[z] for z in S]
        
        # Check if target is in subset sums of current_weights
        possible_sums = subset_sums_bounded(current_weights, target)
        
        if target in possible_sums:
            # Conflict! Try single swap
            found_swap = False
            # Sort S by value descending to prefer removing large z (keeps smaller elements)
            # Actually, prefer removing the element that's "least needed" - try all
            candidates = sorted(S, key=lambda z: weights[z], reverse=True)
            
            for z in candidates:
                if time.time() > deadline:
                    break
                # Check if removing z resolves conflict
                new_weights = [weights[zz] for zz in S if zz != z]
                new_sums = subset_sums_bounded(new_weights, target)
                
                if target not in new_sums:
                    # Also verify the new set is still valid (removing z might break validity for z's target? No, z is removed)
                    # But we need to check that x's addition doesn't create issues with other elements
                    # The conflict was only about x's target. Other elements' targets were already valid.
                    # Adding x could only create new conflicts for OTHER elements if x's weight helps form their target.
                    # So we need to verify full validity.
                    S.remove(z)
                    S.add(x)
                    if is_valid_set(S, weights):
                        found_swap = True
                        break
                    else:
                        # This swap broke validity elsewhere, undo
                        S.remove(x)
                        S.add(z)
            
            if not found_swap:
                # Try double swap: remove two elements
                candidates = sorted(S, key=lambda z: weights[z], reverse=True)
                n = len(candidates)
                found_double_swap = False
                for i in range(n):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n):
                        z1, z2 = candidates[i], candidates[j]
                        new_weights = [weights[zz] for zz in S if zz != z1 and zz != z2]
                        new_sums = subset_sums_bounded(new_weights, target)
                        
                        if target not in new_sums:
                            S.discard(z1)
                            S.discard(z2)
                            S.add(x)
                            if is_valid_set(S, weights):
                                found_double_swap = True
                                break
                            else:
                                S.remove(x)
                                S.add(z1)
                                S.add(z2)
                        if found_double_swap:
                            break
                    if found_double_swap:
                        break
            
            # If no swap works, skip x
        else:
            # No conflict
            S.add(x)
    
    return S

def local_improve(S, weights, deadline):
    """
    Try to improve S by swapping elements.
    For each x in S and y not in S (y > x), try swapping x out and y in.
    """
    S = set(S)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # Try adding each non-selected element
        for y in range(24, 0, -1):
            if time.time() > deadline:
                break
            if y in S:
                continue
            
            target_y = weights[y]
            # Check if y can be added (no conflict with current S)
            current_weights = [weights[z] for z in S]
            sums = subset_sums_bounded(current_weights, target_y)
            if target_y in sums:
                # Conflict, try swapping out some x in S
                for x in sorted(S, key=lambda z: weights[z]):
                    if time.time() > deadline:
                        break
                    
                    # Try removing x and adding y
                    new_S = set(S)
                    new_S.remove(x)
                    new_S.add(y)
                    
                    if is_valid_set(new_S, weights):
                        S = new_S
                        improved = True
                        break
            else:
                # No conflict, just add y
                S.add(y)
                improved = True
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Precompute reciprocal weights
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i
    
    best_S = set()
    
    # Try multiple orders
    orders = [
        list(range(24, 0, -1)),  # Descending
        list(range(1, 25)),      # Ascending
    ]
    
    # Add a few random orders
    random.seed(42)
    for _ in range(5):
        order = list(range(1, 25))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline:
            break
        
        S = greedy_construct(order, weights, deadline)
        if len(S) > len(best_S):
            best_S = set(S)
    
    # Local improvement on the best set found
    if time.time() < deadline:
        best_S = local_improve(best_S, weights, deadline)
    
    # Final validation
    if not is_valid_set(best_S, weights):
        # Fallback: return a known valid set
        best_S = set(range(13, 25))
    
    return list(best_S)
