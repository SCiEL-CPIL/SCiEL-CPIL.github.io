import time
import random
import sys

def _check_candidate(x, S, existing_sums):
    """
    Check if x can be added to S without triple-sum collisions.
    """
    n = len(S)
    new_sums = set()
    
    new_sums.add(3 * x)
    for y in S:
        new_sums.add(2 * x + y)
        new_sums.add(x + 2 * y)
    for i in range(n):
        for j in range(i + 1, n):
            new_sums.add(x + S[i] + S[j])
            
    if new_sums & existing_sums:
        return False, None
        
    expected = 1 + 2 * n + n * (n - 1) // 2
    if len(new_sums) < expected:
        return False, None
        
    return True, new_sums

def _remove_element_sums(x, S, existing_sums):
    """
    Remove sums involving x from existing_sums.
    S contains x.
    Returns the modified existing_sums.
    """
    # Create a copy to avoid modifying the original if we revert
    # But for efficiency, we modify in place and revert if needed
    # Actually, let's just return a new set for safety and simplicity, 
    # or modify in place.
    
    # Sums to remove:
    to_remove = set()
    to_remove.add(3 * x)
    
    S_without_x = [y for y in S if y != x]
    
    for y in S_without_x:
        to_remove.add(2 * x + y)
        to_remove.add(x + 2 * y)
        
    for i in range(len(S_without_x)):
        for j in range(i + 1, len(S_without_x)):
            to_remove.add(x + S_without_x[i] + S_without_x[j])
            
    # Remove from existing_sums
    existing_sums -= to_remove
    return existing_sums

def _greedy_build(order, deadline):
    S = []
    existing_sums = set()
    
    for x in order:
        if time.time() > deadline:
            break
            
        valid, new_sums = _check_candidate(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums |= new_sums
            
    return S, existing_sums

def _local_search(S, existing_sums, deadline, rng):
    best_S = S[:]
    best_sums = existing_sums.copy()
    best_size = len(S)
    
    current_S = S[:]
    current_sums = existing_sums.copy()
    
    candidates = list(range(1, 4097))
    
    for _ in range(100): # More iterations with faster ops
        if time.time() > deadline:
            break
            
        if not current_S:
            break
            
        idx = rng.randint(0, len(current_S) - 1)
        removed = current_S.pop(idx)
        
        # Remove sums involving removed
        _remove_element_sums(removed, current_S + [removed], current_sums)
        
        # Try to find replacement
        rng.shuffle(candidates)
        found = False
        for x in candidates:
            if x == removed:
                continue
            if x in current_S:
                continue
                
            valid, new_sums = _check_candidate(x, current_S, current_sums)
            if valid:
                current_S.append(x)
                current_S.sort()
                current_sums |= new_sums
                found = True
                break
                
        if found and len(current_S) > best_size:
            best_S = current_S[:]
            best_sums = current_sums.copy()
            best_size = len(current_S)
        elif not found:
            # Revert
            current_S.append(removed)
            current_S.sort()
            # Re-add sums involving removed
            # We need to regenerate them
            # This is O(k^2)
            n = len(current_S)
            new_sums = set()
            new_sums.add(3 * removed)
            for y in current_S:
                if y == removed: continue
                new_sums.add(2 * removed + y)
                new_sums.add(removed + 2 * y)
            for i in range(n):
                for j in range(i + 1, n):
                    if current_S[i] == removed or current_S[j] == removed: continue
                    new_sums.add(removed + current_S[i] + current_S[j])
            current_sums |= new_sums

    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_S = []
    best_size = 0
    
    # 1. Greedy Increasing
    order1 = list(range(1, 4097))
    S1, sums1 = _greedy_build(order1, deadline)
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
        
    # 2. Randomized Greedy
    for seed in range(50):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        order2 = list(range(1, 4097))
        rng.shuffle(order2)
        
        S2, sums2 = _greedy_build(order2, deadline)
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
            
    # 3. Local Search on best
    if best_S and time.time() < deadline:
        # Rebuild sums if not kept
        # In _greedy_build we return sums, but we didn't store them for best_S
        # Let's just rebuild for safety
        existing_sums = set()
        for i in range(len(best_S)):
            for j in range(i, len(best_S)):
                for k in range(j, len(best_S)):
                    existing_sums.add(best_S[i] + best_S[j] + best_S[k])
                    
        rng = random.Random(999)
        S3 = _local_search(best_S, existing_sums, deadline, rng)
        if len(S3) > best_size:
            best_S = S3
            best_size = len(S3)
            
    return best_S
