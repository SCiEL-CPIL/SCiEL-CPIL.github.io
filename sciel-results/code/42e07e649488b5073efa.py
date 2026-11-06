import time
import random
import numpy as np

N = 512

def _is_4ap_free_numpy(arr):
    """
    Check if the boolean array (size N) contains no 4-term arithmetic progression.
    Uses vectorized operations where possible, but loops over differences.
    """
    # arr[i] is True if i+1 is in the set
    for d in range(1, (N - 3) // 3 + 1):
        # We need a, a+d, a+2d, a+3d all True
        # a can range from 0 to N - 1 - 3d
        # Indices: a, a+d, a+2d, a+3d
        # We can check if there exists any 'a' such that all are True
        # Let's just check the intersection of shifted arrays
        
        # To be efficient, we can check:
        # mask1 = arr[0:N-3*d]
        # mask2 = arr[d:N-2*d]
        # mask3 = arr[2*d:N-d]
        # mask4 = arr[3*d:N]
        # if np.any(mask1 & mask2 & mask3 & mask4): return False
        
        # This is O(N) per d, and d goes up to ~170. Total ~170*512 ops. Very fast.
        if np.any(arr[:-3*d] & arr[d:-2*d] & arr[2*d:-d] & arr[3*d:]):
            return False
    return True

def _can_add_numpy(arr, x):
    """
    Check if adding element x (1-indexed) to the set represented by arr (0-indexed boolean)
    creates a 4-AP.
    """
    idx = x - 1
    
    # We need to check all 4-APs that include x.
    # Let the 4-AP be a, a+d, a+2d, a+3d.
    # x must be one of these terms.
    
    # 1. x is the 4th term (a+3d = x => a = x-3d)
    # d can be 1..(x-1)//3
    # Check if a, a+d, a+2d are in set
    # a = x-3d, a+d = x-2d, a+2d = x-d
    # Indices: x-1-3d, x-1-2d, x-1-d
    for d in range(1, (x - 1) // 3 + 1):
        if arr[x - 1 - 3*d] and arr[x - 1 - 2*d] and arr[x - 1 - d]:
            return False
            
    # 2. x is the 3rd term (a+2d = x => a = x-2d)
    # d can be 1..(x-1)//2
    # Check if a, a+d, a+3d are in set
    # a = x-2d, a+d = x-d, a+3d = x+d
    # Indices: x-1-2d, x-1-d, x-1+d
    for d in range(1, (x - 1) // 2 + 1):
        if arr[x - 1 - 2*d] and arr[x - 1 - d] and (x + d <= N and arr[x - 1 + d]):
            return False
            
    # 3. x is the 2nd term (a+d = x => a = x-d)
    # d can be 1..min(x-1, (N-x)//2)
    # Check if a, a+2d, a+3d are in set
    # a = x-d, a+2d = x+d, a+3d = x+2d
    # Indices: x-1-d, x-1+d, x-1+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        if arr[x - 1 - d] and (x + d <= N and arr[x - 1 + d]) and (x + 2*d <= N and arr[x - 1 + 2*d]):
            return False
            
    # 4. x is the 1st term (a = x)
    # d can be 1..(N-x)//3
    # Check if a+d, a+2d, a+3d are in set
    # Indices: x-1+d, x-1+2d, x-1+3d
    for d in range(1, (N - x) // 3 + 1):
        if (x + d <= N and arr[x - 1 + d]) and (x + 2*d <= N and arr[x - 1 + 2*d]) and (x + 3*d <= N and arr[x - 1 + 3*d]):
            return False
            
    return True

def _get_conflicts_numpy(arr, x):
    """
    Return a list of conflicts (tuples of 4 indices 1-based) if adding x creates 4-APs.
    """
    conflicts = []
    idx = x - 1
    
    for d in range(1, (x - 1) // 3 + 1):
        if arr[x - 1 - 3*d] and arr[x - 1 - 2*d] and arr[x - 1 - d]:
            conflicts.append((x - 3*d, x - 2*d, x - d, x))
            
    for d in range(1, (x - 1) // 2 + 1):
        if arr[x - 1 - 2*d] and arr[x - 1 - d] and (x + d <= N and arr[x - 1 + d]):
            conflicts.append((x - 2*d, x - d, x, x + d))
            
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        if arr[x - 1 - d] and (x + d <= N and arr[x - 1 + d]) and (x + 2*d <= N and arr[x - 1 + 2*d]):
            conflicts.append((x - d, x, x + d, x + 2*d))
            
    for d in range(1, (N - x) // 3 + 1):
        if (x + d <= N and arr[x - 1 + d]) and (x + 2*d <= N and arr[x - 1 + 2*d]) and (x + 3*d <= N and arr[x - 1 + 3*d]):
            conflicts.append((x, x + d, x + 2*d, x + 3*d))
            
    return conflicts

def construct():
    deadline = time.time() + 58.0  # Safety margin
    
    # 1. Constructive Initialization: Base-3 Cantor Set variant
    # Numbers with only digits 0 and 1 in base 3 are 3-AP free.
    # Are they 4-AP free?
    # Let's check. The standard Cantor set is 3-AP free.
    # It is known that sets with digits {0,1} in base 3 are 3-AP free.
    # What about 4-AP?
    # Actually, the set of numbers with digits {0, 1} in base 3 is 3-AP free.
    # Is it 4-AP free?
    # 0, 1, 3, 4, 9, 10, 12, 13...
    # 0, 1, 2, 3 is not in it (2 has digit 2).
    # 0, 3, 6, 9? 6 is 20 in base 3 (digit 2).
    # It is believed/known that {0,1}_3 is 4-AP free as well?
    # Let's just use it and verify. If it's not 4-AP free, we filter it.
    
    base3_set = []
    for i in range(1, N + 1):
        # Check if i has only 0s and 1s in base 3
        n = i
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid:
            base3_set.append(i)
            
    # This set might not be 4-AP free. We need to ensure it is.
    # Let's convert to numpy array and check.
    arr = np.zeros(N, dtype=bool)
    for x in base3_set:
        arr[x-1] = True
        
    if not _is_4ap_free_numpy(arr):
        # Fallback to greedy if Cantor set fails 4-AP test
        arr = np.zeros(N, dtype=bool)
        for x in range(1, N + 1):
            if _can_add_numpy(arr, x):
                arr[x-1] = True
    else:
        # We have a valid starting set.
        # We can try to add more elements greedily to this base.
        for x in range(1, N + 1):
            if not arr[x-1]:
                if _can_add_numpy(arr, x):
                    arr[x-1] = True

    best_arr = arr.copy()
    best_size = np.sum(best_arr)
    
    # 2. Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any missing element
        missing = [x for x in range(1, N + 1) if not arr[x-1]]
        random.shuffle(missing)
        
        added_any = False
        for x in missing:
            if time.time() > deadline:
                break
            if _can_add_numpy(arr, x):
                arr[x-1] = True
                improved = True
                added_any = True
                if np.sum(arr) > best_size:
                    best_arr = arr.copy()
                    best_size = np.sum(arr)
                break # Restart loop with new state
                
        if added_any:
            continue
            
        # If no simple add, try swaps (remove one, add one)
        # We pick a random element to remove and see if we can add a "better" one
        # Or just try to add any element by removing a conflicting one.
        
        # Let's try a random swap strategy:
        # Pick a random missing element x.
        # If it conflicts, pick a random conflict, remove a random element from that conflict.
        # If the new set is valid, accept.
        
        if missing:
            x = random.choice(missing)
            conflicts = _get_conflicts_numpy(arr, x)
            if conflicts:
                # Pick a random conflict
                ap = random.choice(conflicts)
                # Remove a random element from the AP (other than x)
                others = [y for y in ap if y != x]
                y = random.choice(others)
                
                # Tentative swap
                new_arr = arr.copy()
                new_arr[y-1] = False
                new_arr[x-1] = True
                
                if _is_4ap_free_numpy(new_arr):
                    arr = new_arr
                    improved = True
                    if np.sum(arr) > best_size:
                        best_arr = arr.copy()
                        best_size = np.sum(arr)
                        
    return [i+1 for i in range(N) if best_arr[i]]

# EVOLVE-BLOCK-END
