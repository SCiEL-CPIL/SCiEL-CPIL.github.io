import time
import random

# EVOLVE-BLOCK-START

def _precompute_conflicts():
    """
    Precompute for each x in 1..512, the list of bitmasks representing
    the other 3 elements of any 4-AP containing x.
    conflicts[x] is a list of integers (bitmasks).
    A 4-AP is (a, a+d, a+2d, a+3d).
    """
    n = 512
    conflicts = [[] for _ in range(n + 1)]
    
    # Iterate over all possible 4-APs
    # a is start, d is difference
    # a + 3d <= 512
    for d in range(1, (n - 3) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # The set is {a, b, c, e}
            # For each element in the set, the other 3 form a conflict mask
            
            # If we add 'a', we need {b, c, e} to be absent
            mask_a = (1 << b) | (1 << c) | (1 << e)
            conflicts[a].append(mask_a)
            
            # If we add 'b', we need {a, c, e} to be absent
            mask_b = (1 << a) | (1 << c) | (1 << e)
            conflicts[b].append(mask_b)
            
            # If we add 'c', we need {a, b, e} to be absent
            mask_c = (1 << a) | (1 << b) | (1 << e)
            conflicts[c].append(mask_c)
            
            # If we add 'e', we need {a, b, c} to be absent
            mask_e = (1 << a) | (1 << b) | (1 << c)
            conflicts[e].append(mask_e)
            
    return conflicts

# Global precomputed conflicts
_N = 512
_CONFLICTS = _precompute_conflicts()

def _can_add(x, S_mask):
    """
    Check if element x can be added to the set represented by S_mask
    without creating a 4-AP.
    x is 1-indexed.
    """
    for mask in _CONFLICTS[x]:
        if (S_mask & mask) == mask:
            return False
    return True

def _greedy_construct():
    """
    Construct a 4-AP-free set using a greedy algorithm.
    Returns a bitmask.
    """
    S = 0
    for x in range(1, _N + 1):
        if _can_add(x, S):
            S |= (1 << x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Initial Greedy Construction
    S = _greedy_construct()
    current_size = bin(S).count('1')
    
    # List of elements in S for random selection
    # We can maintain this list or just pick random x and check if it's in S
    # Picking random x from 1..512 and checking S & (1<<x) is fast enough
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any element that is currently free
        # We iterate through all x. If x is not in S and can be added, add it.
        # To make it more effective, we can randomize the order or just go sequential.
        # Sequential is fine for initial filling.
        for x in range(1, _N + 1):
            if not (S & (1 << x)):
                if _can_add(x, S):
                    S |= (1 << x)
                    current_size += 1
                    improved = True
        
        if improved:
            continue
            
        # Phase 2: Local Search / Kick
        # If we couldn't add anything, try removing a random element and refilling.
        if current_size == 0:
            break
            
        # Pick a random element to remove
        # We can pick a random index 1..512 and check if it's in S
        # Or generate a random number 1..current_size and find the k-th set bit?
        # Simpler: random x in 1..512, if in S, remove it.
        # To avoid getting stuck in a local minimum, we can try multiple kicks.
        
        kick_successful = False
        for _ in range(100): # Try up to 100 kicks per "stall"
            if time.time() > deadline:
                break
                
            # Pick random candidate to remove
            y = random.randint(1, _N)
            if not (S & (1 << y)):
                continue
                
            S_backup = S
            S_tmp = S & ~(1 << y)
            size_tmp = current_size - 1
            
            # Try to add elements to S_tmp greedily
            # We can try to add elements that were blocked by y specifically, 
            # but a full scan is safer and fast enough.
            # Optimization: Only check elements that are NOT in S_tmp.
            
            # We can speed up the refill by checking only elements that 
            # might have been blocked by y. But for simplicity and robustness:
            for x in range(1, _N + 1):
                if not (S_tmp & (1 << x)):
                    if _can_add(x, S_tmp):
                        S_tmp |= (1 << x)
                        size_tmp += 1
                        
            if size_tmp > current_size:
                S = S_tmp
                current_size = size_tmp
                kick_successful = True
                break # Restart the outer loop to try adding more
            # If not better, we revert implicitly by not updating S
        
        if not kick_successful:
            # If several kicks fail, we might be in a local minimum.
            # We could try a larger perturbation (remove 2 elements) if time permits.
            # For now, continue trying random kicks.
            pass

    return sorted([i for i in range(1, _N + 1) if S & (1 << i)])

# EVOLVE-BLOCK-END
