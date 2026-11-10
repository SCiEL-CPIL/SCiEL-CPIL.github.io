import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_valid_solution(arr):
    """
    Verifies:
    1. All elements are distinct integers in [1, N].
    2. It is a Sidon set (all unordered pair sums a+b with a<=b are distinct).
    3. It is maximal (adding any missing element from [1, N] destroys the Sidon property).
    """
    if not arr:
        return False
    
    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False
    
    for x in arr:
        if x < 1 or x > N:
            return False
            
    # Compute all sums
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
            
    # Check maximality
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        
        # Try to add x
        # New sums are {x+a for a in arr} U {2x}
        # If any of these are in `sums`, or if there's a conflict among them, x cannot be added.
        # If x CAN be added (no conflicts), the set is NOT maximal.
        
        # Check conflicts with existing sums
        conflict = False
        for a in arr:
            if (x + a) in sums:
                conflict = True
                break
        if not conflict:
            if (2 * x) in sums:
                conflict = True
        
        if not conflict:
            # x can be added, so the set is not maximal.
            return False
        
    return True

def _greedy_construct(order, rng=None):
    """
    Greedily constructs a Sidon set by iterating through `order`.
    """
    if order is None:
        order = list(range(1, N + 1))
        if rng:
            rng.shuffle(order)
        else:
            random.shuffle(order)
            
    selected = []
    sums = set()
    
    for x in order:
        conflict = False
        for a in selected:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
            
        if (2 * x) in sums:
            continue
            
        selected.append(x)
        for a in selected:
            sums.add(x + a)
        sums.add(2 * x)
        
    return selected

def _try_swap_improve(current_set, rng, deadline):
    """
    Attempts to reduce the size of the set by finding a different maximal set
    via local search. 
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = []
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Initial deterministic passes
    orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s = _greedy_construct(order)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
    # Randomized restarts
    while time.time() < deadline:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        s = _greedy_construct(order, rng)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
            if best_size <= 13:
                break

    if not _is_valid_solution(best_set):
        fallback = [1, 3, 7, 12, 20, 31, 45, 62, 82, 105, 131, 160, 192, 227, 256]
        if not _is_valid_solution(fallback):
            s = _greedy_construct(list(range(1, N+1)))
            return s
            
    return best_set

# EVOLVE-BLOCK-END
