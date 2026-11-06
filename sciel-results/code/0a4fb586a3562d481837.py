"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 512

def _can_add(S, x):
    """
    Check if x can be added to the set represented by bytearray S (1-indexed).
    S[i] == 1 if i is in the set.
    """
    # Case 1: x is the 4th term (a, a+d, a+2d, x)
    # x-3d, x-2d, x-d must be in S
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if S[x - d] and S[x - 2*d] and S[x - 3*d]:
            return False
            
    # Case 2: x is the 3rd term (a, a+d, x, a+3d)
    # x-2d, x-d, x+d must be in S
    max_d = (x - 1) // 2
    for d in range(1, max_d + 1):
        y = x + d
        if y <= N and S[x - 2*d] and S[x - d] and S[y]:
            return False
            
    # Case 3: x is the 2nd term (a, x, a+2d, a+3d)
    # x-d, x+d, x+2d must be in S
    max_d = x - 1
    for d in range(1, max_d + 1):
        y1 = x + d
        y2 = x + 2*d
        if y2 <= N and S[x - d] and S[y1] and S[y2]:
            return False
            
    # Case 4: x is the 1st term (x, x+d, x+2d, x+3d)
    # x+d, x+2d, x+3d must be in S
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if S[x + d] and S[x + 2*d] and S[x + 3*d]:
            return False
            
    return True

def _random_greedy():
    """
    Construct a 4-AP free set using randomized greedy.
    """
    S = bytearray(N + 1)
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if _can_add(S, x):
            S[x] = 1
            
    return S

def _local_search(S, max_iters=2000):
    """
    Improve the set by removing one element and trying to add others.
    """
    S = bytearray(S)
    current_size = sum(S)
    
    # Precompute list of elements in S for fast removal/iteration if needed
    # But we iterate 1..N for addition, so we just need to know what's in S.
    
    for _ in range(max_iters):
        # Pick a random element to remove
        # To pick randomly, we need a list. 
        # Building a list every time is O(N).
        # We can maintain a list.
        
        # Optimization: Just pick a random index 1..N and check if it's in S.
        # If not, try again? No, that's inefficient if S is dense.
        # Let's maintain a list of active elements.
        
        # For simplicity and speed in Python, we'll just scan for a random element.
        # Actually, let's just pick a random candidate for removal from the current set.
        # We can use a list of elements in S.
        
        # To avoid O(N) list creation every iter, we'll just do this:
        # 1. Find all elements in S.
        # 2. Pick one.
        # This is O(N).
        
        elements = [i for i in range(1, N + 1) if S[i]]
        if not elements:
            break
            
        u = random.choice(elements)
        S[u] = 0
        
        # Try to add any element not in S
        added_count = 0
        added_elements = []
        
        for v in range(1, N + 1):
            if S[v]:
                continue
            if _can_add(S, v):
                S[v] = 1
                added_count += 1
                added_elements.append(v)
                
        new_size = current_size - 1 + added_count
        
        if new_size > current_size:
            current_size = new_size
        else:
            # Revert
            S[u] = 1
            for v in added_elements:
                S[v] = 0
            # If we didn't improve, we might want to break or continue?
            # Continuing allows exploring different removals.
            
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_S = bytearray(N + 1)
    best_size = 0
    
    # 1. Run multiple randomized greedy passes
    # Each pass is O(N^2) ~ 260k ops.
    # Let's run as many as possible.
    
    while time.time() < deadline - 15.0:  # Leave 15s for local search
        S = _random_greedy()
        size = sum(S)
        if size > best_size:
            best_size = size
            best_S = bytearray(S)
            
    # 2. Apply local search to the best set found
    if time.time() < deadline - 2.0:
        improved_S = _local_search(best_S, max_iters=1000)
        improved_size = sum(improved_S)
        
        if improved_size > best_size:
            best_S = improved_S
            best_size = improved_size

    # Return sorted list of elements
    return [i for i in range(1, N + 1) if best_S[i]]
# EVOLVE-BLOCK-END
