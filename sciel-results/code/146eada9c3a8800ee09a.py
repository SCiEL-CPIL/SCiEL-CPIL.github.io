"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 512

def _can_add(arr, x):
    """
    Check if x can be added to the set represented by arr (bytearray) without creating a 4-AP.
    arr[i] == 1 if i is in the set.
    """
    # Case 1: x is the 1st term: x, x+d, x+2d, x+3d
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if arr[x + d] and arr[x + 2*d] and arr[x + 3*d]:
            return False
    
    # Case 2: x is the 2nd term: x-d, x, x+d, x+2d
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if arr[x - d] and arr[x + d] and arr[x + 2*d]:
            return False
    
    # Case 3: x is the 3rd term: x-2d, x-d, x, x+d
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        if arr[x - 2*d] and arr[x - d] and arr[x + d]:
            return False
    
    # Case 4: x is the 4th term: x-3d, x-2d, x-d, x
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if arr[x - 3*d] and arr[x - 2*d] and arr[x - d]:
            return False
            
    return True

def _greedy_construct(order):
    """
    Greedily construct a set by adding elements in the given order.
    Returns a bytearray representation of the set.
    """
    arr = bytearray(N + 1)
    for x in order:
        if _can_add(arr, x):
            arr[x] = 1
    return arr

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 60s. We stop at 55s to be safe.
    deadline = start_time + 55.0
    
    best_size = 0
    best_arr = bytearray(N + 1)
    
    # Pre-generate a few random orders to start quickly
    # We will loop until time runs out
    
    while time.time() < deadline:
        # Create a random permutation of [1, N]
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        arr = _greedy_construct(order)
        
        # Calculate size
        # Summing a bytearray is fast
        size = sum(arr)
        
        if size > best_size:
            best_size = size
            best_arr = arr[:] # Copy the best array
        
        # Optional: Early termination if we reach a theoretically high bound?
        # For N=512, max 4-AP free set is around 170-200? 
        # If we get very high, we might want to stop, but let's keep searching for max.
        
        # If we have very little time left, break to ensure we return
        if time.time() > deadline - 1.0:
            break

    # Convert best_arr to sorted list
    result = [i for i in range(1, N + 1) if best_arr[i]]
    return result
# EVOLVE-BLOCK-END
