"""Construct a 4-AP-free subset of [1,512] using optimized local search."""
import time
import random

N = 512
DEADLINE = 55.0  # Safety margin: 90s kill, 55s limit leaves buffer for import/eval overhead

def _check_4ap(arr, x):
    """
    Check if adding x to the set represented by boolean array `arr` creates a 4-AP.
    arr is 0-indexed, so index i corresponds to number i+1.
    x is the number to add (1-indexed).
    """
    # x is 1st term: x, x+d, x+2d, x+3d
    # Indices: x-1, x+d-1, x+2d-1, x+3d-1
    # We need arr[x+d-1], arr[x+2d-1], arr[x+3d-1] to be True.
    # d max such that x+3d <= N => d <= (N-x)//3
    for d in range(1, (N - x) // 3 + 1):
        if arr[x + d - 1] and arr[x + 2*d - 1] and arr[x + 3*d - 1]:
            return True

    # x is 2nd term: x-d, x, x+d, x+2d
    # Need arr[x-d-1], arr[x+d-1], arr[x+2d-1]
    # d max such that x+2d <= N => d <= (N-x)//2
    # Also x-d >= 1 => d <= x-1
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if arr[x - d - 1] and arr[x + d - 1] and arr[x + 2*d - 1]:
            return True

    # x is 3rd term: x-2d, x-d, x, x+d
    # Need arr[x-2d-1], arr[x-d-1], arr[x+d-1]
    # d max such that x+d <= N => d <= N-x
    # Also x-2d >= 1 => 2d <= x-1 => d <= (x-1)//2
    max_d = min(N - x, (x - 1) // 2)
    for d in range(1, max_d + 1):
        if arr[x - 2*d - 1] and arr[x - d - 1] and arr[x + d - 1]:
            return True

    # x is 4th term: x-3d, x-2d, x-d, x
    # Need arr[x-3d-1], arr[x-2d-1], arr[x-d-1]
    # d max such that x-3d >= 1 => 3d <= x-1 => d <= (x-1)//3
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if arr[x - 3*d - 1] and arr[x - 2*d - 1] and arr[x - d - 1]:
            return True

    return False

def _greedy_add(arr, candidates, deadline):
    """
    Greedily add candidates to arr if no 4-AP is formed.
    Returns the number of additions.
    """
    added = 0
    for x in candidates:
        if time.time() > deadline:
            break
        if arr[x - 1]:
            continue
        if not _check_4ap(arr, x):
            arr[x - 1] = 1
            added += 1
    return added

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_size = 0
    best_arr = bytearray(N)
    
    rng = random.Random(42)
    
    # Strategy: Multiple random greedy restarts + local search (swap)
    # N=512 is small enough for many restarts.
    
    for restart in range(100): # Max 100 restarts, but time-limited
        if time.time() > deadline - 1.0:
            break
            
        arr = bytearray(N)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        # 1. Initial Greedy Fill
        _greedy_add(arr, candidates, deadline)
        current_size = sum(arr)
        
        if current_size > best_size:
            best_size = current_size
            best_arr = bytearray(arr)
        
        # 2. Local Search: Try to swap out one element to allow adding more
        # Simple heuristic: Try removing each element and see if we can add more
        # This is O(N^2) per pass, but N=512, so 250k checks.
        # We limit passes to keep time down.
        
        improved = True
        passes = 0
        max_passes = 10
        
        while improved and passes < max_passes and time.time() < deadline - 0.5:
            improved = False
            passes += 1
            # Try removing each element in the set
            # To speed up, only try a subset or all? All is safer for quality.
            # Iterate over indices where arr[i] == 1
            indices_to_check = [i for i in range(N) if arr[i] == 1]
            rng.shuffle(indices_to_check)
            
            for idx in indices_to_check:
                if time.time() > deadline - 0.5:
                    break
                
                x = idx + 1
                # Try removing x
                arr[idx] = 0
                size_before = current_size - 1
                
                # Try to add elements that were previously blocked or just any empty slot
                # We can just run a quick greedy on all empty slots?
                # That's O(N * cost_check). 
                # Let's try adding specific candidates: those that formed 4-APs with x?
                # Simpler: Just run _greedy_add on all empty spots.
                # But _greedy_add iterates all N.
                
                # Optimization: Only try to add elements that are "close" or just do a full pass.
                # Given N=512, a full pass is ~512 * (avg 100 ops) = 50k ops.
                # If we do this for 512 removals, that's 25M ops. Too slow per pass.
                
                # Alternative: Just check if removing x allows adding x_back? No.
                # Let's stick to a simpler local move: 
                # Remove x, then try to add a random set of candidates.
                
                # Actually, a very effective move for AP-free sets is:
                # If we remove x, we might be able to add some y.
                # Let's just try to add all elements that were not in the set, in random order.
                # But we need to be fast.
                
                # Let's try a limited greedy: 
                # Take the current list of candidates (empty slots) and try to add them.
                # We can cache the empty slots.
                
                # To save time, we'll only do this swap logic for a few random removals per pass.
                if idx % 10 != 0: # Only try 1/10th of removals per pass for speed
                    arr[idx] = 1
                    continue
                
                # Get empty slots
                empty_slots = [i+1 for i in range(N) if arr[i] == 0]
                rng.shuffle(empty_slots)
                
                # Try to add from empty_slots
                added_any = False
                for y in empty_slots:
                    if time.time() > deadline - 0.5:
                        break
                    if not _check_4ap(arr, y):
                        arr[y-1] = 1
                        current_size += 1
                        added_any = True
                        # We can stop adding for this removal if we got a net gain?
                        # No, keep adding until no more can be added.
                
                if current_size > size_before:
                    improved = True
                    # Keep the new state
                else:
                    # Revert: remove what we added, put x back
                    # This is hard to revert easily if we added multiple.
                    # Better approach: Work on a copy?
                    # Copying bytearray(N) is fast.
                    
                    # Let's refactor: Use a copy for the swap attempt.
                    pass
                # Re-enable x if we didn't keep the improvement?
                # The logic above is messy. Let's simplify.
                
        # Refined Local Search:
        # Instead of complex in-place revert, we'll do:
        # 1. Pick a random element x in S.
        # 2. Create S' = S \ {x}.
        # 3. Run greedy add on S' with all candidates.
        # 4. If |S'| > |S|, keep S'.
        
        # This is O(N * Cost(Greedy)). Cost(Greedy) is O(N * Cost(Check)).
        # Cost(Check) is ~100 ops. Cost(Greedy) ~ 50k ops.
        # 512 * 50k = 25M ops per full pass.
        # Python does ~10M-50M simple ops/sec. So 1 pass takes ~0.5-2s.
        # We have 55s. We can afford ~20-50 passes of this local search.
        
        # Let's implement this cleaner local search.
        
        improved = True
        local_passes = 0
        max_local_passes = 20
        
        while improved and local_passes < max_local_passes and time.time() < deadline - 0.5:
            improved = False
            local_passes += 1
            
            # Get current elements
            current_elements = [i+1 for i in range(N) if arr[i] == 1]
            if not current_elements:
                break
                
            rng.shuffle(current_elements)
            
            # Try a few random removals per pass
            for x in current_elements[:5]: # Try 5 random removals per pass
                if time.time() > deadline - 0.5:
                    break
                
                # Copy current arr
                arr_copy = bytearray(arr)
                arr_copy[x-1] = 0
                size_before = current_size - 1
                
                # Get candidates (all empty in arr_copy)
                # We can just iterate 1..N
                added = _greedy_add(arr_copy, list(range(1, N+1)), deadline)
                
                if size_before + added > current_size:
                    # Improvement found
                    arr = arr_copy
                    current_size = size_before + added
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_arr = bytearray(arr)
                    
        # End local search for this restart

    # Return sorted list of elements in best_arr
    return [i+1 for i in range(N) if best_arr[i] == 1]
