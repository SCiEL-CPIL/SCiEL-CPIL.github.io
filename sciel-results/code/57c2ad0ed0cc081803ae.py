"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _subset_sums_distinct(arr):
    """Check if all subset sums are distinct. Returns True if valid."""
    n = len(arr)
    seen = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in seen:
            return False
        seen.add(s)
    return True

def _is_valid(arr):
    """Check if the array satisfies all constraints."""
    n = len(arr)
    if n != 12:
        return False
    if len(set(arr)) != n:
        return False
    if any(x <= 0 for x in arr):
        return False
    if max(arr) > 1048576:
        return False
    return _subset_sums_distinct(arr)

def construct():
    """
    Construct exactly 12 distinct positive Python integers <= 1048576
    such that every subset has a different sum. Minimize the largest integer.
    """
    n = 12
    # Start with powers of 2
    arr = [1 << i for i in range(n)]
    best_arr = arr[:]
    best_max = max(arr)
    
    start_time = time.time()
    timeout = 58.0
    
    while time.time() - start_time < timeout:
        # Focus on the largest element
        max_idx = max(range(n), key=lambda i: arr[i])
        
        # Try to reduce the largest element
        # Sample a few smaller values
        # We can also try to swap the largest with a smaller one to free up the value
        # But direct replacement is simpler
        
        found_improvement = False
        for _ in range(100):
            # Pick a random value smaller than the current max
            # Bias towards values close to current max to find small reductions first?
            # Or uniform?
            new_val = random.randint(1, arr[max_idx] - 1)
            
            # Ensure distinctness
            if new_val in arr:
                continue
                
            new_arr = arr[:]
            new_arr[max_idx] = new_val
            
            if _subset_sums_distinct(new_arr):
                arr = new_arr
                current_max = max(arr)
                if current_max < best_max:
                    best_max = current_max
                    best_arr = arr[:]
                found_improvement = True
                break
        
        if not found_improvement:
            # Try to perturb a random other element to "unlock" the max
            # This is a heuristic to escape local minima
            i = random.randint(0, n - 1)
            if i != max_idx:
                for _ in range(10):
                    new_val = random.randint(1, arr[i] + 100) # Allow increase?
                    # Actually, increasing other elements might allow decreasing the max?
                    # No, increasing other elements increases the sum space, making distinctness harder?
                    # Actually, we want to *reduce* the max.
                    # If we can't reduce the max, maybe we need to change the structure.
                    # Let's just try to replace a random element with a random value
                    # to shuffle the set.
                    new_val = random.randint(1, max(arr))
                    if new_val in arr and new_val != arr[i]:
                        continue
                    new_arr = arr[:]
                    new_arr[i] = new_val
                    if _subset_sums_distinct(new_arr):
                        arr = new_arr
                        break

    return best_arr
# EVOLVE-BLOCK-END
