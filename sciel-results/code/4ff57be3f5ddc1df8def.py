import time
import random
import itertools

def _is_valid(A):
    """Check if all subset sums are distinct."""
    n = len(A)
    sums = set()
    # Iterate through all 2^n subsets
    for mask in range(1 << n):
        s = 0
        # Calculate sum for this mask
        # Optimization: use bit operations
        # But simple loop is fine for n=12
        for i in range(n):
            if mask & (1 << i):
                s += A[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 12
    best_set = None
    best_max = 1048576
    
    # We expect the optimal max to be well below 1000.
    # Let's search in a range that covers likely optimal solutions.
    # Based on small n cases:
    # n=4: max 7 (vs 8)
    # n=5: max 15? [1,2,4,7,15] -> sums 0..31 distinct?
    # 1,2,4,7,15. Max 15.
    # Powers of 2: 16.
    # n=6: max 31?
    # It seems M(n) is close to 2^n - something.
    # For n=12, 2^12 = 4096. But we are not filling the whole range.
    # The density is low.
    
    # Let's try a random search with local improvement.
    
    # Initialize with a random set
    # We will try multiple restarts
    
    while time.time() < deadline:
        # Random restart
        # Choose a range for random generation
        # Let's try [1, 300]
        A = random.sample(range(1, 300), n)
        A.sort()
        
        # Local search to improve
        # We want to minimize max(A)
        
        # 1. If current set is valid, try to reduce max
        # 2. If not valid, try to fix collisions
        
        # Let's just do a hill climb on max(A) with validity constraint.
        # If invalid, we can't accept.
        
        # Strategy:
        # Try to find a valid set.
        # Once found, try to reduce the largest element.
        
        current_set = A.copy()
        current_max = max(current_set)
        
        # Check if valid
        if _is_valid(current_set):
            if current_max < best_max:
                best_max = current_max
                best_set = current_set.copy()
        
        # Now try to improve current_set (even if invalid, try to make it valid and smaller)
        # Actually, if it's invalid, we should try to make it valid first.
        
        # Simple local search:
        # Pick a random element, change it to a random value in [1, 300].
        # Accept if it improves the "score".
        # Score: -max(A) if valid, -10000 + max(A) if invalid?
        # No, let's just search for valid sets.
        
        # Let's do a random walk.
        for _ in range(1000): # Limit steps per restart
            if time.time() > deadline:
                break
            
            # Generate a neighbor
            i = random.randint(0, n - 1)
            old_val = current_set[i]
            new_val = random.randint(1, 300)
            
            if new_val == old_val:
                continue
            if new_val in current_set:
                continue
                
            current_set[i] = new_val
            current_set.sort()
            
            if _is_valid(current_set):
                new_max = max(current_set)
                # If we found a valid set, we might want to keep it if it's better
                # Or just update best
                if new_max < best_max:
                    best_max = new_max
                    best_set = current_set.copy()
                
                # If we found a valid set, let's try to reduce its max further
                # by specifically targeting the largest element
                # But for now, just keep this as the current state for further exploration
            else:
                # If invalid, revert? Or keep exploring?
                # If we keep exploring invalid states, we might never find a valid one.
                # Let's revert if we were previously valid, or if we want to stay near valid regions.
                # However, random restarts handle this.
                # Let's revert to avoid drifting too far.
                current_set[i] = old_val
                current_set.sort()

    if best_set is None:
        # Fallback: Powers of 2
        return [2**i for i in range(n)]
        
    return best_set
