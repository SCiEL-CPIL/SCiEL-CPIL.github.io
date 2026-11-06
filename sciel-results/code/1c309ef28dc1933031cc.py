"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 512
    
    # Step 1: Base-3 digit-based construction (Cantor-like set)
    # Select numbers in [1, n] whose base-3 representation contains no digit '2'
    # This gives a 4-AP-free set (by the classic Cantor set argument for k=3, 
    # and also works for k=4 since it's a subset of the 3-AP-free set)
    
    # Actually, let's verify: The Cantor set (no digit 2 in base 3) is 3-AP-free.
    # A set that is 3-AP-free is automatically 4-AP-free (since a 4-AP contains a 3-AP).
    # So this gives us a valid 4-AP-free set.
    
    base3_set = set()
    for x in range(1, n + 1):
        # Convert to base 3 and check for digit 2
        num = x
        has_two = False
        while num > 0:
            if num % 3 == 2:
                has_two = True
                break
            num //= 3
        if not has_two:
            base3_set.add(x)
    
    # Convert to sorted list for bitset operations
    # We'll work with a bitset for efficiency
    # Represent the set as a boolean array of length n+1 (index 0 unused)
    present = np.zeros(n + 1, dtype=np.int8)
    for x in base3_set:
        present[x] = 1
    
    # Step 2: Local search using bitset-based operations
    # Try to add elements that are not in the set, checking if they create a 4-AP
    # Also try to swap elements to potentially increase density
    
    def creates_4ap(x, present_arr):
        """Check if adding x to the set would create a nonconstant 4-term AP."""
        # A 4-term AP is a, a+d, a+2d, a+3d for d > 0
        # x could be any of the four positions
        # Case 1: x is the first term: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if present_arr[x + d] and present_arr[x + 2*d] and present_arr[x + 3*d]:
                return True
        # Case 2: x is the second term: x-d, x, x+d, x+2d
        for d in range(1, min(x - 1, (n - x) // 2) + 1):
            if d >= 1 and x - d >= 1 and present_arr[x - d] and present_arr[x + d] and present_arr[x + 2*d]:
                return True
        # Case 3: x is the third term: x-2d, x-d, x, x+d
        for d in range(1, min((x - 1) // 2, n - x) + 1):
            if x - 2*d >= 1 and present_arr[x - 2*d] and present_arr[x - d] and present_arr[x + d]:
                return True
        # Case 4: x is the fourth term: x-3d, x-2d, x-d, x
        for d in range(1, (x - 1) // 3 + 1):
            if x - 3*d >= 1 and present_arr[x - 3*d] and present_arr[x - 2*d] and present_arr[x - d]:
                return True
        return False
    
    # Try to add elements not in the set
    improved = True
    while improved:
        improved = False
        for x in range(1, n + 1):
            if present[x] == 0:
                if not creates_4ap(x, present):
                    present[x] = 1
                    improved = True
    
    # Step 3: Swap-based local search to further improve density
    # Try removing one element and adding another to see if we can increase size
    # We'll do multiple passes of swap attempts
    
    def can_add_after_removal(x_add, x_remove, present_arr):
        """Check if we can add x_add after removing x_remove without creating a 4-AP."""
        # Temporarily remove x_remove
        present_arr[x_remove] = 0
        
        # Check if x_add can be added
        valid = not creates_4ap(x_add, present_arr)
        
        # Restore x_remove
        present_arr[x_remove] = 1
        
        return valid
    
    # Perform swap-based local search
    # For each element in the set, try to swap it with an element not in the set
    # This can help escape local optima from the greedy addition phase
    
    for _ in range(3):  # Multiple passes of swap attempts
        improved = False
        current_size = int(np.sum(present[1:]))
        
        # Try all possible swaps
        for x_remove in range(1, n + 1):
            if present[x_remove] == 0:
                continue
                
            for x_add in range(1, n + 1):
                if present[x_add] == 1:
                    continue
                
                # Check if swapping x_remove with x_add is beneficial
                # First, we need to check that removing x_remove doesn't break anything
                # (it won't, since we're removing an element)
                # Then check if x_add can be added after removal
                
                if can_add_after_removal(x_add, x_remove, present):
                    # Perform the swap
                    present[x_remove] = 0
                    present[x_add] = 1
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Extract the final set
    result = [x for x in range(1, n + 1) if present[x] == 1]
    
    return sorted(result)
# EVOLVE-BLOCK-END
