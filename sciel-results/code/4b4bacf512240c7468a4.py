"""Construct 12 distinct positive integers with all 4096 subset sums distinct.
The theoretical lower bound for the largest element is 2^(n-1). Powers of 2 achieve it.
"""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return exactly 12 distinct positive integers <= 1048576
    such that all 2^12 = 4096 subset sums are distinct.
    Minimize the largest integer.
    
    Uses a greedy algorithm to select the smallest possible next integer
    that preserves the distinctness of subset sums. This yields the powers
    of two, which is the provably optimal solution (largest element = 2^(n-1)).
    """
    n = 12
    max_val = 1048576
    deadline = time.time() + 50  # safety margin before 90s kill

    selected = []
    current_sums = {0}
    
    for i in range(n):
        if time.time() > deadline:
            # Fallback: return powers of two if we somehow time out
            # (should not happen for n=12)
            return [1 << j for j in range(n)]

        if not selected:
            x = 1
        else:
            x = sum(selected) + 1
            
        # Verify distinctness
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + x)
        
        if len(new_sums) != 2 * len(current_sums):
            # Fallback search if the simple rule fails (should not happen)
            found_x = False
            for candidate in range(1, max_val + 1):
                test_sums = set(current_sums)
                for s in current_sums:
                    test_sums.add(s + candidate)
                if len(test_sums) == 2 * len(current_sums):
                    x = candidate
                    found_x = True
                    break
            if not found_x:
                raise ValueError("No valid integer found")
        
        selected.append(x)
        current_sums = new_sums

    # Final verification
    if len(set(selected)) != n:
        raise ValueError("Values not distinct")
    if any(x <= 0 for x in selected):
        raise ValueError("Values must be positive")
    if max(selected) > max_val:
        raise ValueError("Value exceeds maximum")
        
    # Check all subset sums
    all_sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += selected[i]
        all_sums.add(s)
    if len(all_sums) != (1 << n):
        raise ValueError("Subset sums not distinct")

    return selected
# EVOLVE-BLOCK-END
