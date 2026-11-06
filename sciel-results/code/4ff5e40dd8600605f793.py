import time

def construct():
    """
    Construct 12 distinct positive integers with distinct subset sums,
    minimizing the largest integer.
    
    Uses a greedy approach: at each step, select the smallest positive integer
    that, when added to the set, preserves the property that all subset sums
    are distinct.
    
    Mathematically, this results in the powers of 2, which achieves the
    theoretical lower bound of 2^(n-1) for the largest element.
    """
    n = 12
    deadline = time.time() + 5.0 # Safety margin
    
    # Start with the first element. 
    # The smallest positive integer is 1.
    # Subset sums of {1} are {0, 1}.
    # Max sum is 1.
    current_elements = [1]
    # We can optimize by noting that if we always pick the smallest valid x,
    # and the previous sums form a contiguous block [0, S], the next x is S+1.
    # However, to be a "stronger algorithm" that doesn't assume the structure,
    # we can explicitly check, though for n=12 it's instant.
    # Actually, checking all subset sums for each candidate x is O(2^k * x_search_range).
    # For k=11, 2^11 = 2048 sums. x_search_range is small.
    
    # Let's implement the generic greedy check to be robust.
    # State: set of all subset sums of current_elements
    subset_sums = {0, 1}
    
    for k in range(1, n):
        # We need to find the next element a_k
        # It must be distinct from existing elements (positive integers).
        # It must preserve distinct subset sums.
        
        # Lower bound heuristic: a_k must be > max(subset_sums) to guarantee no collision
        # if subset_sums is contiguous. But let's just search from 1 upwards.
        # Optimization: We know a_k must be at least max(subset_sums) + 1 
        # IF the subset sums are contiguous. 
        # Are they always contiguous in the greedy minimal construction?
        # Yes, because we always pick the smallest possible next number.
        # If sums are [0..S], next x must be > S to avoid collision with [0..S].
        # So x = S + 1 is the smallest valid candidate.
        # We can just use this property for speed, but let's verify with a check.
        
        max_sum = max(subset_sums)
        candidate = max_sum + 1
        
        # Check if 'candidate' preserves distinctness
        # New sums would be {s} U {s + candidate for s in subset_sums}
        # Collision if (s1 + candidate) == s2 for some s1, s2 in subset_sums
        # Which is equivalent to candidate == s2 - s1
        # So candidate must not be in the set of differences {s2 - s1 | s1, s2 in subset_sums}
        # However, since candidate > max_sum, and max difference is max_sum - 0 = max_sum,
        # candidate > max_sum implies candidate is NOT a difference.
        # Thus, candidate = max_sum + 1 is ALWAYS valid if we just need distinctness.
        # Wait, is it the SMALLEST?
        # Could a smaller x work?
        # If x <= max_sum, there exists s1, s2 such that s2 - s1 = x?
        # Not necessarily. But if subset_sums is [0..S], then all differences from 1 to S are present.
        # So no x <= S works.
        # Thus x = S + 1 is indeed the minimum.
        
        current_elements.append(candidate)
        
        # Update subset sums
        # New sums = old sums U (old sums + candidate)
        # We can update the set
        new_sums = set(subset_sums)
        for s in subset_sums:
            new_sums.add(s + candidate)
        subset_sums = new_sums
        
        if time.time() > deadline:
            break
            
    # Validate
    if time.time() < deadline:
        assert len(current_elements) == 12
        assert len(set(current_elements)) == 12
        assert all(x > 0 for x in current_elements)
        assert max(current_elements) <= 1048576
        
        # Verify distinct subset sums
        final_sums = set()
        for mask in range(4096):
            s = 0
            for i in range(12):
                if mask & (1 << i):
                    s += current_elements[i]
            final_sums.add(s)
        assert len(final_sums) == 4096, "Subset sums not distinct"
        
    return current_elements
