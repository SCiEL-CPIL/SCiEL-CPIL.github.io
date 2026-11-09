"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 distinct positive integers <= 1048576 such that all 4096 subset sums
    are distinct. Minimize the largest integer.
    
    Strategy: Greedy minimal extension. Start with empty set of sums {0}.
    For each new element, pick the smallest positive integer not yet used such that
    adding it to any existing subset sum does not collide with any existing subset sum.
    This is equivalent to: the new number x must satisfy that {x + s : s in sums} 
    is disjoint from sums.
    
    We maintain the set of all subset sums. When we add a new element x, the new sums
    are the old sums union (old sums + x). For no collision, we need x not in sums and 
    x + s not in sums for all s in sums. Equivalently, x must not be in the set of
    differences {a - b : a, b in sums, a != b} ∪ sums (but actually more simply:
    the set of forbidden values for x is {s1 - s2 : s1, s2 in sums, s1 > s2} ∪ {s : s in sums, s > 0}).
    
    Actually, the condition is: for all s in sums, x + s not in sums. This means x not in {s' - s : s, s' in sums, s' > s}.
    So forbidden = {s' - s : s, s' in sums, s' > s}.
    
    We also need x > 0 and x not already used (but if x is a difference, it's forbidden anyway; 
    actually x could equal some sum element but that's covered: if x = s for some s in sums, then x + 0 = x = s which is in sums, collision).
    
    So: forbidden = {s' - s : s, s' in sums, s' > s}
    Pick smallest positive integer not in forbidden.
    """
    deadline = time.time() + 80  # safety margin
    
    n = 12
    max_val = 1048576
    
    # Start with empty subset sums = {0}
    sums = {0}
    elements = []
    
    for _ in range(n):
        if time.time() > deadline:
            break
        
        # Compute forbidden values: all positive differences between elements of sums
        # forbidden = {s' - s : s, s' in sums, s' > s}
        sums_list = sorted(sums)
        m = len(sums_list)
        
        forbidden = set()
        for i in range(m):
            si = sums_list[i]
            for j in range(i + 1, m):
                sj = sums_list[j]
                forbidden.add(sj - si)
        
        # Find smallest positive integer not in forbidden
        x = 1
        while x in forbidden:
            x += 1
        
        if x > max_val:
            # Can't find a valid element within bounds
            break
        
        elements.append(x)
        # Update sums: new sums = old sums union (old sums + x)
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums = sums | new_sums
    
    # If we didn't get all 12 elements, return what we have (shouldn't happen for n=12)
    if len(elements) < n:
        # Fallback to binary representation
        return [1 << i for i in range(n)]
    
    return elements
# EVOLVE-BLOCK-END
