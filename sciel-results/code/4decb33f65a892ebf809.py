# EVOLVE-BLOCK-START
def construct():
    """
    Construct exactly 12 distinct positive Python integers <= 1048576
    such that every subset has a different sum. Minimize the largest integer.
    
    Approach: Greedy construction using the property that a new element x 
    can be added to a set S if and only if x is not a difference between 
    any two elements in S. We maintain the set of differences D.
    """
    n = 12
    # S is the set of subset sums. We don't need to store it explicitly if we track D?
    # Actually, we need to know the last element to ensure we pick x > last_val.
    # And we need to update D.
    
    # Initial state:
    # Elements: []
    # Subset Sums: {0}
    # Differences D: set()
    # Last element: 0
    
    elements = []
    D = set()
    last_val = 0
    
    for _ in range(n):
        # Find the smallest x > last_val such that x not in D
        x = last_val + 1
        while x in D:
            x += 1
        
        # x is a valid candidate
        elements.append(x)
        
        # Update D
        # New differences are |x - d| for d in D, plus x itself
        # Note: We must iterate over a copy of D or build a new set
        # because D is being updated.
        new_diffs = set()
        for d in D:
            new_diffs.add(abs(x - d))
        new_diffs.add(x)
        
        # D_new = D union new_diffs
        D.update(new_diffs)
        last_val = x
        
    return sorted(elements)
# EVOLVE-BLOCK-END
