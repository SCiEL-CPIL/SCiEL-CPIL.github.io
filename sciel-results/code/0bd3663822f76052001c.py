"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from collections import deque
    
    denoms = list(range(1, 11))
    invs = [pow(d, -1, 101) for d in denoms]
    
    # BFS on state (residue, mask)
    # dist is implicitly the step count in BFS
    # We want the minimum size (popcount of mask) for each residue.
    # Since BFS expands by adding one element, the first time we reach a residue
    # with a valid mask, it is via the shortest path (fewest elements).
    
    best_mask = [None] * 101
    visited = set()
    q = deque()
    
    # Start state: residue 0, mask 0
    q.append((0, 0))
    visited.add((0, 0))
    
    found_count = 0
    # We need to find non-empty representations.
    # Residue 0 is represented by mask 0 (empty) at start, but we need non-empty.
    # Other residues cannot be reached with mask 0.
    
    while q:
        r, m = q.popleft()
        
        # Check if this state provides a valid representation for residue r
        # Valid if: best_mask[r] is None AND (r != 0 OR m != 0)
        if best_mask[r] is None:
            if r != 0 or m != 0:
                best_mask[r] = m
                found_count += 1
                if found_count == 101:
                    break
        
        # Expand to neighbors: add one more denominator
        for i in range(10):
            bit = 1 << i
            if not (m & bit):
                new_m = m | bit
                new_r = (r + invs[i]) % 101
                state = (new_r, new_m)
                if state not in visited:
                    visited.add(state)
                    q.append(state)
    
    # Construct the output lists
    result = [None] * 101
    for r in range(101):
        mask = best_mask[r]
        if mask is None:
            # Fallback: should not happen, but pick a small set if it does
            # Find any non-empty subset summing to r
            # This is extremely unlikely given the coverage
            mask = 1 
        terms = [d for i, d in enumerate(denoms) if (mask >> i) & 1]
        result[r] = terms
        
    return result
# EVOLVE-BLOCK-END
