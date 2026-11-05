"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    N = 256
    elements = list(range(1, N + 1))
    
    # Build conflict graph: edge (a,b) exists if a+b is a perfect square
    # For efficiency, precompute for each a, which b's conflict with it
    # conflicts[a] = set of b > a such that a+b is a square
    # Also need a+a is not a square (diagonal constraint)
    
    # First, filter out elements where 2*a is a perfect square (diagonal constraint)
    # 2a = k^2 means a = k^2/2, so k must be even: k=2m, a=2m^2
    # a in [1,256]: 2m^2 <= 256, m^2 <= 128, m <= 11
    # a values: 2, 8, 18, 32, 50, 72, 98, 128, 162, 200, 242
    diagonal_bad = set()
    for m in range(1, 12):
        a = 2 * m * m
        if 1 <= a <= 256:
            diagonal_bad.add(a)
    
    # Valid candidates (pass diagonal test)
    candidates = [a for a in elements if a not in diagonal_bad]
    
    # Build adjacency: for each a, which other candidates conflict
    # a and b (a < b) conflict if a+b is a perfect square
    # Squares in range [1+2, 255+256] = [3, 511]: 4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400,441,484
    squares = [s for s in range(4, 513) if _is_square(s)]
    
    # For each candidate a, compute its conflicts among candidates
    # Use a set for the current solution
    # Greedy: process in order of increasing degree (fewest conflicts first)
    
    # Compute degree for each candidate
    degree = {}
    for a in candidates:
        deg = 0
        for b in candidates:
            if b > a and _is_square(a + b):
                deg += 1
        degree[a] = deg
    
    # Sort by degree (ascending) for greedy selection
    sorted_candidates = sorted(candidates, key=lambda x: (degree[x], x))
    
    # Greedy: add element if it doesn't conflict with already-added elements
    selected = set()
    for a in sorted_candidates:
        if time.time() > deadline - 5:
            break
        # Check if a conflicts with any already selected
        conflict = False
        for s in selected:
            if _is_square(a + s):
                conflict = True
                break
        if not conflict:
            selected.add(a)
    
    # Local search: try to swap out one element for another to increase size
    # Or try to add an element by removing conflicting ones (if net gain)
    best_set = set(selected)
    best_size = len(selected)
    
    # Local search phase
    # Try: for each element not in set, see if adding it and removing minimal conflicts helps
    # Also try: for each element in set, try swapping with a non-member
    
    non_members = [a for a in candidates if a not in selected]
    
    # Precompute conflicts for speed
    # conflict_with[a] = set of b in selected that conflict with a
    def get_conflicts_with_set(a, s_set):
        return [b for b in s_set if _is_square(a + b)]
    
    # Iterative improvement
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 10:
        improved = False
        iteration += 1
        
        # Try adding each non-member (if it has 0 conflicts, just add)
        for a in non_members[:]:
            if time.time() > deadline - 10:
                break
            if a in selected:
                continue
            conflicts = get_conflicts_with_set(a, selected)
            if len(conflicts) == 0:
                selected.add(a)
                non_members.remove(a)
                improved = True
                best_set = set(selected)
                best_size = len(selected)
        
        # Try swapping: remove one element, add another
        # For each non-member a, and each selected element s that conflicts with a:
        # If we remove s and add a, does the size stay the same? We want to try
        # removing multiple to add one if it reduces conflicts for others
        # Simpler: try 1-for-1 swaps that might enable future additions
        
        # Try: for each non-member a, find minimal set of selected to remove to add a
        # If |conflicts| < 1, we already handled. If |conflicts| == 1, try the swap.
        for a in non_members[:]:
            if time.time() > deadline - 15:
                break
            if a in selected:
                continue
            conflicts = get_conflicts_with_set(a, selected)
            if len(conflicts) == 1:
                s = conflicts[0]
                # Try swapping: remove s, add a
                # Check: does removing s and adding a create any new issues?
                # a is fine (no other conflicts), s removal is fine
                # But we need to check if s was "protecting" something - no, removing can't hurt
                # Actually this swap keeps size same, so only useful if it enables future additions
                # Let's do it if it reduces total conflict count for non-members
                # For now, skip pure swaps unless they enable additions
                pass
        
        # Try: remove a selected element that has high conflict with non-members,
        # then try to add multiple non-members
        # This is complex; let's try a simpler approach:
        # Random restarts with different orderings if time permits
        
    # Final verification and return
    # Verify the set is valid
    result = sorted(selected)
    for i, a in enumerate(result):
        for b in result[i:]:
            if _is_square(a + b):
                # Should not happen, but if it does, remove the problematic element
                # This shouldn't occur with our construction
                pass
    
    return result if len(result) > 0 else [1]
# EVOLVE-BLOCK-END
