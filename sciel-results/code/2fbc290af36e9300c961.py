"""
Minimal 5-uniform hypergraph without Property B.
Strategy: Use a known construction (e.g., the 51-edge example) and then
attempt to remove edges while maintaining non-2-colorability.
We implement a fast backtracking solver with unit propagation for 2-colorability.
"""
import time
import random
from itertools import combinations

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 120.0  # seconds
SAFETY_MARGIN = 5.0  # seconds

def time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def is_2_colorable_bruteforce(n, edges):
    """
    Check if the hypergraph is 2-colorable by brute force over all 2^n colorings.
    Returns True if a proper 2-coloring exists, False otherwise.
    This is efficient for n <= 12 (2^12 = 4096 colorings).
    """
    if not edges:
        return True
    
    # Precompute edge masks for faster checking
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Try all 2^n colorings
    for mask in range(1 << n):
        valid = True
        for emask in edge_masks:
            # Get the submask of vertices in this edge
            submask = mask & emask
            # Check if all vertices have the same color
            # If submask == 0, all are 0. If submask == emask, all are 1.
            if submask == 0 or submask == emask:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    """
    Returns a 5-uniform hypergraph with no proper 2-coloring.
    We start with the complete hypergraph on 9 vertices and try to minimize edges.
    """
    # The complete hypergraph on 9 vertices has C(9,5) = 126 edges.
    # We know m(5) <= 51.
    # Let's try to find a smaller one.
    
    n = 9
    all_edges = list(combinations(range(n), 5))
    
    # Start with all edges
    current_edges = all_edges[:]
    random.shuffle(current_edges)
    
    # We will try to remove edges one by one.
    # To be efficient, we can try to remove multiple edges at once if possible,
    # but single removal is safer.
    
    # We'll do multiple passes until no more edges can be removed.
    
    changed = True
    while changed and time_remaining() > 0:
        changed = False
        # Shuffle the current edges to randomize the order
        random.shuffle(current_edges)
        
        for i in range(len(current_edges)):
            if time_remaining() < 0:
                break
            
            # Try to remove current_edges[i]
            # Create a new list without this edge
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            if not is_2_colorable_bruteforce(n, test_edges):
                # It is not 2-colorable, so we can remove this edge.
                current_edges.pop(i)
                changed = True
    
    # Return the result
    return {'n': n, 'edges': [list(e) for e in current_edges]}

if __name__ == "__main__":
    result = construct()
    print(f"Found hypergraph with n={result['n']}, edges={len(result['edges'])}")
