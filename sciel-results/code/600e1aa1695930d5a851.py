"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph on n vertices is 2-colorable.
    Uses bitmask iteration with vertex 0 fixed to 0.
    """
    if not edges:
        return True
    
    # Precompute bitmask for each edge
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Fix vertex 0 to color 0. Iterate over colorings of vertices 1..n-1.
    # Total 2^(n-1) colorings.
    for coloring in range(1 << (n - 1)):
        full_mask = coloring << 1  # Bit 0 is 0, bits 1..n-1 from coloring
        
        is_valid = True
        for emask in edge_masks:
            bits_in_edge = full_mask & emask
            if bits_in_edge == 0 or bits_in_edge == emask:
                is_valid = False
                break
        if is_valid:
            return True
    return False

def minimize_hypergraph(n, time_limit):
    """
    Start with K^5_n and remove edges while maintaining non-2-colorability.
    """
    all_edges = [list(e) for e in combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        random.shuffle(current_edges)
        found_removal = False
        i = 0
        while i < len(current_edges):
            if time.time() - start_time > time_limit:
                break
            
            edge_to_remove = current_edges[i]
            
            # Create a list without this edge
            temp_edges = current_edges[:i] + current_edges[i+1:]
            
            if not is_2_colorable(n, temp_edges):
                current_edges = temp_edges
                found_removal = True
                break
            else:
                i += 1
        
        if not found_removal:
            break
            
    return current_edges

def construct():
    best_edges = None
    best_n = 0
    best_len = float('inf')
    
    time_start = time.time()
    # Allocate time to different n values
    # n=9 is dense, n=12 is sparser.
    # Try n=9, 10, 11, 12.
    time_alloc = {9: 40, 10: 30, 11: 20, 12: 10}
    
    for n in [9, 10, 11, 12]:
        remaining = 110 - (time.time() - time_start)
        if remaining <= 1:
            break
        
        t_limit = min(time_alloc.get(n, 10), remaining)
        
        edges = minimize_hypergraph(n, t_limit)
        
        if edges:
            # Verify non-2-colorability
            if not is_2_colorable(n, edges):
                if len(edges) < best_len:
                    best_len = len(edges)
                    best_edges = edges
                    best_n = n
    
    if best_edges is None:
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
