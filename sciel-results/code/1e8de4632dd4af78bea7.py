"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Generate all edges of Q7
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    m = len(edges)  # 448
    
    # Build conflict graph: two edges conflict if they are opposite edges of a 2-face
    conflict = [set() for _ in range(m)]
    
    # Group edges by direction
    by_dir = [[] for _ in range(d)]
    for idx, (x, y) in enumerate(edges):
        diff = x ^ y
        dir_bit = diff.bit_length() - 1
        by_dir[dir_bit].append(idx)
    
    # For each direction, check pairs of parallel edges
    for dir_bit in range(d):
        dir_edges = by_dir[dir_bit]
        for i in range(len(dir_edges)):
            for j in range(i + 1, len(dir_edges)):
                idx1 = dir_edges[i]
                idx2 = dir_edges[j]
                x1, y1 = edges[idx1]
                x2, y2 = edges[idx2]
                # Check if they form a 2-face: endpoints differ in exactly one bit (other than dir_bit)
                diff12 = x1 ^ x2
                if diff12 & (1 << dir_bit) == 0 and diff12.bit_count() == 1:
                    conflict[idx1].add(idx2)
                    conflict[idx2].add(idx1)
    
    # Convert conflict sets to lists for faster iteration if needed, but sets are fine for intersection
    
    best_selected = []
    best_size = 0
    
    # Use random seed for reproducibility in evaluation if required, but random restarts help
    # We run multiple randomized greedy searches
    num_restarts = 50
    
    for _ in range(num_restarts):
        remaining = set(range(m))
        selected = []
        
        while remaining:
            # Find edges with minimum degree in remaining
            min_deg = float('inf')
            candidates = []
            for idx in remaining:
                deg = len(conflict[idx] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [idx]
                elif deg == min_deg:
                    candidates.append(idx)
            
            # Pick a random candidate among those with minimum degree
            best = random.choice(candidates)
            selected.append(best)
            remaining.remove(best)
            # Remove all conflicts of best from remaining
            for c in conflict[best]:
                if c in remaining:
                    remaining.remove(c)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
            
        # Early stop if we hit theoretical upper bound or high target (308 is upper bound, 304 is best known)
        if best_size >= 308:
            break

    # Convert back to edge list
    result = []
    for idx in best_selected:
        x, y = edges[idx]
        result.append([x, y])
    
    return result
# EVOLVE-BLOCK-END
