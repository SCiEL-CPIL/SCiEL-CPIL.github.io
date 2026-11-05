import itertools
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe against 150s limit
    
    n = 9
    vertices = range(n)
    all_edges = list(itertools.combinations(vertices, 5))
    num_edges = len(all_edges)
    
    # Precompute for each edge, which colorings make it monochromatic
    # Represent colorings as integers 0 to 2^n - 1
    # Skip trivial colorings (all 0s and all 1s)
    num_colorings = 1 << n
    trivial_colorings = {0, (1 << n) - 1}
    valid_colorings = [m for m in range(num_colorings) if m not in trivial_colorings]
    num_valid = len(valid_colorings)
    
    # For each edge, compute the set of valid colorings that make it monochromatic
    # Use numpy arrays for efficiency
    edge_coloring_masks = np.zeros((num_edges, num_valid), dtype=bool)
    
    for i, e in enumerate(all_edges):
        for j, coloring in enumerate(valid_colorings):
            # Check if all vertices in e have the same color under this coloring
            colors = [(coloring >> v) & 1 for v in e]
            if len(set(colors)) == 1:
                edge_coloring_masks[i, j] = True
    
    # Greedy set cover: start with no edges, add edges that cover the most uncovered colorings
    # uncovered[j] = True if coloring j is not yet covered
    uncovered = np.ones(num_valid, dtype=bool)
    selected_edges = []
    
    while uncovered.any():
        if time.time() > deadline - 5.0:
            break
        
        # Find the edge that covers the most uncovered colorings
        # Count how many uncovered colorings each edge covers
        coverage_counts = np.sum(edge_coloring_masks & uncovered[None, :], axis=1)
        
        # If no edge covers any uncovered coloring, we're stuck
        if coverage_counts.max() == 0:
            break
        
        # Select the edge with maximum coverage
        best_edge_idx = np.argmax(coverage_counts)
        selected_edges.append(best_edge_idx)
        
        # Update uncovered: colorings covered by this edge are now covered
        uncovered[edge_coloring_masks[best_edge_idx]] = False
    
    # Convert selected edges to list
    final_edges = [list(all_edges[i]) for i in selected_edges]
    
    return {'n': n, 'edges': final_edges}
