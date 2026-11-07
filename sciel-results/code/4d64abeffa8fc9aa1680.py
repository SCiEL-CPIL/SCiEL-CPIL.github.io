"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 14
    m = 35
    values = list(range(1, n + 1))
    
    # Generate all possible edges with their index sums
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j, i + j))
            
    # Sort edges by index sum to cluster them in the middle
    # This minimizes the number of distinct value sums (i+j+2)
    all_edges.sort(key=lambda x: x[2])
    
    # We want 35 edges.
    # The middle of the sorted list corresponds to index sums around 13.
    # To minimize the range of index sums, we pick a contiguous block of 35 edges
    # centered around the median index sum.
    
    # Find the median index sum
    median_sum = all_edges[len(all_edges)//2][2]
    
    # Select edges closest to the median sum
    # This ensures the index sums are tightly packed, minimizing distinct sums.
    # We sort by distance to median, then by sum to break ties.
    all_edges.sort(key=lambda x: (abs(x[2] - median_sum), x[2]))
    
    selected_edges = [list(e[:2]) for e in all_edges[:m]]
    
    return {'values': values, 'edges': selected_edges}
# EVOLVE-BLOCK-END
