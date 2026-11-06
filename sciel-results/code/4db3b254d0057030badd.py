import time
import sys

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Helper to get edge index in lexicographic order (i, j) with 0 <= i < j < N
    # Index = i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
    # Let's verify:
    # i=0, j=1: 0 + 0 = 0. Correct (0,1) is first.
    # i=0, j=N-1: 0 + (N-2) = N-2. Correct.
    # i=1, j=2: 1*(N-1) - 1 + 0 = N-2. Correct.
    # i=N-2, j=N-1: (N-2)(N-1) - (N-2)(N-1)/2 + 0 = (N-2)(N-1)/2.
    # Total edges = N(N-1)/2.
    # Last edge index should be N(N-1)/2 - 1.
    # Let's check sum of previous rows.
    # Row i has N-1-i edges.
    # Sum_{k=0}^{i-1} (N-1-k) = i(N-1) - i(i-1)/2.
    # So offset for row i is i(N-1) - i(i-1)/2.
    # Index within row is j - (i+1).
    # Total index = i(N-1) - i(i-1)/2 + j - i - 1
    #             = i(N-1) - (i^2 - i)/2 + j - i - 1
    #             = iN - i - i^2/2 + i/2 + j - i - 1
    #             = iN - i^2/2 - i/2 + j - 1
    # Let's re-derive simply.
    # Number of edges with first vertex < i is sum_{k=0}^{i-1} (N-1-k).
    # Sum = i(N-1) - (i-1)i/2.
    # Edges in row i start at this sum.
    # Edge (i, j) is the (j-i-1)-th edge in row i (0-indexed).
    # So idx = i(N-1) - i(i-1)/2 + (j-i-1).
    
    def get_edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    colors = [0] * (N * (N - 1) // 2)
    color_id = 0

    # Recursive function to assign colors to edges crossing the partition (left, right)
    def process_partition(vertices):
        nonlocal color_id
        if len(vertices) <= 1:
            return
        
        # Split vertices into two roughly equal halves
        mid = len(vertices) // 2
        left = vertices[:mid]
        right = vertices[mid:]
        
        # Assign a new unique color to all edges between left and right
        for u in left:
            for v in right:
                idx = get_edge_idx(u, v)
                colors[idx] = color_id
        
        # Increment color ID for the next partition
        color_id += 1
        
        # Recurse on left and right partitions
        # Note: We must ensure we don't exceed time, but N=16 is small.
        # Depth is log2(16)=5. Total nodes 15.
        # Edges assigned:
        # Level 0: 8*8 = 64 edges
        # Level 1: 2 * (4*4) = 32 edges
        # Level 2: 4 * (2*2) = 16 edges
        # Level 3: 8 * (1*1) = 8 edges
        # Total = 64+32+16+8 = 120 edges.
        # This covers all edges exactly once?
        # Yes, any pair (u,v) is separated at exactly one level of the binary tree.
        
        # Check time
        if time.time() > deadline:
            return

        process_partition(left)
        process_partition(right)

    process_partition(list(range(N)))
    
    return colors
# EVOLVE-BLOCK-END
