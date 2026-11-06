import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def construct():
    # Optimal construction for n=16
    # Base coloring: c(i,j) = min(i,j)
    # This uses colors 0..14
    # Modify edges (13,15) and (14,15) to use new colors 15 and 16
    # Total 16 colors, which is the theoretical maximum n
    
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
            
    # Add new colors
    colors[edge_idx(13, 15)] = 15
    colors[edge_idx(14, 15)] = 16
    
    return colors
