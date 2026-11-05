import time
import random
import sys

_START_TIME = time.time()
DEADLINE = 110.0  # Safety margin: 120s limit, leave 10s for overhead

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def construct():
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    best_edges = 0
    best_graph = []
    
    attempts = 0
    while _time_left() > 5.0:
        random.seed(42 + attempts)
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        # Use bitmasks for fast neighbor intersection
        neighbor_mask = [0] * n
        graph_edges = []
        
        for u, v in shuffled_edges:
            # Check if adding (u, v) creates a C4
            # A C4 is u-x-y-v where x in N(u), y in N(x) ∩ N(v), y != u
            # We can check: for each x in N(u), if N(x) ∩ N(v) contains any y != u, we have a C4
            # More precisely: if (N(x) & N(v)) has any bit set that is not u
            creates_c4 = False
            # Iterate over neighbors of u
            x_mask = neighbor_mask[u]
            while x_mask:
                # Get lowest set bit
                x = (x_mask & (-x_mask)).bit_length() - 1
                x_mask &= x_mask - 1
                # Check if N(x) ∩ N(v) contains any vertex other than u
                common = neighbor_mask[x] & neighbor_mask[v]
                # Remove u from common
                if common & ~(1 << u):
                    creates_c4 = True
                    break
            
            if not creates_c4:
                neighbor_mask[u] |= (1 << v)
                neighbor_mask[v] |= (1 << u)
                graph_edges.append((u, v))
        
        if len(graph_edges) > best_edges:
            best_edges = len(graph_edges)
            best_graph = graph_edges[:]
            
        attempts += 1
        # Periodic check to avoid too many small iterations
        if attempts % 50 == 0 and _time_left() < 10.0:
            break
    
    # Return the best graph found, colored 0
    result = [[u, v, 0] for u, v in best_graph]
    return result
