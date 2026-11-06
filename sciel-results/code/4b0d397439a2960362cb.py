"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_list = P['edge_order']
total_edges = len(edge_list)

# Precompute edge position (1-based) for each pair
edge_pos = [[-1] * n for _ in range(n)]
for i, (u, v) in enumerate(edge_list):
    edge_pos[u][v] = i + 1
    edge_pos[v][u] = i + 1

def score_cycle(cycle):
    """Compute the score of a cycle: max 1-based arrival index of its edges."""
    max_idx = 0
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        idx = edge_pos[u][v]
        if idx > max_idx:
            max_idx = idx
    return max_idx

def dijkstra_bottleneck_hc(deadline):
    """
    Find the minimum-bottleneck Hamiltonian cycle using Dijkstra on (mask, last) states.
    Returns (score, cycle) or (None, None) if not found within deadline.
    """
    start_time = time.time()
    
    # State: (bottleneck_cost, mask, last_vertex, path)
    # We use a priority queue ordered by bottleneck cost
    # mask: bitmask of visited vertices
    # last: last vertex in the path
    # path: list of vertices (for reconstruction)
    
    all_mask = (1 << n) - 1
    
    # best_cost[mask][last] = minimum bottleneck cost to reach this state
    # Using a dict for sparse access
    best_cost = {}
    
    # Priority queue: (bottleneck_cost, mask, last_vertex, path_tuple)
    # We store path as a tuple for immutability, but this is memory-heavy
    # Alternative: store parent pointers for reconstruction
    # Let's use parent pointers: parent[(mask, last)] = (prev_mask, prev_last)
    
    parent = {}  # (mask, last) -> (prev_mask, prev_last)
    
    # Start state: mask=1 (vertex 0 visited), last=0, cost=0
    start_mask = 1
    start_last = 0
    best_cost[(start_mask, start_last)] = 0
    parent[(start_mask, start_last)] = None
    
    # Heap: (bottleneck_cost, mask, last)
    heap = [(0, start_mask, start_last)]
    
    while heap:
        if time.time() > deadline:
            return None, None
        
        cost, mask, last = heapq.heappop(heap)
        
        # Check if this is the goal state
        if mask == all_mask:
            # Check if last can connect back to 0
            close_cost = edge_pos[last][0]
            if close_cost <= cost or cost == 0:
                # Found a valid Hamiltonian cycle
                # Reconstruct path
                path = []
                state = (mask, last)
                while state is not None:
                    _, v = state
                    path.append(v)
                    state = parent[state]
                path.reverse()
                # The bottleneck cost of the cycle is max(cost, close_cost)
                final_cost = max(cost, close_cost) if cost > 0 else close_cost
                return final_cost, path
        
        # If we've seen this state with a lower or equal cost, skip
        if best_cost.get((mask, last), float('inf')) < cost:
            continue
        
        # Expand: try all neighbors of last that are not yet visited
        for v in range(n):
            if mask & (1 << v):
                continue  # already visited
            
            edge_cost = edge_pos[last][v]
            if edge_cost == -1:
                continue  # no edge (shouldn't happen in complete graph)
            
            new_cost = max(cost, edge_cost)
            new_mask = mask | (1 << v)
            new_state = (new_mask, v)
            
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent[new_state] = (mask, last)
                heapq.heappush(heap, (new_cost, new_mask, v))
    
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Fallback: identity permutation
    identity = list(range(n))
    best_score = score_cycle(identity)
    best_cycle = identity
    
    # Try Dijkstra bottleneck search
    score, cycle = dijkstra_bottleneck_hc(deadline)
    
    if cycle is not None and score < best_score:
        best_score = score
        best_cycle = cycle
    
    return best_cycle

# EVOLVE-BLOCK-END
