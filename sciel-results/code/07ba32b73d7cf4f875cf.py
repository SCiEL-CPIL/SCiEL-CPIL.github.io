import time
import sys
import heapq

# EVOLVE-BLOCK-START

n = 24
edge_order = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute arrival index (0-based) for each edge
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    key = (min(u, v), max(u, v))
    arrival[key] = idx

def get_arrival(u, v):
    key = (min(u, v), max(u, v))
    return arrival[key]

def compute_score(cycle):
    """Compute the score (max 1-based arrival index) of a Hamilton cycle."""
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = get_arrival(u, v) + 1  # 1-based
        if idx > max_idx:
            max_idx = idx
    return max_idx

def find_min_bottleneck_cycle(time_limit):
    """
    Find a Hamiltonian cycle minimizing the maximum edge arrival index
    using Dijkstra's algorithm on the state space (visited_mask, last_vertex).
    Correctly handles the closing edge by continuing search until no better 
    cycle bottleneck is possible.
    """
    deadline = time.time() + time_limit
    
    # Cost matrix: cost[u][v] = arrival index of edge (u,v) (0-based)
    cost = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = get_arrival(i, j)
            cost[i][j] = c
            cost[j][i] = c
    
    # State: (bottleneck_cost, visited_mask, last_vertex)
    # Start from vertex 0
    start_mask = 1  # bit 0 set
    start_state = (0, start_mask, 0)
    
    # Dijkstra's priority queue
    pq = [start_state]
    # best[mask][last] = minimum bottleneck cost to reach (mask, last)
    # Use a dict for sparse storage
    best = {}
    best[(start_mask, 0)] = 0
    
    full_mask = (1 << n) - 1
    parent = {}  # parent[(mask, last)] = (prev_mask, prev_last)
    
    best_cycle_score = float('inf')
    best_cycle_path = None
    
    while pq:
        if time.time() > deadline:
            break
        
        b, mask, last = heapq.heappop(pq)
        
        # If the current bottleneck is already >= best cycle score found, 
        # we can stop because all subsequent states will have bottleneck >= b
        if b >= best_cycle_score:
            break
        
        # Check if this is better than what we've seen for this state
        if (mask, last) in best and best[(mask, last)] < b:
            continue
        
        # Goal: all vertices visited
        if mask == full_mask:
            # Check if edge (last, 0) exists and compute final bottleneck
            final_b = max(b, cost[last][0])
            
            if final_b < best_cycle_score:
                best_cycle_score = final_b
                # Reconstruct path
                path = []
                cur_mask, cur_last = mask, last
                while (cur_mask, cur_last) in parent:
                    path.append(cur_last)
                    cur_mask, cur_last = parent[(cur_mask, cur_last)]
                path.append(0)
                path.reverse()
                best_cycle_path = path
        
        # Expand: try adding each unvisited neighbor
        for nxt in range(n):
            if mask & (1 << nxt):
                continue
            if nxt == 0:
                continue  # Can't revisit start until the end
            
            new_cost = max(b, cost[last][nxt])
            
            # Prune if new_cost is already >= best_cycle_score
            if new_cost >= best_cycle_score:
                continue
                
            new_mask = mask | (1 << nxt)
            new_state = (new_mask, nxt)
            
            if new_state not in best or best[new_state] > new_cost:
                best[new_state] = new_cost
                parent[new_state] = (mask, last)
                heapq.heappush(pq, (new_cost, new_mask, nxt))
    
    return best_cycle_path, best_cycle_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Default cycle: 0-1-2-...-23-0
    default_cycle = list(range(n))
    best_score = compute_score(default_cycle)
    best_cycle = default_cycle
    
    # Try Dijkstra for minimum bottleneck Hamilton cycle
    if time.time() < deadline - 5:
        cycle, score = find_min_bottleneck_cycle(deadline - start_time - 5)
        if cycle is not None and score < best_score:
            best_score = score
            best_cycle = cycle
    
    return best_cycle

# EVOLVE-BLOCK-END
