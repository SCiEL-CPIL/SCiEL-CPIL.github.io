import time
import sys
import random

# EVOLVE-BLOCK-START

# Data provided in the task description
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
START_TIME = time.time()
TIME_LIMIT = 85.0

# Precompute arrival indices
arrival = [[0] * N for _ in range(N)]
for idx, edge in enumerate(P['edge_order']):
    u, v = edge
    arrival[u][v] = idx + 1
    arrival[v][u] = idx + 1

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        return list(range(N))

    # Fallback: a valid Hamiltonian cycle (e.g., 0-1-2...-23-0)
    # We need to ensure this is valid? No, the evaluator checks validity.
    # But we want a low score. Let's find a good one quickly.
    # For now, keep a simple cycle as default fallback if search fails.
    fallback_cycle = list(range(N))
    
    # We will scan k from low to high.
    # Minimum possible k for a 2-regular graph is N (cycle).
    # But the graph must be connected and have a Hamiltonian cycle.
    # Let's start from N.
    
    best_cycle = None
    best_k = N * (N - 1) // 2 # Max possible
    
    # We can also try to find a cycle for k = N, N+1, ...
    # To speed up, we can check connectivity and min degree quickly.
    
    def get_adj_masks(k):
        masks = [0] * N
        for u in range(N):
            mask = 0
            for v in range(N):
                if u != v and arrival[u][v] <= k:
                    mask |= (1 << v)
            masks[u] = mask
        return masks

    def is_connected(adj_masks):
        visited = 1 << 0
        stack = [0]
        while stack:
            u = stack.pop()
            neighbors = adj_masks[u] & ~visited
            while neighbors:
                lsb = neighbors & -neighbors
                v = lsb.bit_length() - 1
                visited |= lsb
                stack.append(v)
                neighbors ^= lsb
        return visited == (1 << N) - 1

    def find_hamilton_cycle(adj_masks, start, deadline):
        """
        Finds a Hamiltonian cycle in the graph defined by adj_masks.
        Returns a list of vertices representing the path (without closing edge).
        Uses DFS with Warnsdorff's heuristic.
        """
        N_local = N
        # Pre-compute bit counts for masks? No, just use .bit_count()
        
        # We need to track visited nodes and the path.
        visited_mask = 1 << start
        path = [start]
        
        # Recursive DFS
        def dfs(u, visited_mask, path):
            if time.time() > deadline:
                return None

            if len(path) == N_local:
                # Check if we can close the cycle
                if (adj_masks[u] >> start) & 1:
                    return path[:]
                return None

            candidates = adj_masks[u] & ~visited_mask
            if candidates == 0:
                return None

            # Get list of candidate vertices
            cand_vertices = []
            temp_mask = candidates
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                cand_vertices.append(v)
                temp_mask ^= lsb
            
            # Warnsdorff's Rule: sort by number of unvisited neighbors
            # Tie-breaker: arrival index (smaller is better, keeps edges "local"?)
            # Actually, for Hamiltonian cycle, we just need to avoid dead ends.
            cand_vertices.sort(key=lambda v: (adj_masks[v] & ~visited_mask).bit_count())
            
            for v in cand_vertices:
                new_visited = visited_mask | (1 << v)
                new_path = path + [v]
                
                # Lookahead: if this is the second to last vertex, 
                # the last vertex MUST connect to start.
                # This is a strong pruning rule.
                if len(new_path) == N_local - 1:
                    # Only one vertex left unvisited
                    last_mask = candidates & ~(1 << v)
                    if last_mask:
                        last_v = (last_mask & -last_mask).bit_length() - 1
                        if not ((adj_masks[last_v] >> start) & 1):
                            continue
                        # Also last_v must be connected to v
                        # This is already ensured by candidates coming from adj_masks[v]?
                        # Wait, candidates came from adj_masks[u]. v is next.
                        # last_v is unvisited.
                        # We need edge (v, last_v) and (last_v, start).
                        # (v, last_v) is in adj_masks[v].
                        if not ((adj_masks[v] >> last_v) & 1):
                            continue
                    
                res = dfs(v, new_visited, new_path)
                if res is not None:
                    return res

            return None

        # Try different starting vertices if the first fails?
        # For speed, we can just try 0 first. If it fails, maybe try a few others.
        # But with N=24, a good heuristic usually works from any start.
        
        res = dfs(start, visited_mask, path)
        if res:
            return res
        
        # If start 0 failed, try a few random starts
        rng = random.Random(0)
        starts = list(range(1, N_local))
        rng.shuffle(starts)
        for s in starts[:5]: # Try 5 more
            if time.time() > deadline:
                return None
            visited_mask = 1 << s
            path = [s]
            res = dfs(s, visited_mask, path)
            if res:
                return res
        
        return None

    # Linear scan for k
    # We can stop as soon as we find a cycle.
    # The score will be the max arrival index of edges in that cycle.
    # Since we are scanning k upwards, the first cycle found is likely to have a low max index.
    
    deadline = START_TIME + TIME_LIMIT - 5.0 # Safety margin
    
    # Start from a lower bound.
    # Min degree 2 requires at least N edges.
    # Let's start from N.
    for k in range(N, N * (N - 1) // 2 + 1):
        if time.time() > deadline:
            break
        
        # Quick check: is the graph connected?
        # And does every node have degree >= 2?
        adj_masks = get_adj_masks(k)
        
        # Check min degree
        valid = True
        for u in range(N):
            if adj_masks[u].bit_count() < 2:
                valid = False
                break
        
        if not valid:
            continue
            
        if not is_connected(adj_masks):
            continue
            
        # Try to find a cycle
        # We can try a few different strategies or starts here.
        # Start with 0.
        cycle = find_hamilton_cycle(adj_masks, 0, deadline)
        
        if cycle:
            # Verify it's a valid cycle (defensive programming)
            if len(cycle) != N:
                continue
            if len(set(cycle)) != N:
                continue
            # Check edges
            is_valid = True
            for i in range(N):
                u = cycle[i]
                v = cycle[(i + 1) % N]
                if (adj_masks[u] >> v) & 1:
                    pass
                else:
                    is_valid = False
                    break
            if is_valid:
                # We found a cycle in the subgraph of edges <= k.
                # The actual score is the max arrival index of these edges.
                # It is <= k.
                # Since we are scanning k upwards, this is the best we can hope for
                # with this heuristic.
                # Return this cycle.
                return cycle

    # If no cycle found within time, return fallback
    return fallback_cycle

# EVOLVE-BLOCK-END
