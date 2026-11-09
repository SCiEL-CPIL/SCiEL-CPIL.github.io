import random
import time
import math

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    # Build position map for O(1) lookup
    edge_pos = {}
    for idx, (u, v) in enumerate(P['edge_order'], 1):
        if u > v:
            u, v = v, u
        edge_pos[(u, v)] = idx
    
    def get_pos(u, v):
        if u > v:
            u, v = v, u
        return edge_pos.get((u, v), 1000)

    def calc_cost(perm):
        max_pos = 0
        for i in range(n):
            u = perm[i]
            v = perm[(i + 1) % n]
            p = get_pos(u, v)
            if p > max_pos:
                max_pos = p
        return max_pos

    def two_opt(perm, time_limit):
        """
        Applies 2-opt moves to minimize the maximum edge weight.
        """
        current_cost = calc_cost(perm)
        start_time = time.time()
        
        # Precompute edge positions for the current cycle to speed up evaluation?
        # Or just recalc. n=24 is small.
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # Try all 2-opt moves
            for i in range(n - 1):
                u1 = perm[i]
                v1 = perm[i+1]
                pos1 = get_pos(u1, v1)
                
                for j in range(i + 2, n):
                    # If j is n-1, the edge is (perm[n-1], perm[0])
                    # Otherwise edge is (perm[j], perm[j+1])
                    u2 = perm[j]
                    v2 = perm[(j+1) % n]
                    pos2 = get_pos(u2, v2)
                    
                    # New edges: (u1, u2) and (v1, v2)
                    # Note: 2-opt reverses segment i+1..j
                    # Original edges removed: (u1, v1) and (u2, v2)
                    # New edges added: (u1, u2) and (v1, v2)
                    
                    new_pos1 = get_pos(u1, u2)
                    new_pos2 = get_pos(v1, v2)
                    
                    # The new max cost is max( max of all other edges, new_pos1, new_pos2 )
                    # We don't know "max of all other edges" easily without recalc or maintaining a structure.
                    # Given n=24, full recalc is 24 ops. 2-opt loop is ~24^2/2 = 288 moves.
                    # 288 * 24 = ~7000 ops per iteration. Very fast.
                    
                    # Let's just recalc for simplicity and correctness first.
                    new_perm = perm[:]
                    new_perm[i+1:j+1] = reversed(new_perm[i+1:j+1])
                    new_cost = calc_cost(new_perm)
                    
                    delta = new_cost - current_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
                        best_perm = new_perm
                    
            if best_i != -1:
                perm = best_perm
                current_cost = current_cost + best_delta
                improved = True
                
        return perm, current_cost

    # Strategy: Greedy construction + Local Search
    # We will run several attempts with different random starting points
    # and keep the best result.
    
    rng = random.Random(614)
    best_overall_cost = float('inf')
    best_overall_perm = list(range(n))
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s margin
    
    # Number of attempts
    # Each attempt: Greedy (O(n^2)) + 2-Opt (O(n^3) approx but small constant)
    # n=24. 2-Opt is very fast.
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        # Greedy Construction
        # Start with a random vertex
        start_v = rng.randint(0, n-1)
        current_path = [start_v]
        unvisited = set(range(n))
        unvisited.remove(start_v)
        
        # Greedy step: pick next vertex that minimizes the max edge weight in the path?
        # Actually, to minimize the *maximum* edge weight, we should pick the edge with the smallest weight possible.
        # This is like building a minimum spanning tree path?
        # Let's try: pick the unvisited vertex u that minimizes get_pos(last, u).
        
        while unvisited:
            last = current_path[-1]
            best_u = None
            best_w = float('inf')
            for u in unvisited:
                w = get_pos(last, u)
                if w < best_w:
                    best_w = w
                    best_u = u
            if best_u is None:
                break
            current_path.append(best_u)
            unvisited.remove(best_u)
        
        # Close the cycle
        # Check if closing edge is valid (it always is in complete graph)
        # But we might want to check if it's a good close.
        # For now, just close it.
        
        # Local Search
        perm, cost = two_opt(current_path, time_limit - (time.time() - start_time))
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_perm = perm[:]
            
        # If we found a very low cost, we can stop early?
        # Theoretical minimum is 24 (if all edges are in first 24 positions).
        # Unlikely.
        
        # Add some randomization to the greedy step to escape local optima in construction?
        # For now, just change start vertex.
        
    return best_overall_perm
# EVOLVE-BLOCK-END
