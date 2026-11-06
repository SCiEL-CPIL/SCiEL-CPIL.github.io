"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _build_edge_positions():
    edge_order = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]
    pos = {}
    for i, (u, v) in enumerate(edge_order):
        if u > v:
            u, v = v, u
        pos[(u, v)] = i  # 0-indexed position
    return pos, edge_order

def _find_ham_cycle_k(pos, k, n=24, time_limit=55):
    """Find Hamilton cycle using only edges with 0-indexed position <= k-1 (i.e., 1-indexed <= k).
    Returns the cycle as list of vertices, or None."""
    # Build adjacency: adj[v] = list of (neighbor, pos) for edges with pos < k
    adj = [[] for _ in range(n)]
    for (u, v), p in pos.items():
        if p < k:
            adj[u].append((v, p))
            adj[v].append((u, p))
    
    # Sort adjacency by position (earliest first) for better search
    for v in range(n):
        adj[v].sort(key=lambda x: x[1])
    
    # Quick check: every vertex must have degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Backtracking with pruning
    start = 0
    path = [start]
    used = [False] * n
    used[start] = True
    best_cycle = [None]
    
    deadline = time.time() + time_limit
    
    def backtrack():
        if time.time() > deadline:
            return
        if len(path) == n:
            # Check if last vertex connects back to start
            last = path[-1]
            for (nb, p) in adj[last]:
                if nb == start:
                    best_cycle[0] = path[:]
                    return
            return
        
        cur = path[-1]
        # Try neighbors in order of position
        for (nb, p) in adj[cur]:
            if used[nb]:
                continue
            # Pruning: if nb is not start and we're not at the end, 
            # nb must have at least one other available neighbor (besides cur)
            # unless nb is the last vertex to be placed
            if len(path) < n - 1:
                # nb needs degree >= 2 in the remaining graph (including edge to cur)
                # Count available neighbors of nb (not used, or is start if we're closing)
                avail = 0
                for (nb2, p2) in adj[nb]:
                    if nb2 == cur:
                        continue
                    if not used[nb2] or nb2 == start:
                        avail += 1
                if avail < 1:  # needs at least one more connection (besides cur)
                    continue
            
            used[nb] = True
            path.append(nb)
            backtrack()
            if best_cycle[0] is not None:
                return
            path.pop()
            used[nb] = False
    
    backtrack()
    return best_cycle[0]

def _score_cycle(cycle, pos, n=24):
    """Compute the max 1-indexed position of edges in the cycle."""
    max_pos = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        a, b = min(u, v), max(u, v)
        p = pos[(a, b)] + 1  # 1-indexed
        if p > max_pos:
            max_pos = p
    return max_pos

def construct():
    start_time = time.time()
    pos, edge_order = _build_edge_positions()
    n = 24
    total_edges = len(edge_order)
    
    # First, compute the score of the seed cycle
    seed_cycle = list(range(n))
    seed_score = _score_cycle(seed_cycle, pos, n)
    
    best_cycle = seed_cycle
    best_score = seed_score
    
    # Binary search / linear search on k
    # Lower bound: need at least n edges, and min degree >= 2
    # Try k from a reasonable lower bound up to seed_score
    # Lower bound: the (n-1)th smallest edge position + 1, but really we need a cycle
    
    # Let's try values of k
    # The minimum possible k is at least the position of the 24th edge that forms a cycle
    # Let's just try from 24 upward, but with a time limit
    
    # First, find a good lower bound by checking connectivity
    # For efficiency, try a range of k values
    
    # Let's try k values: start from 24 and go up
    # But be smart: use the structure of the edge list
    
    # Try increasing k
    found = False
    for k in range(24, min(seed_score + 1, total_edges + 1)):
        if time.time() - start_time > 50:
            break
        cycle = _find_ham_cycle_k(pos, k, n, time_limit=5)
        if cycle is not None:
            score = _score_cycle(cycle, pos, n)
            if score < best_score:
                best_score = score
                best_cycle = cycle
                found = True
            # If we found one at k, the score should be <= k
            # But let's verify
            if score <= k:
                # This is the minimum k, so this is optimal (or close)
                break
    
    # If we didn't find via the binary approach, try local search
    if time.time() - start_time < 55:
        # Try a few random restarts with local search
        import random
        rng = random.Random(42)
        for attempt in range(20):
            if time.time() - start_time > 55:
                break
            # Random cycle
            cycle = list(range(n))
            rng.shuffle(cycle)
            cur_score = _score_cycle(cycle, pos, n)
            if cur_score < best_score:
                best_score = cur_score
                best_cycle = cycle
            # Local search: 2-opt
            improved = True
            while improved and time.time() - start_time < 55:
                improved = False
                for i in range(n):
                    for j in range(i + 1, n):
                        if j == i + 1 or (i == 0 and j == n - 1):
                            continue
                        # 2-opt: reverse segment i+1..j
                        new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                        new_score = _score_cycle(new_cycle, pos, n)
                        if new_score < cur_score:
                            cur_score = new_score
                            cycle = new_cycle
                            improved = True
                            if cur_score < best_score:
                                best_score = cur_score
                                best_cycle = cycle
                            break
                    if improved:
                        break
    
    return best_cycle
# EVOLVE-BLOCK-END
