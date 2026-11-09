"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    n = P['n']
    edge_order = P['edge_order']
    num_edges = len(edge_order)
    DEADLINE = 85.0  # 5s safety margin
    
    # Precompute edge positions
    edge_pos = {}
    for i, (u, v) in enumerate(edge_order):
        edge_pos[(u, v)] = i
        edge_pos[(v, u)] = i
    
    # Precompute adjacency with positions for quick lookup
    adj_pos = [[None] * n for _ in range(n)]
    for i, (u, v) in enumerate(edge_order):
        adj_pos[u][v] = i
        adj_pos[v][u] = i

    def get_cycle_score(cycle):
        """Return the one-based max edge index in the cycle."""
        max_idx = 0
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            idx = edge_pos[(u, v)]
            if idx + 1 > max_idx:
                max_idx = idx + 1
        return max_idx

    def two_opt(cycle, threshold, max_iters=1000):
        """
        Try to improve the cycle by 2-opt moves, keeping all edges < threshold.
        Returns improved cycle or original if no improvement.
        """
        cycle = list(cycle)
        current_score = get_cycle_score(cycle)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > DEADLINE:
                break
            improved = False
            iters += 1
            for i in range(n - 1):
                for j in range(i + 2, n - 1):
                    a, b = cycle[i], cycle[i + 1]
                    c, d = cycle[j], cycle[j + 1]
                    if a == c or a == d or b == c or b == d:
                        continue
                    
                    pos_ac = adj_pos[a][c]
                    pos_bd = adj_pos[b][d]
                    if pos_ac is None or pos_bd is None:
                        continue
                    if pos_ac >= threshold or pos_bd >= threshold:
                        continue
                    
                    old_score = max(adj_pos[a][b], adj_pos[c][d])
                    new_score = max(pos_ac, pos_bd)
                    
                    if new_score < old_score:
                        cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
                        improved = True
        return cycle

    def greedy_build(threshold):
        """
        Build a Hamiltonian path greedily, then try to close it.
        Returns a cycle or None.
        """
        adj = [[] for _ in range(n)]
        for i in range(threshold):
            u, v = edge_order[i]
            adj[u].append(v)
            adj[v].append(u)
        
        for u in range(n):
            adj[u].sort(key=lambda v: adj_pos[u][v])
        
        for start in range(n):
            path = [start]
            visited = [False] * n
            visited[start] = True
            
            for _ in range(n - 1):
                last = path[-1]
                best_next = None
                best_pos = float('inf')
                for next_v in adj[last]:
                    if not visited[next_v]:
                        pos = adj_pos[last][next_v]
                        if pos < threshold and pos < best_pos:
                            best_pos = pos
                            best_next = next_v
                
                if best_next is None:
                    break
                
                path.append(best_next)
                visited[best_next] = True
            
            if len(path) == n:
                last = path[-1]
                pos = adj_pos[last][start]
                if pos is not None and pos < threshold:
                    return path
        
        return None

    def find_hamiltonian_cycle_bt(threshold, time_limit=0.5):
        """
        Find a Hamiltonian cycle using backtracking with strong pruning.
        Returns the cycle (list of vertices) or None if not found.
        """
        adj = [[] for _ in range(n)]
        for i in range(threshold):
            u, v = edge_order[i]
            adj[u].append(v)
            adj[v].append(u)
        
        for u in range(n):
            # Sort by degree (Warnsdorff) then by position
            adj[u].sort(key=lambda v: (len(adj[v]), adj_pos[u][v]))
        
        start = 0
        path = [start]
        visited = [False] * n
        visited[start] = True
        local_start = time.time()
        
        # Precompute degrees
        deg = [len(adj[u]) for u in range(n)]
        
        def is_dead_end(v, remaining):
            """Check if vertex v has no unvisited neighbors (dead end)."""
            for nb in adj[v]:
                if not visited[nb] and nb != start:
                    return False
            # If v is the last vertex, check if it connects to start
            if remaining == 1:
                pos = adj_pos[v][start]
                return pos is None or pos >= threshold
            return True
        
        def backtrack():
            if time.time() - local_start > time_limit:
                return False
            if len(path) == n:
                last = path[-1]
                pos = adj_pos[last][start]
                if pos is not None and pos < threshold:
                    return True
                return False
            
            last = path[-1]
            remaining = n - len(path)
            
            # Try neighbors in order
            for next_v in adj[last]:
                if visited[next_v]:
                    continue
                if next_v == start and remaining > 1:
                    continue
                
                # Dead-end check: if next_v would leave other unvisited vertices isolated
                path.append(next_v)
                visited[next_v] = True
                
                if backtrack():
                    return True
                
                path.pop()
                visited[next_v] = False
            
            return False
        
        if backtrack():
            return path
        return None

    def assignment_based_cycle(threshold):
        """
        Use minimum-cost perfect matching (assignment) to find a 2-factor,
        then try to merge cycles into a Hamiltonian cycle.
        Returns a cycle or None.
        """
        INF = float('inf')
        cost = np.full((n, n), INF, dtype=float)
        for i in range(threshold):
            u, v = edge_order[i]
            cost[u][v] = i
            cost[v][u] = i
        np.fill_diagonal(cost, 1e9)
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        for r, c in zip(row_ind, col_ind):
            if cost[r][c] >= threshold:
                return None
        
        next_v = {}
        for r, c in zip(row_ind, col_ind):
            next_v[r] = c
        
        visited = [False] * n
        cycles = []
        for start in range(n):
            if not visited[start]:
                cycle = []
                cur = start
                while not visited[cur]:
                    visited[cur] = True
                    cycle.append(cur)
                    cur = next_v[cur]
                cycles.append(cycle)
        
        if len(cycles) == 1:
            return cycles[0]
        
        # Merge cycles
        merged = True
        while merged and time.time() < DEADLINE:
            merged = False
            for i in range(len(cycles)):
                if merged:
                    break
                for j in range(i + 1, len(cycles)):
                    if merged:
                        break
                    c1 = cycles[i]
                    c2 = cycles[j]
                    
                    # Try all edge swap combinations
                    for ai in range(len(c1)):
                        if merged:
                            break
                        a = c1[ai]
                        b = c1[(ai + 1) % len(c1)]
                        for ci in range(len(c2)):
                            if merged:
                                break
                            c = c2[ci]
                            d = c2[(ci + 1) % len(c2)]
                            
                            # Option 1: a->c and b->d
                            pos_ac = adj_pos[a][c]
                            pos_bd = adj_pos[b][d]
                            if pos_ac is not None and pos_bd is not None and \
                               pos_ac < threshold and pos_bd < threshold:
                                seg1 = c1[ai+1:] + c1[:ai+1]
                                seg2 = c2[ci:] + c2[:ci]
                                new_cycle = seg1 + seg2
                                if len(new_cycle) == len(c1) + len(c2):
                                    cycles[i] = new_cycle
                                    del cycles[j]
                                    merged = True
                                    break
                            
                            # Option 2: a->d and b->c
                            pos_ad = adj_pos[a][d]
                            pos_bc = adj_pos[b][c]
                            if pos_ad is not None and pos_bc is not None and \
                               pos_ad < threshold and pos_bc < threshold:
                                # a->d and b->c: merge differently
                                # c1 path: b -> ... -> a
                                # c2 path: c -> ... -> d
                                # New edges: a->d, b->c
                                # Cycle: b -> ... -> a -> d -> ... -> c -> b
                                seg1 = c1[ai+1:] + c1[:ai+1]  # b, ..., a
                                seg2 = c2[ci+1:] + c2[:ci+1]  # d, ..., c
                                new_cycle = seg1 + seg2
                                if len(new_cycle) == len(c1) + len(c2):
                                    cycles[i] = new_cycle
                                    del cycles[j]
                                    merged = True
                                    break
        
        if len(cycles) == 1:
            return cycles[0]
        return None

    def find_cycle_any_method(threshold, time_limit=1.0):
        """Try multiple methods to find a cycle within threshold."""
        # Method 1: Greedy
        cycle = greedy_build(threshold)
        if cycle:
            return cycle
        
        # Method 2: Assignment-based
        cycle = assignment_based_cycle(threshold)
        if cycle:
            return cycle
        
        # Method 3: Backtracking with time limit
        cycle = find_hamiltonian_cycle_bt(threshold, time_limit)
        if cycle:
            return cycle
        
        return None

    # Linear scan from lower bound to find minimum threshold
    best_cycle = None
    best_threshold = num_edges
    
    # First, get any valid cycle with full edge set
    full_cycle = find_cycle_any_method(num_edges, 2.0)
    if not full_cycle:
        return list(range(n))
    
    best_cycle = full_cycle
    best_threshold = num_edges
    best_score = get_cycle_score(full_cycle)
    
    # Linear scan from n upward
    for t in range(n, best_threshold):
        if time.time() > DEADLINE - 5:
            break
        
        cycle = find_cycle_any_method(t, 0.8)
        if cycle:
            score = get_cycle_score(cycle)
            if score <= t:
                best_cycle = cycle
                best_threshold = t
                best_score = score
                break
    
    # Polish with 2-opt
    if best_cycle and time.time() < DEADLINE - 3:
        improved = two_opt(best_cycle, best_threshold, max_iters=2000)
        if get_cycle_score(improved) < best_score:
            best_cycle = improved
            best_score = get_cycle_score(improved)
            best_threshold = best_score
    
    # Try to find even better threshold
    if best_cycle and time.time() < DEADLINE - 3:
        for t in range(best_threshold - 1, max(n - 1, best_threshold - 15), -1):
            if time.time() > DEADLINE - 2:
                break
            cycle = find_cycle_any_method(t, 0.5)
            if cycle:
                score = get_cycle_score(cycle)
                if score < best_score:
                    improved = two_opt(cycle, t, max_iters=1000)
                    if get_cycle_score(improved) < score:
                        cycle = improved
                        score = get_cycle_score(cycle)
                    best_cycle = cycle
                    best_threshold = t
                    best_score = score
                break
    
    return best_cycle

# EVOLVE-BLOCK-END
