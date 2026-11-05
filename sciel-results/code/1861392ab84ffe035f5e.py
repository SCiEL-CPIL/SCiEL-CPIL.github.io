import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def is_valid_path(path):
        if not path or len(path) < 1:
            return False
        seen = set(path)
        if len(seen) != len(path):
            return False
        for i in range(len(path) - 1):
            if path[i + 1] not in adj[path[i]]:
                return False
        return True
    
    def greedy_path(start, strategy='min_degree'):
        path = [start]
        seen = {start}
        while True:
            candidates = adj[path[-1]] - seen
            if not candidates:
                break
            if strategy == 'min_degree':
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            elif strategy == 'max_degree':
                v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
            elif strategy == 'random':
                v = random.choice(list(candidates))
            else:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        return path
    
    def try_extend(path, adj, seen):
        """Try to extend path from both ends."""
        changed = True
        while changed:
            changed = False
            # Try extending from end
            if path:
                last = path[-1]
                candidates = adj[last] - seen
                if candidates:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                    path.append(v)
                    seen.add(v)
                    changed = True
            # Try extending from start
            if path:
                first = path[0]
                candidates = adj[first] - seen
                if candidates:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                    path.insert(0, v)
                    seen.add(v)
                    changed = True
        return path
    
    def local_improve(path, adj, max_iters=500):
        """Try to improve path by swapping segments or reordering."""
        best = list(path)
        best_len = len(best)
        seen = set(best)
        
        for _ in range(max_iters):
            improved = False
            # Try 2-opt style: reverse a segment
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    # Check if reversing best[i..j] is valid
                    # best[i-1] -- best[j] and best[i] -- best[j+1] must be edges (if applicable)
                    valid = True
                    if i > 0 and best[j] not in adj[best[i-1]]:
                        valid = False
                    if valid and j < len(best) - 1 and best[i] not in adj[best[j+1]]:
                        valid = False
                    if valid:
                        new_path = best[:i] + best[i:j+1][::-1] + best[j+1:]
                        if len(new_path) > best_len:
                            best = new_path
                            best_len = len(best)
                            seen = set(best)
                            improved = True
                            break
                if improved:
                    break
            
            # Try to extend from both ends
            extended = try_extend(best, adj, set(best))
            if len(extended) > best_len:
                best = extended
                best_len = len(best)
                seen = set(best)
                improved = True
            
            if not improved:
                break
        
        return best
    
    def hungarian_algorithm(cost_matrix):
        """
        Solve assignment problem using Hungarian algorithm.
        cost_matrix: n x n matrix where we want to minimize total cost.
        Returns: list of column assignments for each row (row i assigned to column result[i])
        """
        import numpy as np
        n_mat = len(cost_matrix)
        if n_mat == 0:
            return []
        
        # Convert to numpy for efficiency
        cost = np.array(cost_matrix, dtype=float)
        
        # Hungarian algorithm implementation
        # u[i], v[j] are potentials
        # p[j] = row assigned to column j, or -1 if unassigned
        # way[j] = previous column in augmenting path
        
        INF = float('inf')
        u = np.zeros(n_mat + 1)
        v = np.zeros(n_mat + 1)
        p = np.zeros(n_mat + 1, dtype=int)
        way = np.zeros(n_mat + 1, dtype=int)
        
        for i in range(1, n_mat + 1):
            p[0] = i
            j0 = 0
            minv = np.full(n_mat + 1, INF)
            used = np.zeros(n_mat + 1, dtype=bool)
            
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = INF
                j1 = -1
                
                for j in range(1, n_mat + 1):
                    if not used[j]:
                        cur = cost[i0-1, j-1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                
                for j in range(n_mat + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                
                j0 = j1
                if p[j0] == 0:
                    break
            
            while j0 != 0:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
        
        # Result: p[j] = row assigned to column j
        result = [0] * n_mat
        for j in range(1, n_mat + 1):
            if p[j] > 0:
                result[p[j]-1] = j-1
        
        return result
    
    def max_weight_matching_for_path():
        """
        Use Maximum Weight Matching formulation to find a long path.
        
        Transform: Split each vertex v into two copies v_in and v_out.
        - Edge v_in -> v_out has weight M (large constant) if v is used
        - Edge u_out -> v_in has weight 1 if (u,v) is an edge in original graph
        - We want to find a maximum weight matching that forms a valid path
        
        Actually, for longest path, we use a different approach:
        Find maximum weight matching in a bipartite graph where:
        - Left side: vertices 0..n-1 (representing "outgoing" from vertex)
        - Right side: vertices 0..n-1 (representing "incoming" to vertex)
        - Edge (i, j) with weight 1 if i != j and (i,j) is an edge
        - Additionally, we can have "self-loops" (i, i) with weight 0 to allow unused vertices
        
        A matching M gives us a set of directed edges. We need to extract a path from this.
        The matched edges form a collection of paths and cycles. We take the longest path.
        """
        M = n + 1  # Large weight for using a vertex
        
        # Build cost matrix for Hungarian (minimize cost = maximize weight)
        # cost[i][j] = -weight[i][j]
        cost_matrix = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self-loop: vertex i not used as an internal node
                    # Weight 0 (cost 0)
                    cost_matrix[i][j] = 0
                elif j in adj[i]:
                    # Edge i->j exists
                    # Weight 1 for the edge
                    cost_matrix[i][j] = -1
                else:
                    # No edge: prohibit with large cost
                    cost_matrix[i][j] = M
        
        # Solve assignment problem
        assignment = hungarian_algorithm(cost_matrix)
        
        # Extract path from matching
        # assignment[i] = j means vertex i has outgoing edge to vertex j
        # Build directed graph from matching
        next_vertex = [-1] * n
        prev_vertex = [-1] * n
        for i in range(n):
            j = assignment[i]
            if i != j:  # Not a self-loop
                next_vertex[i] = j
                prev_vertex[j] = i
        
        # Find all paths in the directed graph formed by matching
        # Each vertex has at most one incoming and one outgoing edge
        visited = [False] * n
        best_path = []
        
        for start in range(n):
            if visited[start]:
                continue
            # Start a path from start if it has no incoming edge
            if prev_vertex[start] != -1:
                continue
            
            # Follow the path
            path = [start]
            current = start
            while next_vertex[current] != -1:
                current = next_vertex[current]
                if visited[current]:
                    break
                path.append(current)
            
            # Mark all vertices in this path as visited
            for v in path:
                visited[v] = True
            
            if len(path) > len(best_path):
                best_path = path
        
        return best_path
    
    # Phase 1: Generate many candidate paths using different strategies
    candidates = []
    
    # Greedy from every vertex with different strategies
    for start in range(n):
        for strategy in ['min_degree', 'max_degree']:
            path = greedy_path(start, strategy)
            if path:
                candidates.append(path)
    
    # Random restarts
    rng = random.Random(42)
    for _ in range(200):
        start = rng.randint(0, n - 1)
        path = greedy_path(start, 'random')
        if path:
            candidates.append(path)
    
    # Phase 2: Maximum Weight Matching approach
    if time.time() < deadline - 30:
        mw_path = max_weight_matching_for_path()
        if mw_path and len(mw_path) > 1:
            candidates.append(mw_path)
    
    # Phase 3: Local improvement on top candidates
    candidates.sort(key=lambda p: len(p), reverse=True)
    
    best_path = candidates[0] if candidates else [0]
    best_len = len(best_path)
    
    # Improve top 50 candidates
    for path in candidates[:50]:
        if len(path) <= best_len - 5:  # Skip if much shorter
            continue
        improved = local_improve(path, adj, max_iters=200)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
        if time.time() > deadline:
            break
    
    # Phase 4: More aggressive search - try to find longer paths
    # Use beam search
    beam_width = 50
    beam = []
    for start in range(n):
        beam.append(([start], {start}))
    
    beam.sort(key=lambda x: len(x[0]), reverse=True)
    beam = beam[:beam_width]
    
    for step in range(30):  # Max path length we search for
        new_beam = []
        for path, seen in beam:
            last = path[-1]
            for v in adj[last] - seen:
                new_path = path + [v]
                new_seen = seen | {v}
                new_beam.append((new_path, new_seen))
        
        if not new_beam:
            break
        
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        beam = new_beam[:beam_width]
        
        # Update best
        if beam:
            if len(beam[0][0]) > best_len:
                best_path = beam[0][0]
                best_len = len(best_path)
        
        if time.time() > deadline:
            break
    
    # Phase 5: Simulated annealing on path
    def anneal_path(initial_path, adj, iters=5000):
        current = list(initial_path)
        current_seen = set(current)
        best = list(current)
        best_seen = set(current)
        T = 1.0
        cooling = 0.995
        
        for _ in range(iters):
            if time.time() > deadline:
                break
            
            # Move: try to insert an unvisited vertex
            unvisited = set(range(n)) - current_seen
            if unvisited:
                # Pick a random unvisited vertex
                v = rng.choice(list(unvisited))
                # Find a position to insert
                valid_positions = []
                for i in range(len(current) + 1):
                    if i == 0:
                        if i < len(current) and v in adj[current[0]]:
                            valid_positions.append(i)
                    elif i == len(current):
                        if i > 0 and v in adj[current[-1]]:
                            valid_positions.append(i)
                    else:
                        if v in adj[current[i-1]] and v in adj[current[i]]:
                            valid_positions.append(i)
                
                if valid_positions:
                    pos = rng.choice(valid_positions)
                    new_path = current[:pos] + [v] + current[pos:]
                    new_seen = current_seen | {v}
                    delta = 1  # Length increases by 1
                    if delta > 0 or rng.random() < pow(2.718, delta / T):
                        current = new_path
                        current_seen = new_seen
                        if len(current) > len(best):
                            best = list(current)
                            best_seen = set(current)
            
            # Move: try to remove a vertex (to allow reconfiguration)
            if len(current) > 2:
                i = rng.randint(1, len(current) - 2)  # Don't remove ends
                v = current[i]
                # Check if removing v keeps path valid
                if current[i-1] in adj[current[i+1]]:
                    new_path = current[:i] + current[i+1:]
                    new_seen = current_seen - {v}
                    delta = -1
                    if rng.random() < pow(2.718, delta / T):
                        current = new_path
                        current_seen = new_seen
            
            T *= cooling
        
        return best
    
    # Apply annealing to best path
    if best_path and len(best_path) < n:
        annealed = anneal_path(best_path, adj, iters=10000)
        if len(annealed) > best_len:
            best_path = annealed
            best_len = len(best_path)
    
    # Final validation
    if not is_valid_path(best_path):
        # Fallback to simple greedy
        best_path = greedy_path(0, 'min_degree')
    
    return best_path
# EVOLVE-BLOCK-END
