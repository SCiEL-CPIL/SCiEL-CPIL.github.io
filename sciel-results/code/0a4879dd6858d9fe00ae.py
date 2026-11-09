"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    n = P['n']
    vertices = list(range(n))
    
    # Build position map
    pos = {}
    for i, edge in enumerate(P['edge_order']):
        u, v = edge
        if u > v:
            u, v = v, u
        pos[(u, v)] = i + 1  # 1-based index
    
    # Precompute edge positions in a 2D array for fast access
    edge_pos = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            edge_pos[i][j] = edge_pos[j][i] = pos[(i, j)]
    
    # Compute lower bound K: for each vertex, the 2nd smallest incident edge position
    min_deg2 = 0
    for v in vertices:
        incident = []
        for u in vertices:
            if u != v:
                incident.append(edge_pos[v][u])
        incident.sort()
        if len(incident) >= 2:
            if incident[1] > min_deg2:
                min_deg2 = incident[1]
    
    # Function to score a cycle (max edge position)
    def score_cycle(cycle):
        max_pos = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            p = edge_pos[u][v]
            if p > max_pos:
                max_pos = p
        return max_pos
    
    # Find Hamilton cycle using DFS with pruning for a given K
    def find_ham_cycle(K, time_limit=None):
        # Build adjacency list for edges with pos <= K
        adj = [[] for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if edge_pos[u][v] <= K:
                    adj[u].append(v)
                    adj[v].append(u)
        
        # Check degree constraint
        for v in range(n):
            if len(adj[v]) < 2:
                return None
        
        # DFS with backtracking and pruning
        path = [0]
        visited = [False] * n
        visited[0] = True
        
        def dfs(path, visited, last):
            if time_limit and time.time() > time_limit:
                return None
            
            if len(path) == n:
                if edge_pos[last][0] <= K:
                    return path[:]
                return None
            
            last = path[-1]
            candidates = [nb for nb in adj[last] if not visited[nb]]
            
            if not candidates:
                return None
            
            # Pruning: check if any unvisited vertex is isolated
            unvisited = [v for v in range(n) if not visited[v]]
            forced = None
            for v in unvisited:
                unvisited_neighbors = [nb for nb in adj[v] if not visited[nb]]
                if len(unvisited_neighbors) == 0:
                    if v in candidates:
                        forced = v
                    else:
                        return None
            
            if forced is not None:
                candidates = [forced]
            else:
                # Warnsdorff's rule: fewest unvisited neighbors first
                def key(nb):
                    return len([x for x in adj[nb] if not visited[x]])
                candidates.sort(key=key)
            
            for nb in candidates:
                new_path = path + [nb]
                new_visited = visited[:]
                new_visited[nb] = True
                res = dfs(new_path, new_visited, nb)
                if res:
                    return res
            return None
        
        return dfs(path, visited, 0)
    
    # Binary search for minimal K
    lo = min_deg2
    hi = 276
    best_cycle = None
    best_K = 276
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        cycle = find_ham_cycle(mid)
        if cycle:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle:
        return best_cycle
    
    # Fallback: use Simulated Annealing to find a good cycle
    # Start with identity cycle
    best = list(range(n))
    best_score = score_cycle(best)
    
    rng = random.Random(42)
    current = best[:]
    current_score = best_score
    
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Random 2-opt move
        i, j = rng.sample(range(n), 2)
        if j < i:
            i, j = j, i
        
        # Reverse segment i..j
        new_cycle = current[:i] + current[i:j+1][::-1] + current[j+1:]
        
        # Compute new score
        new_score = 0
        for k in range(n):
            u = new_cycle[k]
            v = new_cycle[(k + 1) % n]
            p = edge_pos[u][v]
            if p > new_score:
                new_score = p
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_score - current_score
        if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
            current = new_cycle
            current_score = new_score
            if current_score < best_score:
                best = current[:]
                best_score = current_score
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Reset temperature occasionally to escape local minima
        if step % 10000 == 0:
            T = 50.0
    
    return best

# EVOLVE-BLOCK-END
