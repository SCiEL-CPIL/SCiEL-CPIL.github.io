import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s safety margin out of 90s
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute adjacency as list of sorted lists for speed
    adj_list = [sorted(adj[i]) for i in range(n)]
    
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
    
    def try_extend_both_ends(path, adj):
        """Try to extend path from both ends greedily."""
        changed = True
        while changed:
            changed = False
            # Try extending from end
            last = path[-1]
            candidates = adj[last] - set(path)
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - set(path)), w))
                path.append(v)
                changed = True
            # Try extending from start
            first = path[0]
            candidates = adj[first] - set(path)
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - set(path)), w))
                path.insert(0, v)
                changed = True
        return path
    
    def local_improve(path, adj, max_iters=300):
        """Try to improve path by 2-opt reversals and extension."""
        best = list(path)
        best_len = len(best)
        
        for _ in range(max_iters):
            improved = False
            
            # Try 2-opt style: reverse a segment
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
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
                            improved = True
                            break
                if improved:
                    break
            
            # Try to extend from both ends
            extended = try_extend_both_ends(best, adj)
            if len(extended) > best_len:
                best = extended
                best_len = len(best)
                improved = True
            
            if not improved:
                break
        
        return best
    
    def hungarian_algorithm(cost_matrix):
        """
        Solve assignment problem using Hungarian algorithm.
        cost_matrix: n x n matrix where we want to minimize total cost.
        Returns: list of column assignments for each row
        """
        import numpy as np
        n_mat = len(cost_matrix)
        if n_mat == 0:
            return []
        
        cost = np.array(cost_matrix, dtype=float)
        
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
        
        result = [0] * n_mat
        for j in range(1, n_mat + 1):
            if p[j] > 0:
                result[p[j]-1] = j-1
        
        return result
    
    def max_weight_matching_for_path():
        """
        Use Maximum Weight Matching formulation to find a long path.
        Find maximum weight matching in a bipartite graph.
        """
        M = n + 1
        
        cost_matrix = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    cost_matrix[i][j] = 0
                elif j in adj[i]:
                    cost_matrix[i][j] = -1
                else:
                    cost_matrix[i][j] = M
        
        assignment = hungarian_algorithm(cost_matrix)
        
        next_vertex = [-1] * n
        prev_vertex = [-1] * n
        for i in range(n):
            j = assignment[i]
            if i != j:
                next_vertex[i] = j
                prev_vertex[j] = i
        
        visited = [False] * n
        best_path = []
        
        for start in range(n):
            if visited[start]:
                continue
            if prev_vertex[start] != -1:
                continue
            
            path = [start]
            current = start
            while next_vertex[current] != -1:
                current = next_vertex[current]
                if visited[current]:
                    break
                path.append(current)
            
            for v in path:
                visited[v] = True
            
            if len(path) > len(best_path):
                best_path = path
        
        return best_path
    
    # Phase 1: Generate many candidate paths using different strategies
    candidates = []
    
    for start in range(n):
        for strategy in ['min_degree', 'max_degree']:
            path = greedy_path(start, strategy)
            if path:
                candidates.append(path)
    
    rng = random.Random(42)
    for _ in range(300):
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
    
    for path in candidates[:80]:
        if len(path) <= best_len - 5:
            continue
        improved = local_improve(path, adj, max_iters=300)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
        if time.time() > deadline:
            break
    
    # Phase 4: Beam search
    beam_width = 80
    beam = []
    for start in range(n):
        beam.append(([start], {start}))
    
    beam.sort(key=lambda x: len(x[0]), reverse=True)
    beam = beam[:beam_width]
    
    for step in range(40):
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
        
        if beam:
            if len(beam[0][0]) > best_len:
                best_path = beam[0][0]
                best_len = len(best_path)
        
        if time.time() > deadline:
            break
    
    # Phase 5: Simulated annealing on path
    def anneal_path(initial_path, adj, iters=15000):
        current = list(initial_path)
        current_seen = set(current)
        best = list(current)
        best_seen = set(current)
        T = 1.0
        cooling = 0.998
        
        for _ in range(iters):
            if time.time() > deadline:
                break
            
            unvisited = set(range(n)) - current_seen
            if unvisited:
                v = rng.choice(list(unvisited))
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
                    delta = 1
                    if delta > 0 or rng.random() < pow(2.718, delta / T):
                        current = new_path
                        current_seen = new_seen
                        if len(current) > len(best):
                            best = list(current)
                            best_seen = set(current)
            
            if len(current) > 2:
                i = rng.randint(1, len(current) - 2)
                v = current[i]
                if current[i-1] in adj[current[i+1]]:
                    new_path = current[:i] + current[i+1:]
                    new_seen = current_seen - {v}
                    delta = -1
                    if rng.random() < pow(2.718, delta / T):
                        current = new_path
                        current_seen = new_seen
            
            T *= cooling
        
        return best
    
    if best_path and len(best_path) < n:
        annealed = anneal_path(best_path, adj, iters=20000)
        if len(annealed) > best_len:
            best_path = annealed
            best_len = len(best_path)
    
    # Phase 6: Final aggressive local search with multiple restarts
    if time.time() < deadline - 10:
        # Try to extend the best path further with a more aggressive local search
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            # Start from a random subset of the best path
            if len(best_path) > 10:
                subset_size = rng.randint(5, min(15, len(best_path)))
                indices = rng.sample(range(len(best_path)), subset_size)
                indices.sort()
                sub_path = [best_path[i] for i in indices]
                # Check if sub_path is a valid path
                valid = True
                for i in range(len(sub_path) - 1):
                    if sub_path[i+1] not in adj[sub_path[i]]:
                        valid = False
                        break
                if valid:
                    improved = local_improve(sub_path, adj, max_iters=200)
                    if len(improved) > best_len:
                        best_path = improved
                        best_len = len(best_path)
    
    # Final validation
    if not is_valid_path(best_path):
        best_path = greedy_path(0, 'min_degree')
    
    return best_path
# EVOLVE-BLOCK-END
