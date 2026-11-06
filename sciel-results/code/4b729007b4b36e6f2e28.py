import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 32
    
    # Precompute all 5-combinations
    combos_5 = list(itertools.combinations(range(n), 5))
    
    def has_c5(adj):
        for combo in combos_5:
            u, v, w, x, y = combo
            # Check if u-v-w-x-y-u is a cycle
            # Edges: (u,v), (v,w), (w,x), (x,y), (y,u)
            # We need to check all 12 permutations? 
            # No, just check if the induced subgraph is a C5.
            # A C5 has degree 2 for all vertices.
            deg = [0]*5
            verts = [u,v,w,x,y]
            for i in range(5):
                for j in range(i+1, 5):
                    if adj[verts[i]][verts[j]]:
                        deg[i] += 1
                        deg[j] += 1
            if all(d == 2 for d in deg):
                return True, combo
        return False, None

    def has_independent_set_9_heuristic(adj, restarts=50):
        for _ in range(restarts):
            order = list(range(n))
            random.shuffle(order)
            current = []
            for v in order:
                can_add = True
                for u in current:
                    if adj[u][v]:
                        can_add = False
                        break
                if can_add:
                    current.append(v)
            if len(current) >= 9:
                return True, current
        return False, None

    def has_independent_set_9_exact(adj):
        # Simple backtracking
        # This is slow, so only use as final check
        adj_mask = [0]*n
        for u in range(n):
            mask = 0
            for v in range(n):
                if adj[u][v]:
                    mask |= (1 << v)
            adj_mask[u] = mask

        def dfs(current_set, candidates):
            if len(current_set) >= 9:
                return True
            if len(candidates) < 9 - len(current_set):
                return False
            if not candidates:
                return False
            
            v = candidates[0]
            
            # Branch 1: Include v
            new_candidates = []
            for u in candidates[1:]:
                if not (adj_mask[v] & (1 << u)):
                    new_candidates.append(u)
            if dfs(current_set + [v], new_candidates):
                return True
            
            # Branch 2: Exclude v
            if dfs(current_set, candidates[1:]):
                return True
            
            return False

        return dfs([], list(range(n)))

    for attempt in range(500):
        if time.time() > deadline:
            break
            
        adj = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.5:
                    adj[u][v] = 1
                    adj[v][u] = 1
        
        for step in range(200):
            if time.time() > deadline:
                break
            
            has_c, c5_combo = has_c5(adj)
            if has_c:
                u, v, w, x, y = c5_combo
                edge_idx = random.randint(0, 4)
                if edge_idx == 0:
                    adj[u][v] = 0
                    adj[v][u] = 0
                elif edge_idx == 1:
                    adj[v][w] = 0
                    adj[w][v] = 0
                elif edge_idx == 2:
                    adj[w][x] = 0
                    adj[x][w] = 0
                elif edge_idx == 3:
                    adj[x][y] = 0
                    adj[y][x] = 0
                else:
                    adj[y][u] = 0
                    adj[u][y] = 0
                continue
            
            has_i9, i9_set = has_independent_set_9_heuristic(adj)
            if has_i9:
                u, v = random.sample(i9_set, 2)
                if not adj[u][v]:
                    adj[u][v] = 1
                    adj[v][u] = 1
                continue
            
            # Final check
            if time.time() > deadline - 5:
                break
            if has_independent_set_9_exact(adj):
                # Break it
                has_i9, i9_set = has_independent_set_9_heuristic(adj, restarts=100)
                if has_i9:
                    u, v = random.sample(i9_set, 2)
                    if not adj[u][v]:
                        adj[u][v] = 1
                        adj[v][u] = 1
                continue
            
            edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u][v]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
            
    # Fallback
    n_fallback = 16
    edges_fallback = []
    for u in range(8):
        for v in range(8, 16):
            edges_fallback.append([u, v])
    return {'n': n_fallback, 'edges': edges_fallback}
