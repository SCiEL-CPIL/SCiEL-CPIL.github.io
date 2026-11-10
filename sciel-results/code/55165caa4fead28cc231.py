import random
import time

def construct():
    n = 20
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def can_add(N, u, v):
        # u, v not connected
        X = N[v] & ~(1 << u)
        Y = N[u] & ~(1 << v)
        
        px = popcount(X)
        py = popcount(Y)
        if px < 2 or py < 2:
            return True
            
        xs = [b for b in range(n) if (X >> b) & 1]
        # No need to list ys, just use bitmask Y
        
        for i in range(len(xs)):
            for j in range(i+1, len(xs)):
                common = N[xs[i]] & N[xs[j]] & Y
                if popcount(common) >= 2:
                    return False
        return True

    def clean(N):
        # Remove edges to ensure no K3,3
        # This is called on the initial random graph
        while True:
            found = False
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        common = N[i] & N[j] & N[k]
                        if popcount(common) >= 3:
                            # Found K3,3
                            # Remove a random edge from the K3,3
                            # Edges are (i, nb), (j, nb), (k, nb) for nb in common
                            neighbors = [b for b in range(n) if (common >> b) & 1]
                            # Pick random neighbor
                            nb = random.choice(neighbors)
                            # Pick random vertex from {i,j,k}
                            u = random.choice([i, j, k])
                            N[u] &= ~(1 << nb)
                            N[nb] &= ~(1 << u)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break

    def run_search(start_graph, time_limit):
        N = [0] * n
        for u, v in start_graph:
            N[u] |= (1 << v)
            N[v] |= (1 << u)
            
        clean(N)
        
        best_edges = sum(popcount(N[i]) for i in range(n)) // 2
        best_N = N[:]
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            missing = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (N[u] >> v) & 1:
                        missing.append((u, v))
                        
            random.shuffle(missing)
            
            for u, v in missing:
                if can_add(N, u, v):
                    N[u] |= (1 << v)
                    N[v] |= (1 << u)
                    improved = True
                    current_edges = sum(popcount(N[i]) for i in range(n)) // 2
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_N = N[:]
                        
            if not improved:
                # Kick: remove random edge
                existing = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (N[u] >> v) & 1:
                            existing.append((u, v))
                if existing:
                    u, v = random.choice(existing)
                    N[u] &= ~(1 << v)
                    N[v] &= ~(1 << u)
                else:
                    break

        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (best_N[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    start_time = time.time()
    time_limit = 140
    
    best_edges_overall = []
    best_count = 0
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        p = 0.5 + 0.2 * (restarts % 10) / 10.0
        start_graph = []
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    start_graph.append((u, v))
                    
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
        alloc_time = min(remaining - 2, 3.0)
        
        res = run_search(start_graph, alloc_time)
        if len(res) > best_count:
            best_count = len(res)
            best_edges_overall = res
            
    return best_edges_overall
