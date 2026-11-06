import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    n_init = 32
    adj = [[-1] * n_init for _ in range(n_init)]
    for i in range(n_init):
        for j in range(i + 1, n_init):
            if i // 16 == j // 16:
                c = log[(i % 16) ^ (j % 16)] % 3
            else:
                c = 3
            adj[i][j] = c
            adj[j][i] = c
            
    current_n = n_init
    best_n = n_init
    best_adj = [row[:] for row in adj]
    
    # Precompute conflict sets for the initial graph
    # conflicts[u][c] = set of v such that adj[u][v] == c
    def build_conflicts(adj, n):
        conflicts = [set() for _ in range(n)] # Wait, need per color
        conflicts = [[set() for _ in range(4)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                c = adj[i][j]
                if c != -1:
                    conflicts[i][c].add(j)
        return conflicts

    conflicts = build_conflicts(adj, current_n)
    
    def try_extend(adj, n, conflicts, time_limit=5.0):
        start_t = time.time()
        max_restarts = 5000
        for restart in range(max_restarts):
            if time.time() > time_limit or time.time() > deadline:
                return None, None
                
            order = list(range(n))
            random.shuffle(order)
            
            assignment = [None] * n
            neighbors = [{}, {}, {}, {}]
            
            def is_compatible(u, c):
                # u is compatible with color c if no neighbor in neighbors[c] has edge color c to u
                # i.e., neighbors[c] is disjoint from conflicts[u][c]
                return neighbors[c].isdisjoint(conflicts[u][c])
                
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    return True
                    
                u = order[idx]
                
                colors = [0, 1, 2, 3]
                random.shuffle(colors)
                
                for c in colors:
                    if is_compatible(u, c):
                        assignment[u] = c
                        neighbors[c].add(u)
                        
                        if backtrack(idx + 1):
                            return True
                            
                        neighbors[c].remove(u)
                        assignment[u] = None
                        
                return False
                
            if backtrack(0):
                new_adj = [row[:] for row in adj]
                for i in range(n + 1):
                    new_adj.append([-1] * (n + 1))
                for i in range(n):
                    new_adj[n][i] = assignment[i]
                    new_adj[i][n] = assignment[i]
                
                # Update conflicts for new graph
                new_conflicts = build_conflicts(new_adj, n + 1)
                return new_adj, new_conflicts
                
        return None, None

    while current_n < 50:
        if time.time() > deadline:
            break
            
        t_limit = 2.0 if current_n < 40 else 0.5
        new_adj, new_conflicts = try_extend(adj, current_n, conflicts, time_limit=t_limit)
        
        if new_adj:
            adj = new_adj
            conflicts = new_conflicts
            current_n += 1
            best_n = current_n
            best_adj = [row[:] for row in adj]
        else:
            break

    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(best_adj[i][j])
            
    return {'n': best_n, 'colors': colors}
