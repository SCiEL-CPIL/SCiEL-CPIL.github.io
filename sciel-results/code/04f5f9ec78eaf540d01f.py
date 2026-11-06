import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Construct K32 using the seed method
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

    n = 32
    # graph[i][j]
    graph = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if i // 16 == j // 16:
                c = log[(i % 16) ^ (j % 16)] % 3
            else:
                c = 3
            graph[i][j] = c
            graph[j][i] = c

    # 2. Extend to n > 32
    # We will try to add vertices one by one.
    # Current n is 32.
    
    # Helper to count violations for a new vertex color assignment
    # new_colors is a list of length n (indices 0 to n-1)
    # Violations: pairs (i, j) such that new_colors[i] == new_colors[j] == graph[i][j]
    
    def get_violations(new_colors, n_curr):
        v = 0
        for i in range(n_curr):
            ci = new_colors[i]
            # Optimization: only check j > i
            for j in range(i + 1, n_curr):
                if new_colors[j] == ci and graph[i][j] == ci:
                    v += 1
        return v

    # Precompute adjacency lists by color for faster delta calculation?
    # adj[i][c] = list of j such that graph[i][j] == c
    # This helps in delta calculation.
    
    def build_adj(n_curr):
        adj = [[[] for _ in range(4)] for _ in range(n_curr)]
        for i in range(n_curr):
            for j in range(n_curr):
                if i != j:
                    c = graph[i][j]
                    adj[i][c].append(j)
        return adj

    adj = build_adj(n)

    for target_n in range(33, 51):
        if time.time() > deadline - 10:
            break
        
        # Try to extend to target_n
        # New vertex index is n (current size)
        # We need to assign colors to edges (n, i) for i in 0..n-1
        # Let new_colors[i] be the color of edge (n, i)
        
        success = False
        max_iters = 2000
        restarts = 20
        
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            new_colors = [random.randint(0, 3) for _ in range(n)]
            
            # Initial violation count
            # We can compute this quickly
            current_v = get_violations(new_colors, n)
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                if current_v == 0:
                    success = True
                    break
                
                # Pick a random variable i to change
                i = random.randint(0, n - 1)
                old_c = new_colors[i]
                
                # Evaluate all 4 colors
                best_c = old_c
                best_v = current_v
                
                # Pre-fetch adj[i]
                adj_i = adj[i]
                
                for c in range(4):
                    if c == old_c:
                        continue
                    
                    # Delta calculation
                    # Violations removed: j in adj_i[old_c] such that new_colors[j] == old_c
                    # Violations added: j in adj_i[c] such that new_colors[j] == c
                    
                    delta = 0
                    # Removed
                    for j in adj_i[old_c]:
                        if new_colors[j] == old_c:
                            delta -= 1
                    # Added
                    for j in adj_i[c]:
                        if new_colors[j] == c:
                            delta += 1
                    
                    new_v = current_v + delta
                    if new_v < best_v:
                        best_v = new_v
                        best_c = c
                
                if best_c == old_c:
                    # No improvement
                    # Random move to escape local minimum
                    if random.random() < 0.2:
                        new_c = random.randint(0, 3)
                        if new_c != old_c:
                            # Calculate delta for random move
                            delta = 0
                            for j in adj_i[old_c]:
                                if new_colors[j] == old_c:
                                    delta -= 1
                            for j in adj_i[new_c]:
                                if new_colors[j] == new_c:
                                    delta += 1
                            new_colors[i] = new_c
                            current_v += delta
                    else:
                        # Stuck, restart
                        break
                else:
                    new_colors[i] = best_c
                    current_v = best_v
        
        if success:
            # Extend the graph
            # Add new row and column
            for i in range(n):
                graph[i].append(new_colors[i])
                graph[n].append(0) # Placeholder, will be filled
            # Actually, graph is a list of lists.
            # We need to add a new row.
            # graph currently has n rows.
            # We append new_colors[i] to row i.
            # We create a new row for index n.
            new_row = [0] * (n + 1)
            for i in range(n):
                new_row[i] = new_colors[i]
            new_row[n] = 0
            graph.append(new_row)
            
            n = target_n
            # Rebuild adj
            adj = build_adj(n)
        else:
            # Failed to extend
            break

    # Convert graph to flat list of colors in lexicographic order
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(graph[i][j])
    
    return {'n': n, 'colors': colors}
