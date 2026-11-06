import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 20
    best_edges = []
    best_count = 0
    best_adj = [[False] * n for _ in range(n)]
    best_common = [[0] * n for _ in range(n)]
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_add(adj, common, u, v):
        # Check if adding (u, v) creates K2,3
        # For w in N(u), common[v][w] must be < 2
        # For w in N(v), common[u][w] must be < 2
        for w in range(n):
            if adj[u][w]:
                if common[v][w] >= 2:
                    return False
            if adj[v][w]:
                if common[u][w] >= 2:
                    return False
        return True

    def add_edge(adj, common, u, v):
        adj[u][v] = True
        adj[v][u] = True
        for w in range(n):
            if adj[u][w]:
                common[v][w] += 1
                common[w][v] += 1
            if adj[v][w]:
                common[u][w] += 1
                common[w][u] += 1

    def remove_edge(adj, common, u, v):
        adj[u][v] = False
        adj[v][u] = False
        for w in range(n):
            if adj[u][w]:
                common[v][w] -= 1
                common[w][v] -= 1
            if adj[v][w]:
                common[u][w] -= 1
                common[w][u] -= 1

    def get_edges(adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j]]

    def greedy_fill(order):
        adj = [[False] * n for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        count = 0
        for u, v in order:
            if is_valid_add(adj, common, u, v):
                add_edge(adj, common, u, v)
                count += 1
        return adj, common, count

    def local_search(adj, common, count):
        nonlocal best_edges, best_count, best_adj, best_common
        improved = True
        # Limit local search steps to avoid long stalls
        steps = 0
        while improved and steps < 50 and time.time() < deadline - 5:
            improved = False
            steps += 1
            existing_edges = get_edges(adj)
            if not existing_edges:
                break
            
            # Try removing a random edge
            u, v = random.choice(existing_edges)
            
            # Backup
            adj_b = [row[:] for row in adj]
            common_b = [row[:] for row in common]
            
            remove_edge(adj, common, u, v)
            count -= 1
            
            # Try to add all possible edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i][j]]
            random.shuffle(missing)
            for x, y in missing:
                if is_valid_add(adj, common, x, y):
                    add_edge(adj, common, x, y)
                    count += 1
            
            if count > best_count:
                best_count = count
                best_edges = get_edges(adj)
                best_adj = [row[:] for row in adj]
                best_common = [row[:] for row in common]
                improved = True
            else:
                # Restore
                adj = adj_b
                common = common_b
                count += 1
        return count

    # Main loop
    while time.time() < deadline - 10:
        random.shuffle(all_edges)
        adj, common, count = greedy_fill(all_edges)
        if count > best_count:
            best_count = count
            best_edges = get_edges(adj)
            best_adj = [row[:] for row in adj]
            best_common = [row[:] for row in common]
            
        # Run local search occasionally
        if random.random() < 0.1:
            adj_ls = [row[:] for row in best_adj]
            common_ls = [row[:] for row in best_common]
            count_ls = best_count
            local_search(adj_ls, common_ls, count_ls)

    return best_edges
