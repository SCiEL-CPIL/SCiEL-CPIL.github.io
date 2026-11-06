import random
import time

def construct():
    n = 20
    best_edges = 0
    best_adj = None
    best_common = None
    
    start_time = time.time()
    timeout = 110.0
    
    def init_graph():
        adj = [set() for _ in range(n)]
        common = [[0]*n for _ in range(n)]
        return adj, common

    def can_add(u, v, adj, common):
        if v in adj[u]:
            return False
        for x in adj[u]:
            if x != v and common[v][x] >= 2:
                return False
        for y in adj[v]:
            if y != u and common[u][y] >= 2:
                return False
        return True

    def add_edge(u, v, adj, common):
        adj[u].add(v)
        adj[v].add(u)
        for x in adj[u]:
            if x != v:
                common[v][x] += 1
        for y in adj[v]:
            if y != u:
                common[u][y] += 1

    def remove_edge(u, v, adj, common):
        adj[u].remove(v)
        adj[v].remove(u)
        for x in adj[u]:
            if x != v:
                common[v][x] -= 1
        for y in adj[v]:
            if y != u:
                common[u][y] -= 1

    def count_edges(adj):
        return sum(len(s) for s in adj) // 2

    def improve(adj, common):
        changed = True
        while changed:
            changed = False
            pairs = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add(u, v, adj, common):
                    add_edge(u, v, adj, common)
                    changed = True

    def repair(adj, common):
        # Remove edges to satisfy K_{2,3}-free
        # Find any u, v with common[u][v] >= 3
        while True:
            found = False
            for u in range(n):
                for v in range(u+1, n):
                    if common[u][v] >= 3:
                        found = True
                        # Find a common neighbor z
                        # z is in adj[u] and adj[v]
                        common_neighbors = adj[u].intersection(adj[v])
                        # Randomly pick one
                        z = random.choice(list(common_neighbors))
                        # Remove (u,z) or (v,z)
                        if random.random() < 0.5:
                            remove_edge(u, z, adj, common)
                        else:
                            remove_edge(v, z, adj, common)
                        break
                if found:
                    break
            if not found:
                break

    for restart in range(500):
        if time.time() - start_time > timeout:
            break
        
        adj, common = init_graph()
        
        # Alternate between empty start and random start
        if restart % 2 == 0:
            # Empty start
            pass
        else:
            # Random start
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.5:
                        add_edge(u, v, adj, common)
            repair(adj, common)
            
        improve(adj, common)
        
        current_edges = count_edges(adj)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [s.copy() for s in adj]
            best_common = [row[:] for row in common]
            
        # Try swaps if stuck
        if current_edges > 30 and time.time() - start_time < timeout - 5.0:
            # Try to remove a random edge and re-improve
            all_edges = [(u, v) for u in range(n) for v in range(u+1, n) if v in adj[u]]
            if all_edges:
                random.shuffle(all_edges)
                improved = False
                for u, v in all_edges[:5]: # Try first 5
                    if time.time() - start_time > timeout:
                        break
                    adj_copy = [s.copy() for s in adj]
                    common_copy = [row[:] for row in common]
                    
                    remove_edge(u, v, adj, common)
                    improve(adj, common)
                    
                    new_edges = count_edges(adj)
                    if new_edges > current_edges:
                        current_edges = new_edges
                        improved = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = [s.copy() for s in adj]
                            best_common = [row[:] for row in common]
                        # Update all_edges for next iteration?
                        # Just break and restart the swap loop for this state
                        break
                    else:
                        adj = adj_copy
                        common = common_copy
                if not improved:
                    # Try a few more random swaps
                    for _ in range(5):
                        if time.time() - start_time > timeout:
                            break
                        u, v = random.choice(all_edges)
                        adj_copy = [s.copy() for s in adj]
                        common_copy = [row[:] for row in common]
                        
                        remove_edge(u, v, adj, common)
                        improve(adj, common)
                        
                        new_edges = count_edges(adj)
                        if new_edges > current_edges:
                            current_edges = new_edges
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = [s.copy() for s in adj]
                                best_common = [row[:] for row in common]
                            break
                        else:
                            adj = adj_copy
                            common = common_copy

    if best_adj is None:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    return edges
