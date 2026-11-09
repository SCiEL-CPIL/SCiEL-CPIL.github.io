import random
import time
from itertools import combinations

def construct():
    n = 20
    deadline = time.time() + 110  # 110s safety margin
    
    best_edges = 0
    best_adj = [[0]*n for _ in range(n)]
    
    # Precompute all pairs for efficiency
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_valid_add(u, v, adj, neighbors):
        """Check if adding edge (u,v) creates a K3,3 using precomputed neighbors."""
        # K3,3: parts {u, a1, a2} and {v, b1, b2}
        # Need: (u,v) added, (u,b1),(u,b2) exist, (a1,v),(a2,v) exist,
        #       (a1,b1),(a1,b2),(a2,b1),(a2,b2) all exist
        
        Nv = neighbors[v]  # neighbors of v (excluding u since not yet added)
        Nu = neighbors[u]  # neighbors of u (excluding v since not yet added)
        
        # For each pair in Nv, check if they have >=2 common neighbors in Nu
        for i in range(len(Nv)):
            a1 = Nv[i]
            for j in range(i+1, len(Nv)):
                a2 = Nv[j]
                # Count common neighbors of a1 and a2 that are in Nu
                count = 0
                for b in Nu:
                    if adj[a1][b] and adj[a2][b]:
                        count += 1
                        if count >= 2:
                            return False
        return True
    
    def can_add(u, v, adj, neighbors):
        """Fast check if edge (u,v) can be added without creating K3,3."""
        if adj[u][v]:
            return False
        return is_valid_add(u, v, adj, neighbors)
    
    def add_edge(u, v, adj, neighbors):
        """Add edge (u,v) and update neighbor lists."""
        adj[u][v] = 1
        adj[v][u] = 1
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    def remove_edge(u, v, adj, neighbors):
        """Remove edge (u,v) and update neighbor lists."""
        adj[u][v] = 0
        adj[v][u] = 0
        neighbors[u].remove(v)
        neighbors[v].remove(u)
    
    def count_edges(adj):
        """Count edges in adjacency matrix."""
        return sum(sum(row) for row in adj) // 2
    
    def build_neighbors(adj):
        """Build neighbor lists from adjacency matrix."""
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        return neighbors
    
    def greedy_random():
        """Randomized greedy construction."""
        adj = [[0]*n for _ in range(n)]
        neighbors = [[] for _ in range(n)]
        edges = []
        
        all_edges = all_pairs[:]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add(u, v, adj, neighbors):
                add_edge(u, v, adj, neighbors)
                edges.append((u, v))
        
        return adj, neighbors, len(edges)
    
    def local_search(adj, neighbors, time_limit):
        """Local search with edge swaps to improve the solution."""
        start_time = time.time()
        improved = True
        current_edges = count_edges(adj)
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # Try removing an edge and adding two new ones
            # Focus on high-degree vertices
            degree = [len(neighbors[i]) for i in range(n)]
            high_degree = [i for i in range(n) if degree[i] > n//2]
            
            if not high_degree:
                high_degree = list(range(n))
            
            random.shuffle(high_degree)
            
            for u in high_degree:
                if time.time() - start_time > time_limit:
                    break
                
                # Try removing edges from u
                neighbors_u = neighbors[u][:]
                random.shuffle(neighbors_u)
                
                for v in neighbors_u:
                    if time.time() - start_time > time_limit:
                        break
                    
                    # Temporarily remove edge (u,v)
                    remove_edge(u, v, adj, neighbors)
                    
                    # Try to add new edges
                    added = []
                    for w in range(n):
                        if w != u and w != v and not adj[u][w]:
                            if can_add(u, w, adj, neighbors):
                                add_edge(u, w, adj, neighbors)
                                added.append(w)
                                if len(added) >= 2:
                                    break
                    
                    if len(added) >= 2:
                        # Net gain of 1 edge
                        improved = True
                        break
                    else:
                        # Undo: remove added edges and restore (u,v)
                        for w in added:
                            remove_edge(u, w, adj, neighbors)
                        add_edge(u, v, adj, neighbors)
            
            # Try 2-for-1 swaps: remove 2 edges, add 3 edges
            if not improved and time.time() - start_time < time_limit:
                # Sample random pairs of edges to remove
                current_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
                if len(current_edges_list) >= 2:
                    for _ in range(min(100, len(current_edges_list))):
                        if time.time() - start_time > time_limit:
                            break
                        
                        e1, e2 = random.sample(current_edges_list, 2)
                        u1, v1 = e1
                        u2, v2 = e2
                        
                        # Check if removing both is valid
                        remove_edge(u1, v1, adj, neighbors)
                        remove_edge(u2, v2, adj, neighbors)
                        
                        # Try to add 3 new edges
                        added = []
                        candidates = [(i, j) for i, j in all_pairs if not adj[i][j]]
                        random.shuffle(candidates)
                        
                        for u, v in candidates:
                            if len(added) >= 3:
                                break
                            if can_add(u, v, adj, neighbors):
                                add_edge(u, v, adj, neighbors)
                                added.append((u, v))
                        
                        if len(added) >= 3:
                            improved = True
                            # Update edge list
                            current_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
                            break
                        else:
                            # Undo
                            for u, v in added:
                                remove_edge(u, v, adj, neighbors)
                            add_edge(u1, v1, adj, neighbors)
                            add_edge(u2, v2, adj, neighbors)
        
        return adj, neighbors, count_edges(adj)
    
    # Run multiple restarts with local search
    restart_count = 0
    while time.time() < deadline:
        adj, neighbors, edge_count = greedy_random()
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = [row[:] for row in adj]
        
        # Apply local search if we have time
        remaining_time = deadline - time.time()
        if remaining_time > 5:  # At least 5 seconds remaining
            local_time = min(3.0, remaining_time * 0.3)
            adj, neighbors, edge_count = local_search(adj, neighbors, local_time)
            
            if edge_count > best_edges:
                best_edges = edge_count
                best_adj = [row[:] for row in adj]
        
        restart_count += 1
        # If we've found a very good solution, reduce search effort
        if best_edges >= 85:
            break
    
    # Final polish: try to add any remaining edges
    if time.time() < deadline - 2:
        neighbors = build_neighbors(best_adj)
        for u, v in all_pairs:
            if not best_adj[u][v]:
                if can_add(u, v, best_adj, neighbors):
                    add_edge(u, v, best_adj, neighbors)
    
    # Convert best_adj to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result.append([i, j])
    return result
