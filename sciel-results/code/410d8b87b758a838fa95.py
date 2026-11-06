"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    all_pairs = list(combinations(range(n), 2))
    
    def is_k23_free(adj, edge):
        """Check if adding edge to adj keeps graph K2,3-free."""
        u, v = edge
        # For every pair of vertices (x, y), check if they have >= 3 common neighbors
        # After adding (u,v), the only pairs that could newly violate are those 
        # where one of the vertices is u or v, or both.
        # Actually, adding edge (u,v) increases common neighbor count for pairs involving u or v.
        # Specifically, for any vertex w != u,v:
        #   common_neighbors(u, w) gains v if v is adjacent to w
        #   common_neighbors(v, w) gains u if u is adjacent to w
        # So we need to check pairs (u, w) and (v, w) for all w != u, v.
        
        # Check pairs (u, w) for w != u, v
        for w in range(n):
            if w == u or w == v:
                continue
            # Count common neighbors of u and w in the NEW graph
            # Common neighbors are vertices z adjacent to both u and w
            # In new graph, u is adjacent to v, so if v is adjacent to w, v becomes a common neighbor
            cnt = 0
            for z in range(n):
                if z == u or z == w:
                    continue
                if adj[u][z] and adj[w][z]:
                    cnt += 1
            if cnt >= 3:
                return False
        
        # Check pairs (v, w) for w != u, v
        for w in range(n):
            if w == u or w == v:
                continue
            cnt = 0
            for z in range(n):
                if z == v or z == w:
                    continue
                if adj[v][z] and adj[w][z]:
                    cnt += 1
            if cnt >= 3:
                return False
        
        # Also need to check pair (u, v) itself
        cnt = 0
        for z in range(n):
            if z == u or z == v:
                continue
            if adj[u][z] and adj[v][z]:
                cnt += 1
        if cnt >= 3:
            return False
        
        return True
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    def edges_from_adj(adj):
        return [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    def greedy_build(order):
        """Build graph by greedily adding edges in given order."""
        adj = [[False]*n for _ in range(n)]
        for edge in order:
            if is_k23_free(adj, edge):
                adj[edge[0]][edge[1]] = True
                adj[edge[1]][edge[0]] = True
        return adj
    
    def local_search(adj, max_iters=5000):
        """Try to add edges via local search."""
        adj = [row[:] for row in adj]
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try adding each missing edge
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i][j]:
                        if is_k23_free(adj, (i, j)):
                            adj[i][j] = True
                            adj[j][i] = True
                            improved = True
                            if time.time() > deadline:
                                return adj
        return adj
    
    best_adj = None
    best_count = 0
    
    # Try multiple random orderings
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Random order
        random_order = all_pairs[:]
        random.shuffle(random_order)
        
        adj = greedy_build(random_order)
        adj = local_search(adj)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [row[:] for row in adj]
        
        # If we've reached a good count, do more targeted search
        if best_count >= 45:
            # Try to improve further
            for _ in range(10):
                if time.time() > deadline:
                    break
                # Random restart from best
                adj2 = [row[:] for row in best_adj]
                adj2 = local_search(adj2, max_iters=2000)
                cnt2 = count_edges(adj2)
                if cnt2 > best_count:
                    best_count = cnt2
                    best_adj = [row[:] for row in adj2]
    
    if best_adj is None:
        # Fallback: trivial 5*K4
        edges = []
        for k in range(5):
            edges += [[4*k + a, 4*k + b] for a in range(4) for b in range(a+1, 4)]
        return edges
    
    return edges_from_adj(best_adj)
# EVOLVE-BLOCK-END
