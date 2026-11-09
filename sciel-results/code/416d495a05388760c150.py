"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 15}

def _is_valid(adj):
    """Check if the graph with adjacency matrix adj is valid.
    Valid means: no simple 3-edge path s-u-v-t has deg(s) == deg(t).
    """
    n = P['n']
    deg = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                deg[i] += 1
                deg[j] += 1
    
    # For every edge (u, v), check all s in N(u)\{v} and t in N(v)\{u}
    # If deg(s) == deg(t) and s != t, invalid.
    # Actually we need s != t. Since s is neighbor of u (not v) and t is neighbor of v (not u),
    # s and t could be equal only if s == t, which would mean s is neighbor of both u and v.
    # But the path is s-u-v-t, so s != v and t != u. If s == t, then s is adjacent to both u and v.
    # The path s-u-v-s is not a simple path (s appears twice). So we need s != t.
    
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                continue
            # Get neighbors of u excluding v
            nu = [w for w in range(n) if adj[u][w] and w != v]
            # Get neighbors of v excluding u
            nv = [w for w in range(n) if adj[v][w] and w != u]
            
            # For each s in nu and t in nv, if s != t and deg[s] == deg[t], invalid
            for s in nu:
                for t in nv:
                    if s == t:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _count_edges(adj):
    n = P['n']
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                count += 1
    return count

def _local_search(adj, deadline):
    """Local search: try adding edges, if invalid remove one, etc."""
    n = P['n']
    best_adj = [row[:] for row in adj]
    best_count = _count_edges(best_adj)
    
    # Build adjacency list for fast neighbor lookup
    def get_neighbors(a):
        return [j for j in range(n) if a[i][j]]  # This won't work, need proper closure
    
    while time.time() < deadline:
        # Try to add a random missing edge
        added = False
        for _ in range(100):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or adj[u][v]:
                continue
            
            # Try adding edge (u, v)
            adj[u][v] = 1
            adj[v][u] = 1
            
            if _is_valid(adj):
                added = True
                new_count = _count_edges(adj)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = [row[:] for row in adj]
            else:
                # Revert
                adj[u][v] = 0
                adj[v][u] = 0
            if added:
                break
        
        if not added:
            # Try removing a random edge and then adding two others (2-opt style)
            removed = False
            for _ in range(50):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v or not adj[u][v]:
                    continue
                
                # Remove edge (u, v)
                adj[u][v] = 0
                adj[v][u] = 0
                removed = True
                
                # Try to add two edges
                added_any = False
                for _2 in range(20):
                    w = random.randint(0, n - 1)
                    x = random.randint(0, n - 1)
                    if w == x or adj[w][x] or (w == u and x == v) or (w == v and x == u):
                        continue
                    adj[w][x] = 1
                    adj[x][w] = 1
                    if _is_valid(adj):
                        added_any = True
                        new_count = _count_edges(adj)
                        if new_count > best_count:
                            best_count = new_count
                            best_adj = [row[:] for row in adj]
                        break
                    else:
                        adj[w][x] = 0
                        adj[x][w] = 0
                
                if not added_any:
                    # Revert removal
                    adj[u][v] = 1
                    adj[v][u] = 1
                break
        
        if time.time() > deadline:
            break
    
    return best_adj, best_count

def construct():
    n = P['n']
    deadline = time.time() + 60  # 60 second deadline, safety margin
    
    random.seed(42)
    
    best_adj = None
    best_count = 0
    
    # Start with a star graph
    adj = [[0] * n for _ in range(n)]
    for v in range(1, n):
        adj[0][v] = 1
        adj[v][0] = 1
    
    # Multiple restarts with different initial configurations
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Different starting points
        if restart == 0:
            # Star
            adj = [[0] * n for _ in range(n)]
            for v in range(1, n):
                adj[0][v] = 1
                adj[v][0] = 1
        elif restart == 1:
            # Path
            adj = [[0] * n for _ in range(n)]
            for i in range(n - 1):
                adj[i][i + 1] = 1
                adj[i + 1][i] = 1
        elif restart == 2:
            # Cycle
            adj = [[0] * n for _ in range(n)]
            for i in range(n):
                adj[i][(i + 1) % n] = 1
                adj[(i + 1) % n][i] = 1
        else:
            # Random sparse graph
            adj = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.2:
                        adj[i][j] = 1
                        adj[j][i] = 1
        
        # Local search from this start
        cur_adj, cur_count = _local_search([row[:] for row in adj], deadline - 5)
        
        if cur_count > best_count:
            best_count = cur_count
            best_adj = cur_adj
    
    # Convert to edge list
    edges = []
    if best_adj is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i][j]:
                    edges.append([i, j])
    else:
        # Fallback: star
        for v in range(1, n):
            edges.append([0, v])
    
    # Ensure types are correct
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
