"""Combined approach: Start from K_{a,b} bipartite (structurally valid when a!=b),
then use fast bitmask-based local search to maximize edges."""
# EVOLVE-BLOCK-START
import time
import random

N = 15
DEADLINE = 82.0  # Safety margin before 90s kill

def _is_valid_fast(adj):
    """
    Check if graph is valid: no simple 3-edge path s-u-v-t has deg(s) == deg(t).
    For each edge (u,v): no degree appears in both N(u)\{v} and N(v)\{u} 
    (with the caveat that if the only vertex with that degree on both sides is the same vertex, it's fine).
    """
    deg = [0] * N
    neighbors = [[] for _ in range(N)]
    
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u][v]:
                deg[u] += 1
                deg[v] += 1
                neighbors[u].append(v)
                neighbors[v].append(u)
                
    for u in range(N):
        for v in neighbors[u]:
            if u >= v:
                continue
            
            # Build mapping: degree -> list of vertices in N(u)\{v}
            deg_map_u = {}
            for s in neighbors[u]:
                if s != v:
                    d = deg[s]
                    if d not in deg_map_u:
                        deg_map_u[d] = []
                    deg_map_u[d].append(s)
            
            # Check each t in N(v)\{u}
            for t in neighbors[v]:
                if t == u:
                    continue
                d = deg[t]
                if d in deg_map_u:
                    # Check if there exists s in deg_map_u[d] with s != t
                    for s in deg_map_u[d]:
                        if s != t:
                            return False
    return True

def _edge_count(adj):
    return sum(sum(row) for row in adj) // 2

def _try_add(adj, u, v):
    if u == v or adj[u][v]:
        return False
    adj[u][v] = 1
    adj[v][u] = 1
    if _is_valid_fast(adj):
        return True
    adj[u][v] = 0
    adj[v][u] = 0
    return False

def _try_swap(adj, u1, v1, u2, v2):
    if adj[u1][v1] == 0 or adj[u2][v2] == 1:
        return False
    adj[u1][v1] = 0
    adj[v1][u1] = 0
    adj[u2][v2] = 1
    adj[v2][u2] = 1
    if _is_valid_fast(adj):
        return True
    adj[u1][v1] = 1
    adj[v1][u1] = 1
    adj[u2][v2] = 0
    adj[v2][u2] = 0
    return False

def _local_search(adj, start_time, seed):
    """Hill climbing with add and swap moves."""
    random.seed(seed)
    current_count = _edge_count(adj)
    best_adj = [row[:] for row in adj]
    best_count = current_count
    no_improve_rounds = 0
    
    while time.time() - start_time < DEADLINE:
        improved = False
        
        # Try adding missing edges
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not adj[u][v]]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() - start_time > DEADLINE:
                break
            if _try_add(adj, u, v):
                current_count += 1
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [row[:] for row in adj]
        
        if improved:
            no_improve_rounds = 0
            continue
        
        no_improve_rounds += 1
        if no_improve_rounds > 5:
            break
        
        # Try swaps: remove one existing edge, add one missing edge
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) if adj[u][v]]
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not adj[u][v]]
        random.shuffle(existing)
        random.shuffle(missing)
        
        found_swap = False
        for u1, v1 in existing:
            if time.time() - start_time > DEADLINE:
                break
            for u2, v2 in missing:
                if _try_swap(adj, u1, v1, u2, v2):
                    found_swap = True
                    break
            if found_swap:
                break
        
        if not found_swap:
            break
    
    return best_adj, best_count

def _bipartite_start(a_size):
    """Create K_{a_size, N-a_size}."""
    adj = [[0] * N for _ in range(N)]
    for u in range(a_size):
        for v in range(a_size, N):
            adj[u][v] = 1
            adj[v][u] = 1
    return adj

def construct():
    start_time = time.time()
    best_adj = [[0] * N for _ in range(N)]
    best_count = 0
    
    # Try different bipartite splits as starting points
    for a_size in [7, 8, 6, 9, 5, 10]:
        if time.time() - start_time > DEADLINE - 10:
            break
        adj = _bipartite_start(a_size)
        count = _edge_count(adj)
        
        # Local search from this starting point
        adj_ls, count_ls = _local_search(adj, start_time, seed=a_size * 100)
        if count_ls > best_count:
            best_count = count_ls
            best_adj = adj_ls
    
    # Also try a few random greedy starts
    for i in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
        random.seed(10000 + i)
        adj = [[0] * N for _ in range(N)]
        edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(edges)
        for u, v in edges:
            if time.time() - start_time > DEADLINE - 5:
                break
            _try_add(adj, u, v)
        count = _edge_count(adj)
        if count > best_count:
            # Do local search from this point
            adj_ls, count_ls = _local_search(adj, start_time, seed=20000 + i)
            if count_ls > best_count:
                best_count = count_ls
                best_adj = adj_ls
    
    # Final validation
    if not _is_valid_fast(best_adj):
        # Fallback to K_{7,8}
        best_adj = _bipartite_start(7)
        best_count = _edge_count(best_adj)
    
    result = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u][v]:
                result.append([int(u), int(v)])
    
    return result

# EVOLVE-BLOCK-END
