"""Rewritten program: start from K_{7,8} and local-search for denser valid graphs."""
# EVOLVE-BLOCK-START
import time
import random

N = 15
DEADLINE = 85.0

def _degrees(adj):
    return [sum(row) for row in adj]

def _is_valid(adj):
    """Check no simple 3-edge path s-u-v-t (s!=t) has deg(s)==deg(t)."""
    deg = _degrees(adj)
    n = N
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v] == 0:
                continue
            for s in range(n):
                if s == v or s == u or adj[u][s] == 0:
                    continue
                for t in range(n):
                    if t == u or t == v or s == t or adj[v][t] == 0:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _edge_count(adj):
    return sum(sum(row) for row in adj) // 2

def _try_add(adj, u, v):
    """Try adding edge (u,v). Return True if valid."""
    if u == v or adj[u][v]:
        return False
    adj[u][v] = 1
    adj[v][u] = 1
    if _is_valid(adj):
        return True
    adj[u][v] = 0
    adj[v][u] = 0
    return False

def _try_swap(adj, u1, v1, u2, v2):
    """Try removing (u1,v1) and adding (u2,v2). Return True if valid."""
    if adj[u1][v1] == 0 or adj[u2][v2] == 1:
        return False
    adj[u1][v1] = 0
    adj[v1][u1] = 0
    adj[u2][v2] = 1
    adj[v2][u2] = 1
    if _is_valid(adj):
        return True
    adj[u1][v1] = 1
    adj[v1][u1] = 1
    adj[u2][v2] = 0
    adj[v2][u2] = 0
    return False

def _make_bipartite(start_time):
    """Create K_{7,8} bipartite graph."""
    adj = [[0] * N for _ in range(N)]
    # Partition: 0-6 in A (7 vertices), 7-14 in B (8 vertices)
    for i in range(7):
        for j in range(7, 15):
            adj[i][j] = 1
            adj[j][i] = 1
    return adj

def _local_search(adj, start_time, max_iterations=5000):
    """Local search: try adding edges and swaps to improve."""
    current_count = _edge_count(adj)
    best_adj = [row[:] for row in adj]
    best_count = current_count
    no_improve = 0
    iter_count = 0
    
    while iter_count < max_iterations:
        if time.time() - start_time > DEADLINE:
            break
        iter_count += 1
        
        improved = False
        
        # Try adding each missing edge
        for u in range(N):
            for v in range(u + 1, N):
                if adj[u][v] == 0:
                    if _try_add(adj, u, v):
                        current_count += 1
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = [row[:] for row in adj]
        
        # Try swaps: remove existing edge, add missing edge
        if not improved:
            existing = [(u, v) for u in range(N) for v in range(u+1, N) if adj[u][v]]
            missing = [(u, v) for u in range(N) for v in range(u+1, N) if not adj[u][v]]
            random.shuffle(existing)
            random.shuffle(missing)
            for (u1, v1) in existing:
                if time.time() - start_time > DEADLINE:
                    break
                for (u2, v2) in missing:
                    if _try_swap(adj, u1, v1, u2, v2):
                        # Edge count stays same, but structure changed
                        # Could enable future additions
                        improved = True
                        break
                if improved:
                    break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 20:
                break
    
    return best_adj, best_count

def _multi_start_local_search(start_time):
    """Multiple random restarts with local search from different bipartite starts."""
    best_adj = None
    best_count = 0
    
    # Start with standard K_{7,8}
    adj = _make_bipartite(start_time)
    best_adj, best_count = _local_search(adj, start_time)
    
    # Try different bipartite splits
    for split in range(2, N - 1):
        if time.time() - start_time > DEADLINE:
            break
        adj = [[0] * N for _ in range(N)]
        for i in range(split):
            for j in range(split, N):
                adj[i][j] = 1
                adj[j][i] = 1
        a, c = _local_search(adj, start_time, max_iterations=2000)
        if c > best_count:
            best_count = c
            best_adj = a
    
    # Random perturbations of the best solution
    if time.time() - start_time < DEADLINE - 10:
        for restart in range(20):
            if time.time() - start_time > DEADLINE:
                break
            adj = [row[:] for row in best_adj]
            # Randomly remove a few edges and re-add
            existing = [(u, v) for u in range(N) for v in range(u+1, N) if adj[u][v]]
            random.shuffle(existing)
            for (u, v) in existing[:3]:
                adj[u][v] = 0
                adj[v][u] = 0
            a, c = _local_search(adj, start_time, max_iterations=3000)
            if c > best_count:
                best_count = c
                best_adj = a
    
    return best_adj, best_count

def construct():
    """Return a list of edges [u, v] forming a valid graph on vertices 0..14."""
    start_time = time.time()
    random.seed(42)
    
    adj, count = _multi_start_local_search(start_time)
    
    # Final validation
    if not _is_valid(adj):
        # Fallback to star
        adj = [[0] * N for _ in range(N)]
        for v in range(1, N):
            adj[0][v] = 1
            adj[v][0] = 1
    
    # Build edge list with proper types
    result = []
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u][v]:
                result.append([int(u), int(v)])
    
    return result

# EVOLVE-BLOCK-END
