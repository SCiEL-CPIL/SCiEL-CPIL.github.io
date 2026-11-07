"""Optimized program: Fast bitmask-based validity check and greedy/local search."""
# EVOLVE-BLOCK-START
import time
import random

N = 15
DEADLINE = 55.0  # Safety margin

def _is_valid_fast(adj):
    """
    Check if graph is valid: no simple 3-edge path s-u-v-t has deg(s) == deg(t).
    Optimized by grouping neighbors by degree.
    Complexity: O(E * (deg(u) + deg(v))) roughly.
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
                
    # For each edge (u, v), check if there exists s in N(u)\{v} and t in N(v)\{u} 
    # such that deg(s) == deg(t).
    for u in range(N):
        for v in neighbors[u]:
            if u >= v: continue # only check each edge once
            
            # Build set of degrees of neighbors of u (excluding v)
            degs_u = set()
            for s in neighbors[u]:
                if s != v:
                    degs_u.add(deg[s])
            
            # Check if any neighbor t of v (excluding u) has degree in degs_u
            for t in neighbors[v]:
                if t != u:
                    if deg[t] in degs_u:
                        return False
    return True

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

def _edge_count(adj):
    return sum(sum(row) for row in adj) // 2

def _greedy_construct(start_time, seed):
    random.seed(seed)
    adj = [[0] * N for _ in range(N)]
    edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    random.shuffle(edges)
    
    for u, v in edges:
        if time.time() - start_time > DEADLINE:
            break
        _try_add(adj, u, v)
    
    return adj

def _local_search(adj, start_time, seed):
    random.seed(seed)
    current_count = _edge_count(adj)
    best_adj = [row[:] for row in adj]
    best_count = current_count
    no_improve = 0
    
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
            no_improve = 0
            continue
            
        no_improve += 1
        if no_improve > 10:
            break
            
        # Try swaps
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) if adj[u][v]]
        random.shuffle(existing)
        random.shuffle(missing)
        
        for u1, v1 in existing:
            if time.time() - start_time > DEADLINE:
                break
            for u2, v2 in missing:
                if _try_swap(adj, u1, v1, u2, v2):
                    improved = True
                    break
            if improved:
                break
                
        if not improved:
            break
            
    return best_adj, best_count

def construct():
    start_time = time.time()
    best_adj = [[0] * N for _ in range(N)]
    best_count = 0
    
    # Run multiple greedy starts
    for i in range(50):
        if time.time() - start_time > DEADLINE:
            break
        adj = _greedy_construct(start_time, seed=1000 + i)
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
            
    # Local search on the best greedy result
    if time.time() - start_time < DEADLINE - 5:
        adj, count = _local_search(best_adj, start_time, seed=42)
        if count > best_count:
            best_count = count
            best_adj = adj
            
    # Final validation
    if not _is_valid_fast(best_adj):
        # Fallback to star
        best_adj = [[0] * N for _ in range(N)]
        for v in range(1, N):
            best_adj[0][v] = 1
            best_adj[v][0] = 1
        best_count = N - 1

    result = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u][v]:
                result.append([int(u), int(v)])
    
    return result

# EVOLVE-BLOCK-END
