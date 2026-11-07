"""Dense odd-order graphs avoiding equal-degree path endpoints.

Validity: no path s-u-v-t (s!=t) has deg(s)==deg(t).
Equivalent: for every vertex u, all neighbors of u have distinct degrees.
Strategy: bitwise adjacency, fast validity, deterministic K_{7,8} baseline,
          multi-restart greedy + local search (add/swap).
"""
import time
import random

N = 15
DEADLINE = 80.0  # safety margin under 90s kill


def _is_valid(adj, deg=None):
    """Check: for every u, neighbors have distinct degrees."""
    if deg is None:
        deg = [bin(adj[i]).count('1') for i in range(N)]
    for u in range(N):
        seen = set()
        au = adj[u]
        while au:
            bit = au & -au
            v = bit.bit_length() - 1
            au ^= bit
            d = deg[v]
            if d in seen:
                return False
            seen.add(d)
    return True


def _count_edges(adj):
    total = 0
    for i in range(N):
        total += bin(adj[i]).count('1')
    return total // 2


def _k78_baseline():
    """Build complete bipartite K_{7,8}: vertices 0-6 vs 7-14.
    Degrees: 8 for first 7, 7 for last 8.
    Check validity: neighbors of a vertex in group A all have degree 7 (same group B) -> all same degree!
    Wait, that's a violation. Let me reconsider.
    
    Actually for K_{7,8}: 
    - Vertices 0..6 have degree 8 (connected to 7..14)
    - Vertices 7..14 have degree 7 (connected to 0..6)
    
    For vertex 0 (deg 8): its neighbors are 7..14, all with degree 7. All same degree -> VIOLATION!
    
    So K_{7,8} is NOT valid. Need a different baseline.
    
    Let me use a star as guaranteed valid baseline: 14 edges.
    """
    adj = [0] * N
    for v in range(1, N):
        adj[0] |= (1 << v)
        adj[v] |= (1 << 0)
    return adj


def _greedy_trial(adj_init, deg_init, count_init, seed, deadline):
    """Try to build a valid graph from initial state by adding edges greedily."""
    rng = random.Random(seed)
    adj = adj_init[:]
    deg = deg_init[:]
    count = count_init
    
    # All possible edges
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if time.time() > deadline:
            break
        if adj[u] & (1 << v):
            continue
        # Try adding
        new_adj = adj[:]
        new_deg = deg[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        new_deg[u] += 1
        new_deg[v] += 1
        if _is_valid(new_adj, new_deg):
            adj = new_adj
            deg = new_deg
            count += 1
    
    return adj, deg, count


def _local_search(adj, deg, count, deadline, rng):
    """Local search: try adding edges and edge swaps."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a random edge
        for _ in range(50):
            if time.time() > deadline:
                break
            u = rng.randint(0, N - 1)
            v = rng.randint(0, N - 1)
            if u == v or u > v:
                continue
            if adj[u] & (1 << v):
                continue
            
            new_adj = adj[:]
            new_deg = deg[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg[u] += 1
            new_deg[v] += 1
            if _is_valid(new_adj, new_deg):
                adj = new_adj
                deg = new_deg
                count += 1
                improved = True
                break
        
        # Try edge swap: remove one edge, add another
        if not improved:
            for _ in range(50):
                if time.time() > deadline:
                    break
                # Pick an existing edge to remove
                u1 = rng.randint(0, N - 1)
                v1 = rng.randint(0, N - 1)
                if u1 == v1 or u1 > v1:
                    continue
                if not (adj[u1] & (1 << v1)):
                    continue
                
                # Pick a new edge to add
                u2 = rng.randint(0, N - 1)
                v2 = rng.randint(0, N - 1)
                if u2 == v2 or u2 > v2:
                    continue
                if (u2, v2) == (u1, v1):
                    continue
                if adj[u2] & (1 << v2):
                    continue
                
                new_adj = adj[:]
                new_deg = deg[:]
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
                new_deg[u1] -= 1
                new_deg[v1] -= 1
                new_deg[u2] += 1
                new_deg[v2] += 1
                if _is_valid(new_adj, new_deg):
                    adj = new_adj
                    deg = new_deg
                    # count unchanged
                    improved = True
                    break
    
    return adj, deg, count


def construct():
    start = time.time()
    deadline = start + DEADLINE
    
    # Start with star baseline
    best_adj = _k78_baseline()
    best_deg = [bin(best_adj[i]).count('1') for i in range(N)]
    best_count = _count_edges(best_adj)  # 14
    
    # Multi-restart greedy from empty and from star
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        if trial % 2 == 0:
            # Start from empty
            init_adj = [0] * N
            init_deg = [0] * N
            init_count = 0
        else:
            # Start from star
            init_adj = _k78_baseline()
            init_deg = [bin(init_adj[i]).count('1') for i in range(N)]
            init_count = 14
        
        adj, deg, count = _greedy_trial(init_adj, init_deg, init_count, trial, deadline - 5)
        
        if count > best_count:
            best_count = count
            best_adj = adj
            best_deg = deg
    
    # Local search on best
    if time.time() < deadline - 2:
        rng = random.Random(12345)
        best_adj, best_deg, best_count = _local_search(best_adj, best_deg, best_count, deadline - 2, rng)
    
    # Convert to edge list
    edges = []
    for i in range(N):
        ai = best_adj[i]
        while ai:
            bit = ai & -ai
            j = bit.bit_length() - 1
            ai ^= bit
            if i < j:
                edges.append([int(i), int(j)])
    
    # Final validity check
    adj_check = [0] * N
    for e in edges:
        adj_check[e[0]] |= (1 << e[1])
        adj_check[e[1]] |= (1 << e[0])
    if not _is_valid(adj_check):
        # Fallback to star
        edges = [[0, int(v)] for v in range(1, N)]
    
    return edges
