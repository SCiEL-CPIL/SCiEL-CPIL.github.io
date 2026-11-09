import time
import random
import numpy as np

N = 20
DEADLINE = 115.0  # safety margin before the 150s kill

def _popcount(x):
    return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

def _can_add_edge_fast(adj, u, v):
    """
    Check if adding edge (u,v) to a K3,3-free graph keeps it K3,3-free.
    adj is a list of int bitmasks.
    """
    nu = adj[u]
    nv = adj[v]

    # Candidates for B-side (neighbors of u, excluding v)
    b_mask = nu & ~(1 << v)
    # Candidates for A-side (neighbors of v, excluding u)
    a_mask = nv & ~(1 << u)

    if _popcount(b_mask) < 2 or _popcount(a_mask) < 2:
        return True

    a_list = [i for i in range(N) if (a_mask >> i) & 1]
    b_list = [i for i in range(N) if (b_mask >> i) & 1]

    if len(a_list) < len(b_list):
        for i in range(len(a_list)):
            a1 = a_list[i]
            adj_a1 = adj[a1]
            for j in range(i + 1, len(a_list)):
                a2 = a_list[j]
                adj_a2 = adj[a2]
                common = adj_a1 & adj_a2 & b_mask
                if _popcount(common) >= 2:
                    return False
    else:
        for i in range(len(b_list)):
            b1 = b_list[i]
            adj_b1 = adj[b1]
            for j in range(i + 1, len(b_list)):
                b2 = b_list[j]
                adj_b2 = adj[b2]
                common = adj_b1 & adj_b2 & a_mask
                if _popcount(common) >= 2:
                    return False
    return True

def _edge_count(adj):
    return sum(_popcount(adj[u]) for u in range(N)) // 2

def _greedy_fill(adj, time_limit):
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() - start_time > time_limit:
                    return adj
                if (adj[u] >> v) & 1:
                    continue
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
    return adj

def _structured_start():
    """
    Build a dense, highly symmetric starting graph using a polarity-graph-style
    construction. We use a 5x4 block-circulant design and greedily add edges
    that respect K3,3-freedom.
    """
    adj = [0] * N

    # 5 blocks of 4 vertices each
    for i in range(5):
        for j in range(5):
            if i == j:
                # Within block: cycle
                for a in range(4):
                    b = (a + 1) % 4
                    u = i * 4 + a
                    v = i * 4 + b
                    if u < v:
                        if _can_add_edge_fast(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
            else:
                offset = (j - i) % 5
                for a in range(4):
                    for b in range(4):
                        u = i * 4 + a
                        v = j * 4 + b
                        if u >= v:
                            continue
                        # Choose connection pattern based on offset
                        if offset in (1, 4):
                            # Dense: connect if (b-a) mod 4 in {0,1}
                            if (b - a) % 4 in (0, 1):
                                if _can_add_edge_fast(adj, u, v):
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                        elif offset in (2, 3):
                            # Medium: connect if (b-a) mod 4 == 0
                            if (b - a) % 4 == 0:
                                if _can_add_edge_fast(adj, u, v):
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
    return adj

def _random_start(density):
    adj = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if random.random() < density:
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    return adj

def _local_search(adj, time_limit):
    """
    Targeted local search with 2-opt and 3-opt edge swaps, prioritizing
    high-degree vertices for removal and high-potential edges for addition.
    """
    start_time = time.time()
    current_count = _edge_count(adj)
    best_adj = adj[:]
    best_count = current_count

    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]

    while time.time() - start_time < time_limit:
        improved = False

        missing = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
        existing = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]

        # Sort existing edges by degree sum (remove high-impact edges first)
        existing.sort(key=lambda e: _popcount(adj[e[0]]) + _popcount(adj[e[1]]), reverse=True)
        # Sort missing edges by degree sum (add high-potential edges first)
        missing.sort(key=lambda e: _popcount(adj[e[0]]) + _popcount(adj[e[1]]), reverse=True)

        # 2-opt: remove 1 edge, add up to 2
        for u, v in existing:
            if time.time() - start_time > time_limit:
                return best_adj

            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

            added = 0
            new_edges = []
            for u2, v2 in missing:
                if added >= 2:
                    break
                if (adj[u2] >> v2) & 1:
                    continue
                if _can_add_edge_fast(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    added += 1
                    new_edges.append((u2, v2))

            if added >= 2:
                current_count += (added - 1)
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                for eu, ev in new_edges:
                    adj[eu] &= ~(1 << ev)
                    adj[ev] &= ~(1 << eu)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

        # 3-opt: remove 2 edges, add up to 3
        if time.time() - start_time < time_limit:
            for i in range(len(existing)):
                if time.time() - start_time > time_limit:
                    return best_adj
                u1, v1 = existing[i]
                for j in range(i + 1, len(existing)):
                    if time.time() - start_time > time_limit:
                        return best_adj
                    u2, v2 = existing[j]

                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)

                    added = 0
                    new_edges = []
                    for u3, v3 in missing:
                        if added >= 3:
                            break
                        if (adj[u3] >> v3) & 1:
                            continue
                        if _can_add_edge_fast(adj, u3, v3):
                            adj[u3] |= (1 << v3)
                            adj[v3] |= (1 << u3)
                            added += 1
                            new_edges.append((u3, v3))

                    if added >= 3:
                        current_count += (added - 2)
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
                    else:
                        for eu, ev in new_edges:
                            adj[eu] &= ~(1 << ev)
                            adj[ev] &= ~(1 << eu)
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)

        if not improved:
            break

    return best_adj

def _backtracking_maximize(adj, time_limit, target_count):
    """
    Recursive backtracking with MRV heuristic to find a maximal set of edges
    that can be added to the current graph. This acts as a global optimizer
    within the local neighborhood defined by the current adjacency.
    """
    start_time = time.time()
    best_adj = adj[:]
    best_count = _edge_count(adj)
    
    # List of all missing edges
    missing_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if not ((adj[u] >> v) & 1):
                missing_edges.append((u, v))
    
    # Sort by "potential" (degree sum) to prioritize high-impact edges
    missing_edges.sort(key=lambda e: _popcount(adj[e[0]]) + _popcount(adj[e[1]]), reverse=True)
    
    # We will use a stack-based or recursive approach. 
    # Given N=20, the number of missing edges is at most 190.
    # We try to add as many as possible.
    
    # To keep it simple and fast, we do a greedy pass first, then backtracking on the remainder.
    # Actually, let's just do a smart recursive search.
    
    def dfs(index, current_adj, current_count):
        nonlocal best_adj, best_count
        if time.time() - start_time > time_limit:
            return
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]
            
        if index >= len(missing_edges):
            return
            
        # MRV: Pick the next edge that has the most constraints? 
        # For now, stick to the sorted order which is a good heuristic.
        u, v = missing_edges[index]
        
        # Option 1: Add this edge
        if _can_add_edge_fast(current_adj, u, v):
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            dfs(index + 1, current_adj, current_count + 1)
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
        # Option 2: Skip this edge
        dfs(index + 1, current_adj, current_count)
        
    # To prevent stack overflow or excessive runtime on large missing sets, 
    # we limit the depth or use a heuristic cutoff. 
    # With 190 missing edges, full backtracking is impossible.
    # We only use this if the number of missing edges is small enough, or we use it as a polish.
    
    if len(missing_edges) < 50:
        dfs(0, adj[:], _edge_count(adj))
        
    return best_adj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_adj = [0] * N
    best_count = 0

    # Strategy 1: Structured algebraic start + greedy fill
    adj = _structured_start()
    adj = _greedy_fill(adj, min(5.0, deadline - time.time() - 10))
    current_count = _edge_count(adj)
    if current_count > best_count:
        best_adj = adj[:]
        best_count = current_count

    # Strategy 2: Local search on structured start
    remaining = deadline - time.time() - 5
    if remaining > 2:
        adj = best_adj[:]
        adj = _local_search(adj, remaining * 0.4)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

    # Strategy 3: Random starts with varied densities
    densities = [0.6, 0.7, 0.8, 0.5, 0.65, 0.75]
    random.shuffle(densities)

    for density in densities:
        if time.time() > deadline - 10:
            break

        adj = _random_start(density)

        remaining = deadline - time.time() - 5
        if remaining < 1:
            break
        adj = _greedy_fill(adj, min(remaining * 0.2, 2.0))

        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

        remaining = deadline - time.time() - 3
        if remaining < 1:
            break
        adj = _local_search(adj, remaining * 0.5)

        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

    # Final polish with backtracking if we have time and the graph is near-maximal
    remaining = deadline - time.time() - 1
    if remaining > 2:
        adj = best_adj[:]
        # Use backtracking to find the exact maximal extension for the current state
        # This is safe because the number of missing edges in a near-optimal graph is small
        adj = _backtracking_maximize(adj, remaining, best_count)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
