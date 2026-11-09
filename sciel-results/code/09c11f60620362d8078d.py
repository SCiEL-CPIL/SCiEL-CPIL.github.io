import time
import random
import numpy as np

N = 20
DEADLINE = 145.0  # safety margin before the 150s kill

def _popcount(x):
    return bin(x).count('1')

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

def _is_k33_free(adj):
    """
    Verify the graph is K3,3-free.
    A K3,3 subgraph is a bipartite subgraph with parts A and B of size 3,
    such that all 9 edges between them exist.
    """
    for a1 in range(N):
        for a2 in range(a1 + 1, N):
            for a3 in range(a2 + 1, N):
                # Common neighbors of a1, a2, a3
                common = adj[a1] & adj[a2] & adj[a3]
                if _popcount(common) >= 3:
                    return False
    return True

def _greedy_fill(adj, time_limit):
    """
    Greedily add all possible edges that keep the graph K3,3-free.
    """
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
    Build a dense, highly symmetric starting graph using a 5x4 block-circulant design.
    """
    adj = [0] * N

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
    """Generate a random graph with given edge probability, filtered for K3,3."""
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
    Local search with 1-2 swaps (remove 1, add 2) and 2-3 swaps (remove 2, add 3).
    """
    start_time = time.time()
    current_count = _edge_count(adj)
    best_adj = adj[:]
    best_count = current_count

    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]

    while time.time() - start_time < time_limit:
        improved = False

        # Compute degrees for sorting
        degrees = [_popcount(adj[u]) for u in range(N)]
        
        existing = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
        missing = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]

        # Sort existing edges by degree sum (remove high-impact edges first)
        existing.sort(key=lambda e: degrees[e[0]] + degrees[e[1]], reverse=True)
        # Sort missing edges by degree sum (add high-potential edges first)
        missing.sort(key=lambda e: degrees[e[0]] + degrees[e[1]], reverse=True)

        # 1-2 Swap: Remove 1 edge, try to add up to 2
        for u, v in existing:
            if time.time() - start_time > time_limit:
                return best_adj

            # Save state
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

            added = 0
            new_edges = []
            
            # Try to add missing edges
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

            if added > 1:
                # Net gain is added - 1. If added > 1, gain > 0.
                current_count += (added - 1)
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Revert
                for eu, ev in new_edges:
                    adj[eu] &= ~(1 << ev)
                    adj[ev] &= ~(1 << eu)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

        # 2-3 Swap: Remove 2 edges, try to add up to 3
        if time.time() - start_time < time_limit:
            for i in range(len(existing)):
                if time.time() - start_time > time_limit:
                    return best_adj
                u1, v1 = existing[i]
                for j in range(i + 1, len(existing)):
                    if time.time() - start_time > time_limit:
                        return best_adj
                    u2, v2 = existing[j]

                    # Save state
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)

                    added = 0
                    new_edges = []
                    
                    # Try to add missing edges
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

                    if added > 2:
                        # Net gain is added - 2. If added > 2, gain > 0.
                        current_count += (added - 2)
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
                    else:
                        # Revert
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

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_adj = [0] * N
    best_count = 0

    # Strategy 1: Structured start
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
        adj = _local_search(adj, remaining * 0.5)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

    # Strategy 3: Random starts with varied densities
    densities = [0.6, 0.7, 0.8, 0.5]
    random.shuffle(densities)

    for density in densities:
        if time.time() > deadline - 15:
            break

        adj = _random_start(density)

        # Greedy fill
        remaining = deadline - time.time() - 5
        if remaining < 1:
            break
        adj = _greedy_fill(adj, min(remaining * 0.3, 3.0))

        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

        # Local search
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        adj = _local_search(adj, remaining * 0.6)

        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

    # Final polish: local search on best found
    remaining = deadline - time.time() - 1
    if remaining > 2:
        adj = best_adj[:]
        adj = _local_search(adj, remaining)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
