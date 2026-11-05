import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_longest_cycle_karp(adj, n):
    """
    Finds the longest simple cycle in the graph using Karp's algorithm on the directed representation.
    Returns the cycle as a list of vertices or None.
    """
    if n < 3:
        return None

    # Build adjacency matrix A
    # A[u][v] = 1 if edge (u,v) exists, 0 otherwise
    A = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            A[u][v] = 1

    # DP[k][u][v] = length of longest path from u to v with exactly k edges
    # We only need up to n edges
    # Initialize DP[1]
    DP = [None] * (n + 1)
    DP[1] = A

    # Compute DP[k] for k = 2 to n
    for k in range(2, n + 1):
        DP[k] = [[-1] * n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                max_len = -1
                # Transition: max over w of DP[k-1][u][w] + A[w][v]
                for w in range(n):
                    if DP[k-1][u][w] != -1 and A[w][v] == 1:
                        val = DP[k-1][u][w] + 1
                        if val > max_len:
                            max_len = val
                DP[k][u][v] = max_len

    # Find the longest cycle
    best_len = 0
    best_v = -1
    for L in range(n, 2, -1):
        found = False
        for v in range(n):
            if DP[L][v][v] > best_len:
                best_len = DP[L][v][v]
                best_v = v
                found = True
        if found:
            break

    if best_v == -1:
        return None

    # Reconstruct the cycle
    # We have a cycle of length best_len starting and ending at best_v
    # Path: best_v -> w_1 -> ... -> w_{best_len-1} -> best_v
    # DP[best_len][best_v][best_v] = max_w (DP[best_len-1][best_v][w] + A[w][best_v])
    
    path_nodes = [best_v]
    cur_end = best_v
    cur_len = best_len
    cur_start = best_v
    
    while cur_len > 1:
        # Find w such that DP[cur_len-1][cur_start][w] + A[w][cur_end] == DP[cur_len][cur_start][cur_end]
        target_len = DP[cur_len][cur_start][cur_end]
        found_w = -1
        for w in range(n):
            if DP[cur_len-1][cur_start][w] != -1 and A[w][cur_end] == 1:
                if DP[cur_len-1][cur_start][w] + 1 == target_len:
                    found_w = w
                    break
        if found_w == -1:
            # Should not happen
            return None
        path_nodes.append(found_w)
        cur_end = found_w
        cur_len -= 1
        
    # path_nodes is [best_v, w_{L-1}, w_{L-2}, ..., w_1]
    # The cycle is best_v -> w_1 -> ... -> w_{L-1} -> best_v
    # So we need to reverse the collected nodes (excluding the first one which is the start)
    # Actually, the list is [v, w_{L-1}, ..., w_1].
    # The cycle order is v, w_1, w_2, ..., w_{L-1}.
    # So we take path_nodes[0], then path_nodes[-1], path_nodes[-2], ..., path_nodes[1]
    cycle = [path_nodes[0]] + path_nodes[:0:-1]
    
    # Verify the cycle is valid
    # Check edges exist
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        if A[u][v] == 0:
            return None # Fallback to greedy if reconstruction fails

    return cycle


def _decompose_greedy(edges, n, deadline, rng=None):
    """
    Greedy decomposition: repeatedly find the longest cycle and remove it.
    """
    if rng is None:
        rng = random.Random(12345)

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)

    while remaining > 0:
        if time.time() > deadline:
            break

        # Try to find the longest cycle using Karp's algorithm
        best_cycle = _find_longest_cycle_karp(adj, n)

        if best_cycle:
            m = len(best_cycle)
            # Remove edges
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(best_cycle)
        else:
            # No cycle found, remove a single edge
            edge_found = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_found = True
                        break
                if edge_found:
                    break
            if not edge_found:
                break

    # Add any remaining edges as single edges
    if remaining > 0:
        for u in range(n):
            for v in list(adj[u]):
                if u < v:
                    pieces.append([u, v])

    return pieces


def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')

    # The Karp-based greedy is deterministic and strong.
    # We can try it once. If we have time, we might try randomizing the graph orientation?
    # No, the graph is undirected.
    # We can try different tie-breaking in Karp?
    # Or just run it once. It should be very good.
    
    pieces = _decompose_greedy(edges, n, deadline)
    return pieces


def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)

        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
