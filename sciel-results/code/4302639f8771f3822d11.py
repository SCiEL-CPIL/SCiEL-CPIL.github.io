"""Rewritten program: constraint-propagation / backtracking search for densest valid graph."""
import time
import random
from itertools import combinations

N = 15
DEADLINE = 78.0  # safety margin before 90s kill


def _check_edge(adj, u, v):
    """
    Check whether adding edge (u,v) to the current graph (given by adj, with (u,v) NOT yet added)
    would violate the constraint. We need to verify that no simple 3-edge path s-u-v-t (or s-v-u-t)
    has deg(s) == deg(t) in the FULL graph after adding (u,v).
    
    Actually, the constraint is on the full graph. So we need to check:
    - For every path s-u-v-t in the full graph (after adding u-v), deg(s) != deg(t).
    - For every path s-v-u-t in the full graph (after adding u-v), deg(s) != deg(t).
    
    But degrees change when we add the edge. Let's think carefully.
    
    Let deg_old[x] be the degree of x before adding (u,v).
    After adding (u,v): deg_new[u] = deg_old[u]+1, deg_new[v] = deg_old[v]+1, others unchanged.
    
    A path s-u-v-t exists if s~u and v~t (in the full graph after adding).
    s != v (simple path), t != u, s != t.
    We need deg_new[s] != deg_new[t].
    
    Similarly for s-v-u-t: s~v and u~t, s != u, t != v, s != t, need deg_new[s] != deg_new[t].
    
    So we can compute deg_new and check all such paths.
    """
    # Compute new degrees
    deg = [sum(adj[x]) for x in range(N)]
    deg[u] += 1
    deg[v] += 1
    
    # Check paths s-u-v-t
    for s in range(N):
        if s == v or s == u:
            continue
        if adj[u][s] == 0:
            continue
        for t in range(N):
            if t == u or t == s or t == v:
                continue
            if adj[v][t] == 0:
                continue
            if deg[s] == deg[t]:
                return False
    
    # Check paths s-v-u-t
    for s in range(N):
        if s == u or s == v:
            continue
        if adj[v][s] == 0:
            continue
        for t in range(N):
            if t == v or t == s or t == u:
                continue
            if adj[u][t] == 0:
                continue
            if deg[s] == deg[t]:
                return False
    
    return True


def _is_valid_full(adj):
    """Full validation of the graph."""
    deg = [sum(adj[x]) for x in range(N)]
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u][v] == 0:
                continue
            # Path s-u-v-t
            for s in range(N):
                if s == u or s == v:
                    continue
                if adj[u][s] == 0:
                    continue
                for t in range(N):
                    if t == u or t == v or t == s:
                        continue
                    if adj[v][t] == 0:
                        continue
                    if deg[s] == deg[t]:
                        return False
            # Path s-v-u-t
            for s in range(N):
                if s == u or s == v:
                    continue
                if adj[v][s] == 0:
                    continue
                for t in range(N):
                    if t == u or t == v or t == s:
                        continue
                    if adj[u][t] == 0:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True


def _backtrack_search(adj, edges_remaining, current_count, best_count, best_edges, deadline):
    """
    Backtracking search to find a valid subgraph of edges_remaining.
    adj: current adjacency matrix
    edges_remaining: list of (u,v) edges still to consider
    current_count: number of edges currently in adj
    best_count: best count found so far
    best_edges: best edges found so far
    deadline: time limit
    """
    if time.time() > deadline:
        return best_count, best_edges
    
    if current_count >= best_count:
        # Update best
        if current_count > best_count:
            best_count = current_count
            best_edges = [[u, v] for u in range(N) for v in range(u + 1, N) if adj[u][v]]
    
    if not edges_remaining:
        return best_count, best_edges
    
    # Upper bound: current_count + len(edges_remaining)
    if current_count + len(edges_remaining) <= best_count:
        return best_count, best_edges
    
    # Pick an edge to try next
    u, v = edges_remaining[0]
    rest = edges_remaining[1:]
    
    # Option 1: Add edge (u,v) if valid
    if _check_edge(adj, u, v):
        adj[u][v] = 1
        adj[v][u] = 1
        bc, be = _backtrack_search(adj, rest, current_count + 1, best_count, best_edges, deadline)
        adj[u][v] = 0
        adj[v][u] = 0
    else:
        bc, be = best_count, best_edges
    
    # Option 2: Skip edge (u,v)
    bc2, be2 = _backtrack_search(adj, rest, current_count, best_count, best_edges, deadline)
    
    if bc2 > bc:
        bc, be = bc2, be2
    
    return bc, be


def _greedy_with_backtracking():
    """
    Greedy approach with local backtracking for improvement.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(all_edges)
        adj = [[0] * N for _ in range(N)]
        current_count = 0
        
        for (u, v) in all_edges:
            if time.time() > deadline:
                break
            if _check_edge(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
                current_count += 1
        
        if current_count > best_count:
            best_count = current_count
            best_edges = [[u, v] for u in range(N) for v in range(u + 1, N) if adj[u][v]]
    
    return best_edges, best_count


def _local_search(edges, deadline):
    """
    Try to improve a given edge set by removing/adding edges.
    """
    if not edges:
        return edges, 0
    
    adj = [[0] * N for _ in range(N)]
    for (u, v) in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    
    current_count = len(edges)
    best_edges = edges[:]
    best_count = current_count
    start_time = time.time()
    
    # Try removing one edge and then greedily re-adding
    for iter_count in range(50):
        if time.time() > deadline:
            break
        
        # Try removing a random edge
        if not best_edges:
            break
        idx = random.randint(0, len(best_edges) - 1)
        u, v = best_edges[idx]
        adj[u][v] = 0
        adj[v][u] = 0
        new_edges = [e for i, e in enumerate(best_edges) if i != idx]
        
        # Greedily add back edges
        all_other_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj[i][j]]
        random.shuffle(all_other_edges)
        
        for (a, b) in all_other_edges:
            if time.time() > deadline:
                break
            if _check_edge(adj, a, b):
                adj[a][b] = 1
                adj[b][a] = 1
                new_edges.append([a, b])
        
        if len(new_edges) > best_count:
            best_count = len(new_edges)
            best_edges = new_edges[:]
    
    return best_edges, best_count


def construct():
    """
    Return a list of edges [u, v] forming a valid graph on vertices 0..14.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    random.seed(42)
    
    # Phase 1: Greedy with multiple restarts
    edges, count = _greedy_with_backtracking()
    
    # Phase 2: Local search to improve
    if time.time() < deadline:
        edges, count = _local_search(edges, deadline)
    
    # Final validation
    if edges:
        adj = [[0] * N for _ in range(N)]
        for (u, v) in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        if not _is_valid_full(adj):
            # Fallback: try a simple star
            edges = [[0, v] for v in range(1, N)]
            adj = [[0] * N for _ in range(N)]
            for (u, v) in edges:
                adj[u][v] = 1
                adj[v][u] = 1
    
    # Ensure proper types
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result


# EVOLVE-BLOCK-END
