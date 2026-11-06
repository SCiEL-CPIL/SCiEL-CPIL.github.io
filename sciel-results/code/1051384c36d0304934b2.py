"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _triangles_from_adj(n, adj):
    tris = []
    for u in range(n):
        for v in list(adj[u]):
            if v > u:
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        tris.append((u, v, w))
    return tris

def _score_graph(n, edges):
    adj = _build_adj(n, edges)
    tris = _triangles_from_adj(n, adj)
    if not tris:
        return 10**9
    deg = [len(adj[i]) for i in range(n)]
    best = 0
    for u, v, w in tris:
        s = deg[u] + deg[v] + deg[w]
        if s > best:
            best = s
    return best

def _edge_set(edges):
    return frozenset(frozenset((u, v)) for u, v in edges)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    n = 16
    target_edges = 65

    # Start from K_{8,8} + one internal edge
    base_edges = [(u, v) for u in range(8) for v in range(8, 16)] + [(0, 1)]
    best_edges = list(base_edges)
    best_score = _score_graph(n, best_edges)
    best_adj = _build_adj(n, best_edges)
    best_tris = _triangles_from_adj(n, best_adj)
    best_deg = [len(best_adj[i]) for i in range(n)]

    # Also try some random initial structures
    rng = random.Random(42)

    # Local search: try edge swaps (remove one edge, add another) to minimize max triangle degree sum
    def _try_swap(adj, deg, tris, e_remove, e_add, n):
        """Try swapping e_remove for e_add. Return (new_adj, new_deg, new_tris, new_score) or None."""
        u_r, v_r = e_remove
        u_a, v_a = e_add

        # Check if e_add already exists
        if v_a in adj[u_a]:
            return None

        # Remove e_remove
        adj[u_r].discard(v_r)
        adj[v_r].discard(u_r)
        deg[u_r] -= 1
        deg[v_r] -= 1

        # Add e_add
        adj[u_a].add(v_a)
        adj[v_a].add(u_a)
        deg[u_a] += 1
        deg[v_a] += 1

        # Recompute triangles
        new_tris = _triangles_from_adj(n, adj)
        if not new_tris:
            # Revert
            adj[u_r].add(v_r)
            adj[v_r].add(u_r)
            deg[u_r] += 1
            deg[v_r] += 1
            adj[u_a].discard(v_a)
            adj[v_a].discard(u_a)
            deg[u_a] -= 1
            deg[v_a] -= 1
            return None

        new_score = 0
        for u, v, w in new_tris:
            s = deg[u] + deg[v] + deg[w]
            if s > new_score:
                new_score = s

        # Revert
        adj[u_r].add(v_r)
        adj[v_r].add(u_r)
        deg[u_r] += 1
        deg[v_r] += 1
        adj[u_a].discard(v_a)
        adj[v_a].discard(u_a)
        deg[u_a] -= 1
        deg[v_a] -= 1

        return (new_score, new_tris)

    # Generate candidate edges (all possible pairs)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Local search with multiple restarts
    current_adj = _build_adj(n, best_edges)
    current_deg = [len(current_adj[i]) for i in range(n)]
    current_tris = _triangles_from_adj(n, current_adj)
    current_score = _score_graph(n, best_edges)

    improved = True
    max_iters = 5000
    iter_count = 0

    while improved and iter_count < max_iters:
        if time.time() > deadline - 5.0:
            break
        improved = False
        iter_count += 1

        # Shuffle edge order for diversity
        current_edge_list = [(min(u,v), max(u,v)) for u, v in 
                             [(u, v) for u in range(n) for v in current_adj[u] if u < v]]
        rng.shuffle(current_edge_list)

        for e_remove in current_edge_list:
            if time.time() > deadline - 5.0:
                break

            # Try adding a random non-edge
            for _ in range(20):
                e_add = rng.choice(all_pairs)
                if e_add in current_adj[min(e_add[0], e_add[1])] or \
                   (min(e_add[0], e_add[1]) == e_remove[0] and max(e_add[0], e_add[1]) == e_remove[1]):
                    continue
                if max(e_add[0], e_add[1]) in current_adj[min(e_add[0], e_add[1])]:
                    continue

                result = _try_swap(current_adj, current_deg, current_tris, e_remove, e_add, n)
                if result is None:
                    continue
                new_score, new_tris = result

                if new_score < current_score:
                    # Accept the move
                    u_r, v_r = e_remove
                    u_a, v_a = e_add
                    current_adj[u_r].discard(v_r)
                    current_adj[v_r].discard(u_r)
                    current_deg[u_r] -= 1
                    current_deg[v_r] -= 1
                    current_adj[u_a].add(v_a)
                    current_adj[v_a].add(u_a)
                    current_deg[u_a] += 1
                    current_deg[v_a] += 1
                    current_tris = new_tris
                    current_score = new_score
                    improved = True

                    if current_score < best_score:
                        best_score = current_score
                        best_edges = [(min(u,v), max(u,v)) for u in range(n) for v in current_adj[u] if u < v]
                        best_adj = _build_adj(n, best_edges)
                        best_tris = _triangles_from_adj(n, best_adj)
                        best_deg = [len(best_adj[i]) for i in range(n)]
                    break  # Restart edge iteration after an improvement

    # Final verification
    final_edges = [(min(u,v), max(u,v)) for u in range(n) for v in best_adj[u] if u < v]
    assert len(final_edges) == target_edges, f"Expected {target_edges} edges, got {len(final_edges)}"
    assert len(set(frozenset(e) for e in final_edges)) == target_edges, "Duplicate edges found"

    return {'n': n, 'edges': [[u, v] for u, v in final_edges]}
# EVOLVE-BLOCK-END
