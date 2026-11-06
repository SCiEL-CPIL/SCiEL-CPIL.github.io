"""Improved search for small triangle degree sums on 16 vertices, 65 edges."""
import time
import random
import math

N = 16
TARGET_EDGES = 65
DEADLINE_MARGIN = 10  # seconds before hard kill


def _edges_to_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _degrees(adj):
    return [len(s) for s in adj]


def _triangles(adj):
    tris = []
    for u in range(N):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        tris.append((u, v, w))
    return tris


def _score_graph(adj, tris, deg):
    if not tris:
        return 999
    return max(deg[u] + deg[v] + deg[w] for u, v, w in tris)


def _score_edges(edges):
    adj = _edges_to_adj(edges)
    deg = _degrees(adj)
    tris = _triangles(adj)
    return _score_graph(adj, tris, deg)


def _initial_edges():
    # K_{8,8} plus one intra-part edge
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return edges


def _all_pairs():
    return [(u, v) for u in range(N) for v in range(u + 1, N)]


def _edge_index(edges):
    """Map (u,v) with u<v to index in edges list."""
    m = {}
    for i, (u, v) in enumerate(edges):
        a, b = (u, v) if u < v else (v, u)
        m[(a, b)] = i
    return m


def _quick_score_delta(adj, deg, add_pair, remove_edge):
    """
    Estimate the change in max triangle degree sum when adding add_pair and removing remove_edge.
    Returns (new_score_estimate, triangles_created, triangles_destroyed)
    """
    u, v = add_pair
    rem_u, rem_v = remove_edge
    
    # Triangles created by adding (u,v): for each w in adj[u] & adj[v]
    common_uv = adj[u] & adj[v]
    created = [(u, v, w) for w in common_uv]
    
    # Triangles destroyed by removing (rem_u, rem_v): for each w in adj[rem_u] & adj[rem_v]
    # But we need to be careful: after removing, adj changes
    # For estimation, use current adj
    common_r = adj[rem_u] & adj[rem_v]
    # Exclude u and v if they're in common_r (they might be)
    destroyed = [(rem_u, rem_v, w) for w in common_r if w != u and w != v]
    
    # Estimate new degrees
    new_deg = deg[:]
    new_deg[u] += 1
    new_deg[v] += 1
    new_deg[rem_u] -= 1
    new_deg[rem_v] -= 1
    
    # Compute max score over created triangles
    max_created = 0
    for (a, b, c) in created:
        s = new_deg[a] + new_deg[b] + new_deg[c]
        if s > max_created:
            max_created = s
    
    return max_created, created, destroyed


def _try_swap_fast(edges, ei, add_pair, remove_idx, adj, deg, tris, current_score):
    """
    Attempt to add add_pair and remove edges[remove_idx].
    Uses quick check first, then full verification if needed.
    Returns new state or None.
    """
    u, v = add_pair
    if u == v:
        return None
    a, b = (u, v) if u < v else (v, u)
    if (a, b) in ei:
        return None
    
    rem_u, rem_v = edges[remove_idx]
    
    # Quick check: estimate new triangles from adding (u,v)
    common = adj[u] & adj[v]
    if common:
        max_new = 0
        for w in common:
            # deg[u] and deg[v] will increase by 1
            s = (deg[u] + 1) + (deg[v] + 1) + deg[w]
            if s > max_new:
                max_new = s
        if max_new > current_score:
            return None  # Skip if it immediately worsens max
    
    # Perform swap
    ra, rb = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
    
    # Update adj
    adj[u].add(v)
    adj[v].add(u)
    adj[rem_u].remove(rem_v)
    adj[rem_v].remove(rem_u)
    
    # Update degrees
    deg[u] += 1
    deg[v] += 1
    deg[rem_u] -= 1
    deg[rem_v] -= 1
    
    # Update edge list and index
    old_edge = edges[remove_idx]
    edges[remove_idx] = [u, v]
    del ei[(ra, rb)]
    ei[(a, b)] = remove_idx
    
    # Recompute triangles and score
    tris_new = _triangles(adj)
    score_new = _score_graph(adj, tris_new, deg)
    
    # Revert
    adj[u].remove(v)
    adj[v].remove(u)
    adj[rem_u].add(rem_v)
    adj[rem_v].add(rem_u)
    deg[u] -= 1
    deg[v] -= 1
    deg[rem_u] += 1
    deg[rem_v] += 1
    edges[remove_idx] = old_edge
    del ei[(a, b)]
    ei[(ra, rb)] = remove_idx
    
    if score_new < current_score:
        # Accept
        adj[u].add(v)
        adj[v].add(u)
        adj[rem_u].remove(rem_v)
        adj[rem_v].remove(rem_u)
        deg[u] += 1
        deg[v] += 1
        deg[rem_u] -= 1
        deg[rem_v] -= 1
        edges[remove_idx] = [u, v]
        del ei[(ra, rb)]
        ei[(a, b)] = remove_idx
        return (edges, ei, adj, deg, tris_new, score_new)
    else:
        return None


def _simulated_annealing_step(edges, ei, adj, deg, tris, current_score, temperature, all_pairs):
    """
    Perform one simulated annealing step: try a random swap, accept with probability
    based on score change and temperature.
    """
    if not all_pairs:
        return None
    
    # Pick random pair to add
    pair = random.choice(all_pairs)
    u, v = pair
    a, b = (u, v) if u < v else (v, u)
    if (a, b) in ei:
        return None
    
    # Pick random edge to remove
    idx = random.randint(0, len(edges) - 1)
    rem_u, rem_v = edges[idx]
    
    # Perform the swap
    ra, rb = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
    
    adj[u].add(v)
    adj[v].add(u)
    adj[rem_u].remove(rem_v)
    adj[rem_v].remove(rem_u)
    
    deg[u] += 1
    deg[v] += 1
    deg[rem_u] -= 1
    deg[rem_v] -= 1
    
    old_edge = edges[idx]
    edges[idx] = [u, v]
    del ei[(ra, rb)]
    ei[(a, b)] = idx
    
    tris_new = _triangles(adj)
    score_new = _score_graph(adj, tris_new, deg)
    
    # Decide whether to accept
    delta = score_new - current_score
    if delta <= 0:
        accept = True
    else:
        # Boltzmann acceptance
        if temperature <= 0:
            accept = False
        else:
            accept_prob = math.exp(-delta / temperature)
            accept = random.random() < accept_prob
    
    if accept:
        return (edges, ei, adj, deg, tris_new, score_new)
    else:
        # Revert
        adj[u].remove(v)
        adj[v].remove(u)
        adj[rem_u].add(rem_v)
        adj[rem_v].add(rem_u)
        deg[u] -= 1
        deg[v] -= 1
        deg[rem_u] += 1
        deg[rem_v] += 1
        edges[idx] = old_edge
        del ei[(a, b)]
        ei[(ra, rb)] = idx
        return None


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Start with the seed
    edges = _initial_edges()
    ei = _edge_index(edges)
    adj = _edges_to_adj(edges)
    deg = _degrees(adj)
    tris = _triangles(adj)
    current_score = _score_graph(adj, tris, deg)
    
    best_edges = edges[:]
    best_score = current_score
    
    all_pairs = _all_pairs()
    
    # Phase 1: Hill climbing
    while time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(all_pairs)
            for pair in all_pairs:
                if time.time() > deadline:
                    break
                u, v = pair
                if (min(u, v), max(u, v)) in ei:
                    continue
                for idx in range(len(edges)):
                    if time.time() > deadline:
                        break
                    result = _try_swap_fast(edges, ei, pair, idx, adj, deg, tris, current_score)
                    if result is not None:
                        edges, ei, adj, deg, tris, current_score = result
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = edges[:]
                        improved = True
                        break
        
        # Phase 2: Simulated annealing to escape local optima
        if time.time() < deadline:
            # Restore best
            edges = best_edges[:]
            ei = _edge_index(edges)
            adj = _edges_to_adj(edges)
            deg = _degrees(adj)
            tris = _triangles(adj)
            current_score = _score_graph(adj, tris, deg)
            
            # Annealing schedule
            t0 = 5.0
            t_end = 0.1
            cooling = 0.95
            temperature = t0
            
            while temperature > t_end and time.time() < deadline:
                result = _simulated_annealing_step(edges, ei, adj, deg, tris, current_score, temperature, all_pairs)
                if result is not None:
                    edges, ei, adj, deg, tris, current_score = result
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = edges[:]
                temperature *= cooling
            
            # Final hill climb after annealing
            improved = True
            while improved and time.time() < deadline:
                improved = False
                random.shuffle(all_pairs)
                for pair in all_pairs:
                    if time.time() > deadline:
                        break
                    u, v = pair
                    if (min(u, v), max(u, v)) in ei:
                        continue
                    for idx in range(len(edges)):
                        if time.time() > deadline:
                            break
                        result = _try_swap_fast(edges, ei, pair, idx, adj, deg, tris, current_score)
                        if result is not None:
                            edges, ei, adj, deg, tris, current_score = result
                            if current_score < best_score:
                                best_score = current_score
                                best_edges = edges[:]
                            improved = True
                            break
    
    return {'n': N, 'edges': best_edges}


if __name__ == "__main__":
    res = construct()
    adj = _edges_to_adj(res['edges'])
    deg = _degrees(adj)
    tris = _triangles(adj)
    print("Score:", _score_graph(adj, tris, deg))
    print("Edges:", len(res['edges']))
