"""Combined approach: hill-climbing + simulated annealing for triangle degree sums."""
import time
import random
import math

N = 16
TARGET_EDGES = 65
DEADLINE_MARGIN = 10  # seconds before hard kill (150s total - 140s safety)


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


def _initial_edges():
    # K_{8,8} plus one intra-part edge
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return edges


def _all_pairs():
    return [(u, v) for u in range(N) for v in range(u + 1, N)]


def _edge_index(edges):
    m = {}
    for i, (u, v) in enumerate(edges):
        a, b = (u, v) if u < v else (v, u)
        m[(a, b)] = i
    return m


def _quick_triangle_check(adj, deg, u, v):
    """Quick check: if adding edge (u,v) creates triangles, what's the max degree sum?"""
    common = adj[u] & adj[v]
    if not common:
        return 0, []
    max_sum = 0
    new_tris = []
    for w in common:
        s = (deg[u] + 1) + (deg[v] + 1) + deg[w]
        if s > max_sum:
            max_sum = s
        new_tris.append((min(u,v,w), min(max(u,v),w), max(u,v,w)))
    return max_sum, new_tris


def _find_worst_triangles(tris, deg, current_score, max_count=10):
    """Find triangles with degree sum closest to current_score, ordered by sum descending."""
    scored = []
    for tri in tris:
        s = deg[tri[0]] + deg[tri[1]] + deg[tri[2]]
        scored.append((s, tri))
    scored.sort(key=lambda x: -x[0])
    # Return only triangles at or near the worst score
    worst = scored[0][0] if scored else 0
    result = []
    for s, tri in scored:
        if s >= worst - 1:  # within 1 of worst
            result.append((s, tri))
        if len(result) >= max_count:
            break
    return result


def _degree_ordered_swap_search(adj, deg, edges, ei, tris, current_score, best_score, best_edges, deadline):
    """
    Degree-ordered edge-replacement heuristic:
    For each worst triangle, try to remove one of its edges and add a non-edge
    that doesn't create worse triangles, prioritizing moves that reduce degree
    of the highest-degree vertex in the worst triangle.
    """
    improved_any = False
    
    while time.time() < deadline:
        tris = _triangles(adj)
        if not tris:
            break
        current_score = _score_graph(adj, tris, deg)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [e[:] for e in edges]
        
        # Find worst triangles
        worst_tris = _find_worst_triangles(tris, deg, current_score, max_count=5)
        
        if not worst_tris:
            break
        
        found_improvement = False
        
        for tri_score, tri in worst_tris:
            if time.time() > deadline:
                break
            
            u, v, w = tri
            # Sort by degree descending to prioritize reducing high-degree vertices
            verts = sorted([u, v, w], key=lambda x: -deg[x])
            
            for rem_vertex in verts:
                if time.time() > deadline:
                    break
                
                # Try removing each edge incident to rem_vertex in this triangle
                for other1, other2 in [(u, v), (u, w), (v, w)]:
                    if rem_vertex not in (other1, other2):
                        continue
                    
                    rem_u, rem_v = other1, other2
                    a, b = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
                    if (a, b) not in ei:
                        continue
                    rem_idx = ei[(a, b)]
                    
                    # Find candidate non-edges to add
                    # Prioritize adding edges that don't create triangles with high degree sums
                    candidates = []
                    for i in range(N):
                        for j in range(i + 1, N):
                            if (i, j) in ei:
                                continue
                            # Quick check: would adding (i,j) create triangles worse than current?
                            common = adj[i] & adj[j]
                            if common:
                                max_new = 0
                                for k in common:
                                    s = (deg[i] + 1) + (deg[j] + 1) + deg[k]
                                    if s > max_new:
                                        max_new = s
                                if max_new > current_score:
                                    continue
                            # Score this candidate: prefer edges that reduce degree concentration
                            # Higher score = better candidate
                            # Prefer edges between lower-degree vertices or that break up high-degree clusters
                            cand_score = -(deg[i] + deg[j])  # prefer lower degree sum
                            candidates.append((cand_score, i, j))
                    
                    # Sort candidates by preference
                    candidates.sort(key=lambda x: -x[0])
                    
                    for cand_score, add_u, add_v in candidates[:20]:  # Try top 20
                        if time.time() > deadline:
                            break
                        
                        ca, cb = (add_u, add_v) if add_u < add_v else (add_v, add_u)
                        if (ca, cb) in ei:
                            continue
                        
                        # Perform swap
                        adj[add_u].add(add_v)
                        adj[add_v].add(add_u)
                        adj[rem_u].remove(rem_v)
                        adj[rem_v].remove(rem_u)
                        
                        deg[add_u] += 1
                        deg[add_v] += 1
                        deg[rem_u] -= 1
                        deg[rem_v] -= 1
                        
                        edges[rem_idx] = [add_u, add_v]
                        del ei[(a, b)]
                        ei[(ca, cb)] = rem_idx
                        
                        # Recompute
                        tris_new = _triangles(adj)
                        score_new = _score_graph(adj, tris_new, deg)
                        
                        if score_new < current_score:
                            current_score = score_new
                            tris = tris_new
                            if current_score < best_score:
                                best_score = current_score
                                best_edges = [e[:] for e in edges]
                            found_improvement = True
                            improved_any = True
                            break  # Restart with new state
                        else:
                            # Revert
                            adj[add_u].remove(add_v)
                            adj[add_v].remove(add_u)
                            adj[rem_u].add(rem_v)
                            adj[rem_v].add(rem_u)
                            
                            deg[add_u] -= 1
                            deg[add_v] -= 1
                            deg[rem_u] += 1
                            deg[rem_v] += 1
                            
                            edges[rem_idx] = [rem_u, rem_v]
                            del ei[(ca, cb)]
                            ei[(a, b)] = rem_idx
                        
                        if found_improvement:
                            break
                    
                    if found_improvement:
                        break
                
                if found_improvement:
                    break
            
            if found_improvement:
                continue  # Restart with new state
            else:
                break  # No improvement found, exit loop
    
    return best_score, best_edges, improved_any


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
    
    best_edges = [e[:] for e in edges]
    best_score = current_score
    
    all_pairs = _all_pairs()
    
    # Simulated annealing parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    # Phase 1: Simulated annealing
    while time.time() < deadline and T > T_min:
        # Try a random swap
        u, v = random.choice(all_pairs)
        if (min(u,v), max(u,v)) in ei:
            T *= cooling
            continue
        
        # Pick a random edge to remove
        idx = random.randint(0, len(edges) - 1)
        rem_u, rem_v = edges[idx]
        
        a, b = (u, v) if u < v else (v, u)
        ra, rb = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
        
        # Perform swap
        adj[u].add(v)
        adj[v].add(u)
        adj[rem_u].remove(rem_v)
        adj[rem_v].remove(rem_u)
        
        deg[u] += 1
        deg[v] += 1
        deg[rem_u] -= 1
        deg[rem_v] -= 1
        
        edges[idx] = [u, v]
        del ei[(ra, rb)]
        ei[(a, b)] = idx
        
        # Recompute score
        tris_new = _triangles(adj)
        score_new = _score_graph(adj, tris_new, deg)
        
        delta = score_new - current_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_score = score_new
            tris = tris_new
            if current_score < best_score:
                best_score = current_score
                best_edges = [e[:] for e in edges]
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
            
            edges[idx] = [rem_u, rem_v]
            del ei[(a, b)]
            ei[(ra, rb)] = idx
        
        T *= cooling
    
    # Phase 2: Degree-ordered swap search (NEW PRIMARY PHASE)
    edges = [e[:] for e in best_edges]
    ei = _edge_index(edges)
    adj = _edges_to_adj(edges)
    deg = _degrees(adj)
    tris = _triangles(adj)
    current_score = _score_graph(adj, tris, deg)
    
    best_score, best_edges, _ = _degree_ordered_swap_search(
        adj, deg, edges, ei, tris, current_score, best_score, best_edges, deadline
    )
    
    # Phase 3: Hill climbing from best found
    edges = best_edges[:]
    ei = _edge_index(edges)
    adj = _edges_to_adj(edges)
    deg = _degrees(adj)
    tris = _triangles(adj)
    current_score = _score_graph(adj, tris, deg)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(all_pairs)
        for pair in all_pairs:
            if time.time() > deadline:
                break
            u, v = pair
            if (min(u,v), max(u,v)) in ei:
                continue
            
            # Try removing each existing edge
            for idx in range(len(edges)):
                if time.time() > deadline:
                    break
                rem_u, rem_v = edges[idx]
                a, b = (u, v) if u < v else (v, u)
                ra, rb = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
                
                # Quick check: will adding (u,v) create bad triangles?
                max_new, _ = _quick_triangle_check(adj, deg, u, v)
                if max_new > current_score:
                    continue
                
                # Perform swap
                adj[u].add(v)
                adj[v].add(u)
                adj[rem_u].remove(rem_v)
                adj[rem_v].remove(rem_u)
                
                deg[u] += 1
                deg[v] += 1
                deg[rem_u] -= 1
                deg[rem_v] -= 1
                
                edges[idx] = [u, v]
                del ei[(ra, rb)]
                ei[(a, b)] = idx
                
                tris_new = _triangles(adj)
                score_new = _score_graph(adj, tris_new, deg)
                
                if score_new < current_score:
                    current_score = score_new
                    tris = tris_new
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = [e[:] for e in edges]
                    improved = True
                    break  # Restart pair loop
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
                    
                    edges[idx] = [rem_u, rem_v]
                    del ei[(a, b)]
                    ei[(ra, rb)] = idx
    
    # Phase 4: Final polish - try all single-edge swaps
    edges = best_edges[:]
    ei = _edge_index(edges)
    adj = _edges_to_adj(edges)
    deg = _degrees(adj)
    tris = _triangles(adj)
    current_score = _score_graph(adj, tris, deg)
    
    for pair in all_pairs:
        if time.time() > deadline:
            break
        u, v = pair
        if (min(u,v), max(u,v)) in ei:
            continue
        
        for idx in range(len(edges)):
            if time.time() > deadline:
                break
            rem_u, rem_v = edges[idx]
            a, b = (u, v) if u < v else (v, u)
            ra, rb = (rem_u, rem_v) if rem_u < rem_v else (rem_v, rem_u)
            
            adj[u].add(v)
            adj[v].add(u)
            adj[rem_u].remove(rem_v)
            adj[rem_v].remove(rem_u)
            
            deg[u] += 1
            deg[v] += 1
            deg[rem_u] -= 1
            deg[rem_v] -= 1
            
            edges[idx] = [u, v]
            del ei[(ra, rb)]
            ei[(a, b)] = idx
            
            tris_new = _triangles(adj)
            score_new = _score_graph(adj, tris_new, deg)
            
            if score_new < current_score:
                current_score = score_new
                tris = tris_new
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [e[:] for e in edges]
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
                
                edges[idx] = [rem_u, rem_v]
                del ei[(a, b)]
                ei[(ra, rb)] = idx
    
    # Verify edge count
    edge_set = set()
    for u, v in best_edges:
        edge_set.add((min(u, v), max(u, v)))
    
    if len(edge_set) != TARGET_EDGES:
        # Fall back to baseline
        best_edges = _initial_edges()
    
    return {'n': N, 'edges': best_edges}


if __name__ == "__main__":
    res = construct()
    adj = _edges_to_adj(res['edges'])
    deg = _degrees(adj)
    tris = _triangles(adj)
    print("Score:", _score_graph(adj, tris, deg))
    print("Edges:", len(res['edges']))
