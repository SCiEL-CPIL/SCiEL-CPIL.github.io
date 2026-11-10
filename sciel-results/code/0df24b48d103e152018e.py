import time
import random
from collections import defaultdict


def _instance(n, p, seed):
    """Exact copy of the verifier's instance generator."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _edge_set(edges):
    return {tuple(sorted(e)) for e in edges}


def _path_edges_set(path):
    """Return set of edges (as sorted tuples) in a path."""
    return {tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)}


def _is_simple(path):
    """Check if path visits each vertex at most once."""
    return len(set(path)) == len(path)


def _paths_from_edges_greedy(edge_list, adj, rng):
    """Cover edges with simple paths using greedy trail-based method with randomized choices."""
    remaining = set(_edge_set(edge_list))
    deg = defaultdict(int)
    for u, v in remaining:
        deg[u] += 1
        deg[v] += 1
    paths = []
    while remaining:
        # Pick a start vertex: prefer odd-degree, then leaves, then low-degree
        candidates = [v for v, d in deg.items() if d > 0]
        if not candidates:
            break
        # Strategy: pick from odd-degree vertices first, then leaves, then random low-degree
        odd = [v for v in candidates if deg[v] % 2 == 1]
        leaves = [v for v in candidates if deg[v] == 1]
        if odd:
            start = rng.choice(odd)
        elif leaves:
            start = rng.choice(leaves)
        else:
            # Pick a vertex with minimum degree, breaking ties randomly
            min_deg = min(deg[v] for v in candidates)
            low = [v for v in candidates if deg[v] == min_deg]
            start = rng.choice(low)
        
        path = [start]
        in_path = {start}
        while True:
            v = path[-1]
            nbrs = []
            for u in adj[v]:
                e = tuple(sorted((v, u)))
                if e in remaining:
                    nbrs.append(u)
            if not nbrs:
                break
            # Prefer unvisited neighbors; among those, prefer low-degree
            unvis = [u for u in nbrs if u not in in_path]
            if unvis:
                # Weight by degree: lower degree = more urgent to use
                min_d = min(deg[u] for u in unvis)
                best = [u for u in unvis if deg[u] == min_d]
                nxt = rng.choice(best)
            else:
                # All neighbors are in path; try to find a way to continue
                # This means we'd create a cycle; stop to maintain simplicity
                break
            e = tuple(sorted((v, nxt)))
            remaining.discard(e)
            deg[v] -= 1
            deg[nxt] -= 1
            if deg[v] <= 0:
                del deg[v]
            if deg[nxt] <= 0:
                del deg[nxt]
            path.append(nxt)
            in_path.add(nxt)
        if len(path) >= 2:
            paths.append(path)
    # leftover edges as 2-vertex paths
    for e in remaining:
        paths.append([e[0], e[1]])
    return paths


def _try_merge(paths, i, j):
    """Try to merge paths[i] and paths[j]. Returns (new_paths, success)."""
    P1 = list(paths[i])
    P2 = list(paths[j])
    
    # Try all four combinations of endpoint connections
    # P1: a -> ... -> b, P2: c -> ... -> d
    # Merge if b==c (P1+P2[1:]), b==d (P1+rev(P2)[1:]), a==c (rev(P1)+P2[1:]), a==d (rev(P1)+rev(P2)[1:])
    
    # Case 1: P1[-1] == P2[0]
    if P1[-1] == P2[0]:
        merged = P1 + P2[1:]
        if _is_simple(merged):
            return merged, True
    
    # Case 2: P1[-1] == P2[-1]
    if P1[-1] == P2[-1]:
        P2r = P2[::-1]
        merged = P1 + P2r[1:]
        if _is_simple(merged):
            return merged, True
    
    # Case 3: P1[0] == P2[0]
    if P1[0] == P2[0]:
        P1r = P1[::-1]
        merged = P1r + P2[1:]
        if _is_simple(merged):
            return merged, True
    
    # Case 4: P1[0] == P2[-1]
    if P1[0] == P2[-1]:
        P1r = P1[::-1]
        merged = P1r + P2[1:]
        if _is_simple(merged):
            return merged, True
    
    return None, False


def _try_2opt_swap(paths, i, j, eset):
    """Try a 2-opt style swap between two paths.
    
    For paths P1 = [a, ..., x, y, ..., b] and P2 = [c, ..., z, w, ..., d],
    if edge (y,z) exists in the graph and isn't used in either path, and edge (x,w) exists
    and isn't used, we can swap to get P1' = [a,...,x,w,...,z,y,...,b] and P2' = [c,...,d].
    
    Actually, a cleaner formulation: if we have two paths and there are edges connecting
    their interiors that would allow restructuring into two shorter/simpler paths...
    
    A simpler useful operation: if P1 has internal edge (u,v) and P2 has internal edge (w,z),
    and edges (u,w) and (v,z) exist in the graph, we can potentially restructure.
    
    Let me implement a different useful operation: edge exchange between two paths.
    If P1 contains edge (a,b) and P2 contains edge (c,d), and the graph has edges (a,c) and (b,d),
    we can replace (a,b) and (c,d) with (a,c) and (b,d), potentially creating longer paths.
    """
    P1 = list(paths[i])
    P2 = list(paths[j])
    
    if len(P1) < 3 or len(P2) < 3:
        return False
    
    edges1 = _path_edges_set(P1)
    edges2 = _path_edges_set(P2)
    
    # For each pair of edges (e1 in P1, e2 in P2), check if swapping creates valid paths
    for idx1 in range(len(P1) - 1):
        a, b = P1[idx1], P1[idx1 + 1]
        for idx2 in range(len(P2) - 1):
            c, d = P2[idx2], P2[idx2 + 1]
            
            # Check if edges (a,c) and (b,d) exist in graph
            e_ac = tuple(sorted((a, c)))
            e_bd = tuple(sorted((b, d)))
            
            if e_ac not in eset or e_bd not in eset:
                continue
            # Make sure these edges aren't already in either path
            if e_ac in edges1 or e_ac in edges2:
                continue
            if e_bd in edges1 or e_bd in edges2:
                continue
            
            # Try the swap: replace (a,b) in P1 with (a,c) and (b,d) in P2
            # P1 becomes: [.., a, c, ..] and we need to figure out the rest
            # This is complex; let's try a simpler version
            
            # Alternative: just check if we can merge by cutting and reconnecting
            # P1 = [..., a, b, ...], P2 = [..., c, d, ...]
            # If we remove (a,b) from P1 and (c,d) from P2, we get 4 segments.
            # Reconnect as (a,c) and (b,d): 
            #   Segment1 + [a,c] + reversed(Segment2_part) ... this gets complicated.
            # 
            # Let me try a different approach: direct concatenation check
            pass
    
    return False


def _improve_local(paths, edges, adj, time_limit=5.0, rng=None):
    """Local search: repeatedly try merges and simple improvements until no improvement."""
    if rng is None:
        rng = random.Random(777)
    start_t = time.time()
    eset = _edge_set(edges)
    
    while True:
        if time.time() - start_t > time_limit:
            break
        
        improved = False
        n = len(paths)
        
        # Try all pairs for merging
        for i in range(n):
            for j in range(i + 1, n):
                merged, success = _try_merge(paths, i, j)
                if success:
                    # Replace paths[i] and paths[j] with merged
                    if i < j:
                        paths[i] = merged
                        paths.pop(j)
                    else:
                        paths[j] = merged
                        paths.pop(i)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return paths


def _solve_one(edges, time_limit=40.0):
    adj = _build_adj(edges)
    best = None
    best_count = float('inf')
    t0 = time.time()
    
    # Use multiple random seeds for diversity
    for trial in range(500):
        if time.time() - t0 > time_limit - 0.5:
            break
        seed = trial * 7919 + 13
        rng = random.Random(seed)
        paths = _paths_from_edges_greedy(edges, adj, rng)
        paths = _improve_local(paths, edges, adj, time_limit=0.05, rng=rng)
        cnt = len(paths)
        if cnt < best_count:
            best_count = cnt
            best = [list(p) for p in paths]
            if best_count <= 1:
                break
    
    if best is None:
        best = _paths_from_edges_greedy(edges, adj, random.Random(42))
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 148.0
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    n_graphs = len(specs)
    for i, (n, p, seed) in enumerate(specs):
        remaining = deadline - time.time()
        if remaining < 3:
            edges = _instance(n, p, seed)
            adj = _build_adj(edges)
            results.append(_paths_from_edges_greedy(edges, adj, random.Random(1)))
            continue
        alloc = max(2.0, (remaining - 1.0) / (n_graphs - i))
        edges = _instance(n, p, seed)
        paths = _solve_one(edges, time_limit=alloc)
        results.append(paths)
    return results


if __name__ == "__main__":
    res = construct()
    for i, paths in enumerate(res):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in res)}")
