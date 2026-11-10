import time
import random
from collections import defaultdict
import sys

# Increase recursion limit if needed, though we use iterative methods
sys.setrecursionlimit(10000)

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

def _greedy_cover(edges, adj, rng):
    """
    Generate an initial path cover using a greedy strategy.
    We prioritize extending paths from vertices with high remaining degree.
    """
    remaining = set(_edge_set(edges))
    deg = defaultdict(int)
    for u, v in remaining:
        deg[u] += 1
        deg[v] += 1
    
    paths = []
    
    while remaining:
        # Find a vertex with max remaining degree to start a path
        # If no vertices have degree > 0, break (should not happen)
        candidates = [v for v, d in deg.items() if d > 0]
        if not candidates:
            break
        
        # Pick the vertex with the highest degree
        # Tie-break randomly
        max_deg = max(deg[v] for v in candidates)
        best_candidates = [v for v in candidates if deg[v] == max_deg]
        start = rng.choice(best_candidates)
        
        path = [start]
        in_path = {start}
        
        while True:
            v = path[-1]
            # Find neighbors with remaining edges
            nbrs = []
            for u in adj[v]:
                e = tuple(sorted((v, u)))
                if e in remaining:
                    nbrs.append(u)
            
            if not nbrs:
                break
            
            # Choose the next neighbor
            # Prefer unvisited neighbors to keep path simple
            unvis = [u for u in nbrs if u not in in_path]
            
            if unvis:
                # Among unvisited, pick the one with the highest remaining degree
                # This helps create long paths
                max_deg_next = max(deg[u] for u in unvis)
                best_next = [u for u in unvis if deg[u] == max_deg_next]
                nxt = rng.choice(best_next)
            else:
                # All neighbors are in path.
                # We must stop to maintain simplicity (no cycles in path)
                # Or we could allow cycles and break them later, but simple paths are required.
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
            
    # Any leftover edges (should be none if logic is correct, but just in case)
    for e in remaining:
        paths.append([e[0], e[1]])
        
    return paths

def _try_merge_paths(paths, i, j, eset):
    """
    Try to merge paths[i] and paths[j].
    Returns the merged path if successful, else None.
    Conditions for merge:
    1. Endpoints are the same vertex (e.g., P1 ends at v, P2 starts at v).
    2. There is an edge between an endpoint of P1 and an endpoint of P2.
    """
    P1 = list(paths[i])
    P2 = list(paths[j])
    
    if len(P1) < 2 or len(P2) < 2:
        return None
        
    # Case 1: Direct connection of endpoints
    # P1: a ... b, P2: c ... d
    # Check (b, c), (b, d), (a, c), (a, d)
    
    # Try (P1[-1], P2[0])
    if tuple(sorted((P1[-1], P2[0]))) in eset:
        merged = P1 + P2[1:]
        if _is_simple(merged):
            return merged
            
    # Try (P1[-1], P2[-1]) -> P1 + reversed(P2)[1:]
    if tuple(sorted((P1[-1], P2[-1]))) in eset:
        P2r = P2[::-1]
        merged = P1 + P2r[1:]
        if _is_simple(merged):
            return merged
            
    # Try (P1[0], P2[0]) -> reversed(P1) + P2[1:]
    if tuple(sorted((P1[0], P2[0]))) in eset:
        P1r = P1[::-1]
        merged = P1r + P2[1:]
        if _is_simple(merged):
            return merged
            
    # Try (P1[0], P2[-1]) -> reversed(P1) + P2 (wait, P1[0] connects to P2[-1])
    # P1: a...b, P2: c...d. Connect a to d.
    # Path: b...a-d...c ? No, P1 reversed is b...a. P2 is c...d.
    # Connect a to d. So b...a-d...c.
    if tuple(sorted((P1[0], P2[-1]))) in eset:
        P1r = P1[::-1]
        merged = P1r + P2
        # Wait, P1r ends with P1[0]. P2 starts with P2[0].
        # We want to connect P1[0] and P2[-1].
        # So we reverse P1: [b, ..., a]. Then append P2? No, P2 is [c, ..., d].
        # We need to connect a and d.
        # So we take P1 reversed: [b, ..., a]. Then we need to connect a to d.
        # So we append P2 reversed? [d, ..., c].
        # Result: [b, ..., a, d, ..., c].
        # Let's check edges: (a,d) exists.
        # Is it simple?
        merged2 = P1r + P2[::-1]
        if _is_simple(merged2):
            return merged2
            
    return None

def _try_2opt_swap(paths, i, j, eset, adj):
    """
    Try a 2-opt swap between two paths.
    P1 = [v1, ..., vi, vi+1, ..., vk]
    P2 = [u1, ..., uj, uj+1, ..., um]
    
    If we remove edges (vi, vi+1) and (uj, uj+1), and add edges (vi, uj) and (vi+1, uj+1),
    we might form two new valid paths.
    This operation preserves the number of paths but might allow future merges.
    It can also reduce the number of paths if the new structure allows a merge.
    Actually, 2-opt usually keeps the number of paths constant but changes the structure.
    However, if the swap results in a path that can be merged with another, it's useful.
    
    For simplicity, we will implement a "reconnect" move:
    If we have P1 and P2, and edges (P1[i], P2[j]) and (P1[i+1], P2[j+1]) exist,
    we can swap the tails.
    """
    P1 = list(paths[i])
    P2 = list(paths[j])
    
    if len(P1) < 3 or len(P2) < 3:
        return None, None
        
    # Try all pairs of edges
    for i1 in range(len(P1) - 1):
        u1, v1 = P1[i1], P1[i1+1]
        for i2 in range(len(P2) - 1):
            u2, v2 = P2[i2], P2[i2+1]
            
            # Check if (u1, u2) and (v1, v2) are edges
            if tuple(sorted((u1, u2))) in eset and tuple(sorted((v1, v2))) in eset:
                # Construct new paths
                # P1 becomes: P1[:i1+1] + P2[i2+1:]  (u1 connected to u2? No, u1 connected to u2 is the new edge)
                # Wait.
                # Original: ... u1 - v1 ... and ... u2 - v2 ...
                # New edges: (u1, u2) and (v1, v2).
                # New Path 1: ... u1 - u2 ... (tail of P2)
                # New Path 2: ... v1 - v2 ... (tail of P1)
                
                # Let's trace:
                # P1 = [A, ..., u1, v1, ..., B]
                # P2 = [C, ..., u2, v2, ..., D]
                
                # Cut P1 at (u1, v1). Head1 = [A, ..., u1], Tail1 = [v1, ..., B]
                # Cut P2 at (u2, v2). Head2 = [C, ..., u2], Tail2 = [v2, ..., D]
                
                # New Path 1: Head1 + Tail2?
                # Connect u1 to v2? No, we said edges (u1, u2) and (v1, v2).
                # So connect u1 to u2? Then we have Head1 + Head2_reversed?
                # This is getting confusing. Let's stick to the standard 2-opt for TSP-like paths.
                
                # Standard 2-opt:
                # P1 = [a, ..., u1, v1, ..., b]
                # P2 = [c, ..., u2, v2, ..., d]
                # If (u1, u2) and (v1, v2) exist:
                # New P1 = [a, ..., u1, u2, ..., c] ?? No.
                
                # Let's try this combination:
                # New P1 = [a, ..., u1, u2, ..., c] is not a path unless u2 connects to c?
                
                # Actually, a simpler move is:
                # If (P1[i], P2[j]) and (P1[i+1], P2[j+1]) exist:
                # New P1 = P1[:i+1] + P2[j+1:]
                # New P2 = P2[:j+1] + P1[i+1:]
                # Check if these are valid paths (edges exist).
                
                # Edge (P1[i], P2[j+1])? No, we need (P1[i], P2[j]) to connect?
                # If we connect P1[i] to P2[j], the edge must exist.
                # If we connect P1[i+1] to P2[j+1], the edge must exist.
                
                # Let's assume the move is:
                # Cut P1 at i, P2 at j.
                # Connect P1[i] to P2[j] (if edge exists)
                # Connect P1[i+1] to P2[j+1] (if edge exists)
                
                # Then:
                # New P1 = P1[:i+1] + [P2[j]] + P2[j+1:] ? No, P2[j] is already in P2.
                # We are swapping the tails.
                # New P1 = P1[:i+1] + P2[j:]  (Connect P1[i] to P2[j])
                # New P2 = P2[:j+1] + P1[i+1:] (Connect P2[j] to P1[i+1])
                
                # Check edges:
                # 1. (P1[i], P2[j]) must be in eset.
                # 2. (P2[j], P1[i+1]) must be in eset.
                
                if tuple(sorted((P1[i1], P2[i2]))) in eset and tuple(sorted((P2[i2], P1[i1+1]))) in eset:
                    new_P1 = P1[:i1+1] + P2[i2:]
                    new_P2 = P2[:i2+1] + P1[i1+1:]
                    
                    if _is_simple(new_P1) and _is_simple(new_P2):
                        # Also check that the edges in the new paths are valid
                        # (They should be, since we only used existing edges and kept the rest)
                        return new_P1, new_P2
                        
    return None, None

def _solve_one_sa(edges, adj, time_limit=10.0):
    eset = _edge_set(edges)
    rng = random.Random(12345)
    
    # Initial solution
    paths = _greedy_cover(edges, adj, rng)
    best_paths = [list(p) for p in paths]
    best_count = len(paths)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.95
    t0 = time.time()
    
    while time.time() - t0 < time_limit:
        # Try a random move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to merge two random paths
            if len(paths) < 2:
                continue
            i, j = rng.sample(range(len(paths)), 2)
            merged = _try_merge_paths(paths, i, j, eset)
            if merged:
                new_paths = [p for k, p in enumerate(paths) if k not in (i, j)]
                new_paths.append(merged)
                # Accept if better or with probability
                if len(new_paths) < len(paths):
                    paths = new_paths
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = [list(p) for p in paths]
                elif rng.random() < T:
                    paths = new_paths
        elif move_type < 0.8:
            # Try 2-opt swap
            if len(paths) < 2:
                continue
            i, j = rng.sample(range(len(paths)), 2)
            new_P1, new_P2 = _try_2opt_swap(paths, i, j, eset, adj)
            if new_P1 and new_P2:
                new_paths = [p for k, p in enumerate(paths) if k not in (i, j)]
                new_paths.append(new_P1)
                new_paths.append(new_P2)
                # Number of paths is the same, but structure changed.
                # Accept with probability T (to explore)
                if rng.random() < T:
                    paths = new_paths
        else:
            # Try to split a path (to escape local minima)
            if len(paths) > 1:
                i = rng.randrange(len(paths))
                P = paths[i]
                if len(P) > 2:
                    # Split at a random internal edge
                    split_idx = rng.randrange(1, len(P)-1)
                    P1 = P[:split_idx+1]
                    P2 = P[split_idx:]
                    new_paths = [p for k, p in enumerate(paths) if k != i]
                    new_paths.append(P1)
                    new_paths.append(P2)
                    if rng.random() < T:
                        paths = new_paths
                        
        # Cool down
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return best_paths

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    n_graphs = len(specs)
    
    for i, (n, p, seed) in enumerate(specs):
        remaining = deadline - time.time()
        if remaining < 5:
            # Fallback to quick greedy
            edges = _instance(n, p, seed)
            adj = _build_adj(edges)
            results.append(_greedy_cover(edges, adj, random.Random(1)))
            continue
            
        alloc = max(5.0, (remaining - 2.0) / (n_graphs - i))
        edges = _instance(n, p, seed)
        adj = _build_adj(edges)
        
        # Run SA
        paths = _solve_one_sa(edges, adj, time_limit=alloc)
        results.append(paths)
        
    return results

if __name__ == "__main__":
    res = construct()
    for i, paths in enumerate(res):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in res)}")
