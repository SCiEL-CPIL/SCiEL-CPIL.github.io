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

def _is_simple(path):
    return len(set(path)) == len(path)

def _find_eulerian_circuit(edges, adj):
    """
    Find an Eulerian circuit in a connected graph with all even degrees.
    Returns a list of vertices representing the circuit.
    Uses Hierholzer's algorithm.
    """
    # Make a mutable copy of adjacency lists
    # We need to track used edges.
    # Since edges are undirected, we represent them as (u, v) with u < v.
    # To handle multigraphs (virtual edges might duplicate real edges), 
    # we should track edge instances.
    
    # For simplicity and given small graph sizes, we can just use a set of edges.
    # But Hierholzer's needs to pop edges.
    
    # Let's use a list of (neighbor, edge_id) for each vertex.
    # Or just use the set of edges and remove them.
    
    # Note: The input `edges` is a list of tuples.
    # `adj` is a dict of sets.
    
    # We need to be careful with the "used" edges.
    # Let's create a list of edges for the augmented graph.
    
    pass # Placeholder, will implement in _solve_one

def _solve_one(edges, time_limit=40.0):
    adj = _build_adj(edges)
    eset = _edge_set(edges)
    
    # Calculate degrees
    deg = defaultdict(int)
    for u, v in eset:
        deg[u] += 1
        deg[v] += 1
        
    odd_vertices = [v for v, d in deg.items() if d % 2 == 1]
    
    # If no odd vertices, try to find a single cycle (Eulerian circuit)
    # But we need a simple path. A cycle is a path if we cut it?
    # No, a cycle is not a simple path unless it's a triangle?
    # A simple path cannot repeat vertices.
    # If the graph is a cycle, the path is the cycle minus one edge?
    # No, we must cover ALL edges.
    # If we have a cycle, we can't cover it with a simple path unless the cycle is small?
    # Wait. If we have a cycle, we can't decompose it into 1 simple path that covers all edges.
    # A simple path has 2 endpoints. A cycle has 0.
    # So a cycle must be decomposed into at least 2 paths?
    # No, if we have a cycle, we can pick an edge, remove it, and we have a path covering the rest.
    # But we must cover ALL edges.
    # So we need 1 path?
    # If we have a cycle C, can we cover it with 1 simple path?
    # No. A simple path visits each vertex at most once.
    # A cycle visits each vertex exactly once (except start=end).
    # If we break the cycle at one edge, we get a path.
    # But that edge is not covered.
    # So we need 2 paths?
    # Path 1: covers N-1 edges.
    # Path 2: covers 1 edge.
    # Total 2 paths.
    # But Gallai's bound for a cycle (all deg 2, even) is 0?
    # No, Gallai's conjecture is for general graphs.
    # For a cycle, $|V_{odd}| = 0$. Bound $\lceil 0/2 \rceil = 0$.
    # But we need at least 1 path to cover edges.
    # Actually, if $|V_{odd}|=0$, the graph has an Eulerian circuit.
    # Can we decompose it into 1 simple path?
    # Only if the graph is a path? No, it's a cycle.
    # So the lower bound $\lceil |V_{odd}|/2 \rceil$ is a lower bound on the number of paths 
    # IF we allow paths to be closed? No, paths are open.
    # The theorem states: Min number of paths to cover edges is $\max(1, \lceil |V_{odd}|/2 \rceil)$.
    # For a cycle, $|V_{odd}|=0$, so min paths = 1.
    # But a single simple path cannot cover a cycle.
    # So the theorem must imply that if $|V_{odd}|=0$, we need 1 path ONLY IF the graph is not a cycle?
    # Or maybe the "path" can be a cycle?
    # "Paths must be simple". A cycle is not a simple path (start != end).
    # So for a cycle, we need 2 paths?
    # Let's check $C_4$.
    # Edges: (0,1), (1,2), (2,3), (3,0).
    # Path 1: 0-1-2.
    # Path 2: 3-0? No, (3,0) and (2,3) left?
    # Path 2: 2-3-0? No, (2,3) used? No.
    # Path 1: 0-1. Path 2: 1-2. Path 3: 2-3. Path 4: 3-0. (4 paths)
    # Can we do 2?
    # Path 1: 0-1-2. Path 2: 3-0? No.
    # Path 1: 0-1-2-3. (Covers 3 edges).
    # Path 2: 0-3. (Covers 1 edge).
    # Total 2 paths.
    # So for $C_4$, min paths is 2.
    # $|V_{odd}| = 0$. $\lceil 0/2 \rceil = 0$.
    # So the formula is $\max(1, \lceil |V_{odd}|/2 \rceil)$ is WRONG for cycles.
    # It should be related to the number of components?
    
    # Actually, for connected graphs with even degrees, the min number of simple paths is 
    # the minimum number of paths to cover edges.
    # This is known to be 1 if the graph is a path, 2 if it's a cycle?
    # No, for $K_4$ (deg 3), we found 2 paths.
    # For $C_4$ (deg 2), we found 2 paths.
    
    # Let's rely on the Greedy + Local Search. It's robust.
    
    best = None
    best_count = float('inf')
    t0 = time.time()
    
    for trial in range(1000):
        if time.time() - t0 > time_limit - 0.5:
            break
        seed = trial * 7919 + 13
        rng = random.Random(seed)
        
        # Greedy construction
        remaining = set(eset)
        deg_local = dict(deg)
        paths = []
        
        while remaining:
            # Pick start
            candidates = [v for v, d in deg_local.items() if d > 0]
            if not candidates: break
            
            odd = [v for v in candidates if deg_local[v] % 2 == 1]
            if odd:
                start = rng.choice(odd)
            else:
                start = rng.choice(candidates)
                
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
                unvis = [u for u in nbrs if u not in in_path]
                if unvis:
                    # Prefer low degree
                    min_d = min(deg_local[u] for u in unvis)
                    best = [u for u in unvis if deg_local[u] == min_d]
                    nxt = rng.choice(best)
                else:
                    break
                    
                e = tuple(sorted((v, nxt)))
                remaining.discard(e)
                deg_local[v] -= 1
                deg_local[nxt] -= 1
                if deg_local[v] <= 0: del deg_local[v]
                if deg_local[nxt] <= 0: del deg_local[nxt]
                
                path.append(nxt)
                in_path.add(nxt)
                
            if len(path) >= 2:
                paths.append(path)
                
        for e in remaining:
            paths.append([e[0], e[1]])
            
        # Local Search: Merge
        improved = True
        while improved:
            improved = False
            n = len(paths)
            for i in range(n):
                for j in range(i+1, n):
                    P1 = paths[i]
                    P2 = paths[j]
                    # Try merges
                    if P1[-1] == P2[0]:
                        m = P1 + P2[1:]
                        if _is_simple(m):
                            paths[i] = m
                            paths.pop(j)
                            improved = True
                            break
                    if P1[-1] == P2[-1]:
                        m = P1 + P2[::-1][1:]
                        if _is_simple(m):
                            paths[i] = m
                            paths.pop(j)
                            improved = True
                            break
                    if P1[0] == P2[0]:
                        m = P1[::-1] + P2[1:]
                        if _is_simple(m):
                            paths[i] = m
                            paths.pop(j)
                            improved = True
                            break
                    if P1[0] == P2[-1]:
                        m = P1[::-1] + P2
                        if _is_simple(m):
                            paths[i] = m
                            paths.pop(j)
                            improved = True
                            break
                if improved:
                    break
                    
        cnt = len(paths)
        if cnt < best_count:
            best_count = cnt
            best = [list(p) for p in paths]
            if best_count <= 2: # Heuristic stop
                break
                
    if best is None:
        best = []
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
            # Fallback
            results.append([[u, v] for u, v in edges])
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
