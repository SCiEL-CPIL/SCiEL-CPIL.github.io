import time
from collections import defaultdict
from random import Random

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose_paths(edges):
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Count degrees
        deg = {v: len(adj[v]) for v in adj}
        
        # Find odd-degree vertices
        odd_vertices = [v for v in deg if deg[v] % 2 == 1]
        
        # At each vertex, pair up edges. For even degree, all paired. For odd, leave one unpaired.
        # We'll create a mapping: for each directed edge (u->v), which edge (v->w) does it pair with at v?
        # Actually, let's think of it as: at vertex v, we have a list of neighbors. We pair them up.
        
        # Pairing at each vertex
        # For each vertex, we'll create a list of (neighbor, is_unpaired)
        # Then we can trace paths.
        
        # Let's create for each vertex, a list of its incident edges (as frozensets or sorted tuples)
        # and pair them up.
        
        # Better approach: create a dictionary that maps each directed edge to the next directed edge in the path.
        # For each vertex v, we have a list of neighbors. We pair consecutive neighbors: (n0,n1), (n2,n3), etc.
        # If odd, the last one is unpaired.
        
        # Let's build: next_edge[dir_edge] = next_dir_edge
        # where dir_edge is (u, v) meaning we arrived at v from u, and next is (v, w)
        
        next_map = {}
        
        for v in adj:
            neighbors = adj[v]
            d = len(neighbors)
            # Shuffle neighbors for variety (but deterministic per graph)
            # Actually, let's just use the sorted order for determinism
            # neighbors is already sorted if adj lists are built from sorted edges
            # But adj[v] order depends on insertion order. Let's sort.
            neighbors = sorted(neighbors)
            
            if d % 2 == 0:
                # Pair all: (0,1), (2,3), ...
                for i in range(0, d, 2):
                    a, b = neighbors[i], neighbors[i+1]
                    # (a->v) pairs with (v->b), and (b->v) pairs with (v->a)
                    next_map[(a, v)] = (v, b)
                    next_map[(b, v)] = (v, a)
            else:
                # Pair (0,1), (2,3), ..., leave last unpaired
                for i in range(0, d - 1, 2):
                    a, b = neighbors[i], neighbors[i+1]
                    next_map[(a, v)] = (v, b)
                    next_map[(b, v)] = (v, a)
                # The last neighbor is unpaired; no next_map entry for (last, v)
        
        # Now trace paths
        # Start from all "unpaired" directed edges, i.e., directed edges (u,v) where (v,u) is not in next_map as a key
        # Actually, a directed edge (u,v) is a start if there's no (x,u) such that next_map[(x,u)] = (u,v)
        # Simpler: a directed edge (u,v) is a path start if it's not the "next" of any other directed edge.
        
        # Let's find all directed edges that are not in the values of next_map
        all_dir_edges = set()
        for u, v in edges:
            all_dir_edges.add((u, v))
            all_dir_edges.add((v, u))
        
        used_as_next = set(next_map.values())
        starts = [de for de in all_dir_edges if de not in used_as_next]
        
        paths = []
        visited_edges = set()
        
        for start in starts:
            if start in visited_edges:
                continue
            path = [start[0]]
            cur = start
            while True:
                visited_edges.add(cur)
                u, v = cur
                path.append(v)
                if cur in next_map:
                    cur = next_map[cur]
                else:
                    break
            
            # Simplify path: remove consecutive duplicates (shouldn't happen) 
            # Actually the path should be simple since we're following edges
            paths.append(path)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 5:
            # Fallback: each edge as its own path
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        paths = decompose_paths(edges)
        results.append(paths)
    
    return results
