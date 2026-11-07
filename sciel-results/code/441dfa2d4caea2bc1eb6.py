# EVOLVE-BLOCK-START
import itertools

def _get_cycle_lengths(adj):
    n = len(adj)
    lengths = set()
    for start in range(n):
        stack = [(start, [start], frozenset([start]))]
        while stack:
            node, path, visited = stack.pop()
            for nb in adj[node]:
                if nb == start and len(path) >= 3:
                    lengths.add(len(path))
                elif nb > start and nb not in visited:
                    stack.append((nb, path + [nb], visited | {nb}))
    return lengths

def _build_graph(edges):
    n = 12
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def construct():
    # Try multiple constructions
    candidates = []
    
    # Construction 1: Prism over C6 (current)
    edges1 = set()
    for i in range(6):
        edges1.add((i, (i+1) % 6))
        edges1.add((i+6, (i+1) % 6 + 6))
        edges1.add((i, i+6))
    candidates.append(list(edges1))
    
    # Construction 2: Incidence graph of 3-uniform hypergraph
    # Hyperedges: {0,1,2}, {0,3,4}, {1,3,5}, {2,4,5}
    # Vertices 0-5 (hypergraph vertices), 6-11 (hypergraph edges)
    edges2 = set()
    hyperedges = [(0,1,2), (0,3,4), (1,3,5), (2,4,5)]
    for i, he in enumerate(hyperedges):
        for v in he:
            edges2.add((v, 6+i))
    # Check degrees: vertex 0: edges 0,1 -> 2 edges. Not 3-regular.
    # Need each hypergraph vertex in exactly 3 hyperedges.
    # Let me use: {0,1,2}, {0,3,5}, {1,4,5}, {2,3,4}
    # Vertex 0: he0, he1 -> 2. Still not 3.
    # Need 6 hyperedges for 6 vertices each in 3 hyperedges: 6*3/3 = 6 hyperedges.
    # This is the complete 3-uniform hypergraph? No, that has C(6,3)=20.
    # A 3-regular 3-uniform hypergraph on 6 vertices with 6 edges: this is a Steiner system S(2,3,6)? No.
    # Actually, this is equivalent to a 3-regular bipartite graph, which is what we're constructing.
    # Let me just use a known 3-regular bipartite graph.
    
    # Construction 3: K3,3 x K2 (Cartesian product)
    # K3,3 vertices: A={0,1,2}, B={3,4,5}
    # K2: {0,1}
    # Product vertices: (a,0), (a,1) for a in 0..5
    # Map: 0->(0,0), 1->(1,0), 2->(2,0), 3->(3,0), 4->(4,0), 5->(5,0)
    #       6->(0,1), 7->(1,1), 8->(2,1), 9->(3,1), 10->(4,1), 11->(5,1)
    edges3 = set()
    # K3,3 edges in layer 0: A-B edges
    for a in [0,1,2]:
        for b in [3,4,5]:
            edges3.add((a, b))
    # K3,3 edges in layer 1: A-B edges
    for a in [6,7,8]:
        for b in [9,10,11]:
            edges3.add((a, b))
    # Matching edges
    for i in range(6):
        edges3.add((i, i+6))
    candidates.append(list(edges3))
    
    # Construction 4: Random 3-regular bipartite graphs
    import random
    random.seed(42)
    for trial in range(200):
        # Generate random 3-regular bipartite graph on 6+6
        # Use configuration model
        left = list(range(6))
        right = list(range(6, 12))
        # Each left vertex has degree 3, each right vertex has degree 3
        # Create stubs
        left_stubs = [i for i in range(6) for _ in range(3)]
        right_stubs = [i for i in range(6, 12) for _ in range(3)]
        random.shuffle(right_stubs)
        edges4 = set()
        valid = True
        for l, r in zip(left_stubs, right_stubs):
            if l == r:  # Won't happen since different ranges
                valid = False
                break
            if (l, r) in edges4 or (r, l) in edges4:
                valid = False
                break
            edges4.add((l, r))
        if valid and len(edges4) == 18:
            candidates.append(list(edges4))
    
    # Evaluate all candidates
    best_edges = None
    best_count = float('inf')
    
    for edges in candidates:
        # Check: no loops, no duplicates
        if len(edges) != len(set(tuple(sorted(e)) for e in edges)):
            continue
        if any(u == v for u, v in edges):
            continue
        # Check degrees
        deg = [0] * 12
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if min(deg) < 3:
            continue
        # Build adjacency
        adj = [[] for _ in range(12)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check triangle-free
        triangle_free = True
        for i in range(12):
            for j in adj[i]:
                for k in adj[i]:
                    if j < k and k in adj[j]:
                        triangle_free = False
                        break
                if not triangle_free:
                    break
            if not triangle_free:
                break
        if not triangle_free:
            continue
        # Must contain a cycle
        # Check if graph has a cycle: 12 vertices, 18 edges > 11, so must have cycle
        # Count distinct cycle lengths
        lengths = _get_cycle_lengths(adj)
        if not lengths:
            continue  # No cycle, invalid
        count = len(lengths)
        if count < best_count:
            best_count = count
            best_edges = edges
    
    if best_edges is None:
        # Fallback to construction 1
        best_edges = candidates[0]
    
    # Sort and return
    sorted_edges = sorted(tuple(sorted(e)) for e in best_edges)
    return [list(e) for e in sorted_edges]

# EVOLVE-BLOCK-END
