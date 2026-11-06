import time
import random
import numpy as np

def count_c4(n, adj):
    """Count 4-cycles (subgraphs) in an undirected graph."""
    c4 = 0
    # For each pair of vertices, count common neighbors
    # A 4-cycle a-b-c-d-a corresponds to two pairs (a,c) and (b,d) with 2 common neighbors
    # Actually: number of C4 = sum over all pairs {u,v} of C(common_neighbors(u,v), 2)
    # But this counts each C4 exactly 2 times? Let me think...
    # A C4 has vertices v0,v1,v2,v3. The pairs of opposite vertices are (v0,v2) and (v1,v3).
    # For pair (v0,v2): common neighbors are v1 and v3 -> C(2,2)=1
    # For pair (v1,v3): common neighbors are v0 and v2 -> C(2,2)=1
    # So each C4 is counted exactly 2 times (once for each pair of opposite vertices)
    # Wait, that's not right. Let me reconsider.
    # Actually each C4 contributes to exactly 2 pairs of opposite vertices, each with exactly 2 common neighbors.
    # So total count = sum_{u<v} C(cn(u,v), 2) counts each C4 exactly 2 times.
    # Hmm, let me verify with a simple C4: vertices 0-1-2-3-0
    # Pairs: (0,2) have common neighbors {1,3} -> C(2,2)=1
    # (1,3) have common neighbors {0,2} -> C(2,2)=1
    # (0,1): cn = {} -> 0
    # (0,3): cn = {} -> 0
    # (1,2): cn = {} -> 0
    # (2,3): cn = {} -> 0
    # Total = 2. But there's only 1 C4. So we divide by 2.
    
    # Use matrix approach for speed
    A = adj
    # common neighbors = A @ A, diagonal entries are degrees
    common = A @ A
    # For each pair (i,j), common neighbors count = common[i,j]
    # We want sum over i<j of C(common[i,j], 2)
    # = sum over i<j of common[i,j]*(common[i,j]-1)//2
    np.fill_diagonal(common, 0)
    # Take upper triangle
    iu = np.triu_indices(n, 1)
    vals = common[iu]
    c4 = np.sum(vals * (vals - 1) // 2)
    return c4 // 2

def construct():
    deadline = time.time() + 145  # Safety margin
    n = 31
    target_edges = 91
    
    best_c4 = None
    best_adj = None
    best_edges = None
    
    def try_graph(adj):
        nonlocal best_c4, best_adj, best_edges
        c4 = count_c4(n, adj)
        if best_c4 is None or c4 < best_c4:
            best_c4 = c4
            best_adj = adj.copy()
            # Extract edges
            iu = np.triu_indices(n, 1)
            edges = [(int(iu[0][k]), int(iu[1][k])) for k in range(len(iu[0])) if adj[iu[0][k], iu[1][k]]]
            best_edges = edges
        return c4
    
    def random_adj():
        adj = np.zeros((n, n), dtype=np.int32)
        # Generate random graph with ~91 edges
        for _ in range(target_edges):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and adj[u,v] == 0:
                adj[u,v] = 1
                adj[v,u] = 1
        return adj
    
    def greedy_reduce(adj, max_iter=500):
        """Greedy 4-cycle reduction: remove edges that are in many C4s."""
        for it in range(max_iter):
            if time.time() > deadline:
                break
            c4 = count_c4(n, adj)
            if c4 == 0:
                break
            # Find edge in most 4-cycles
            # An edge (u,v) is in a C4 if there exist a,b with a-u-b-v and a-v-b-u
            # i.e., common neighbors of u and v... wait
            # Edge (u,v) is in C4 if there are vertices a,b such that a-u, a-b, b-v, and a-v or... 
            # Actually: C4 containing edge (u,v): u-v-b-a-u where a-b, a-u, b-v
            # So we need a common neighbor b of v and a, and a common neighbor a of u and b... 
            # Simpler: edge (u,v) is in #C4s = number of paths of length 3 from u to v through other vertices
            # = sum over pairs (a,b) where a~u, b~v, a~b, a!=b, a!=v, b!=u
            # Using adjacency: for edge (u,v), count = sum_{a in N(u)\{v}} sum_{b in N(v)\{u}} [a~b and a!=b]
            # = sum_{a in N(u)\{v}} (|N(v) ∩ N(a)| - [v in N(a)] )... getting complex
            # Let's just compute: for each edge (u,v), the number of C4s containing it
            # = number of common neighbors... no.
            # A C4 containing edge (u,v) has form u-v-x-y-u where x~v, x~y, y~u, y!=v, x!=u, x!=y
            # So: x is neighbor of v (x!=u), y is neighbor of u (y!=v), and x~y
            # count = sum_{x in N(v)\{u}} |N(x) ∩ (N(u)\{v})|
            
            # This is expensive. Let's try a simpler heuristic: remove random edges and keep if C4 decreases
            # Or: just try removing each edge and see
            best_reduction = 0
            best_edge = None
            # Sample some edges to remove
            edges_in_graph = []
            iu = np.triu_indices(n, 1)
            for k in range(len(iu[0])):
                if adj[iu[0][k], iu[1][k]]:
                    edges_in_graph.append((iu[0][k], iu[1][k]))
            
            # Try removing each edge, find the one that reduces C4 most
            # Too expensive to try all. Sample.
            sample_size = min(len(edges_in_graph), 50)
            if len(edges_in_graph) <= sample_size:
                sample = edges_in_graph
            else:
                sample = random.sample(edges_in_graph, sample_size)
            
            for (u,v) in sample:
                adj[u,v] = 0
                adj[v,u] = 0
                c4_new = count_c4(n, adj)
                reduction = c4 - c4_new
                adj[u,v] = 1
                adj[v,u] = 1
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_edge = (u,v)
            
            if best_edge is None or best_reduction <= 0:
                break
            
            u, v = best_edge
            adj[u,v] = 0
            adj[v,u] = 0
            c4 = count_c4(n, adj)
        
        return c4
    
    def add_edge_optimal(adj):
        """Add one edge to minimize increase in C4 count."""
        if time.time() > deadline:
            return adj
        # Find edge whose addition creates fewest new C4s
        # Adding edge (u,v) creates C4s: for each pair (a,b) where a~u, a~v, b~u, b~v... 
        # New C4s from edge (u,v): paths of length 3 from u to v: u-a-b-v where a~u, a~b, b~v
        # New C4s = number of such paths = sum_{a in N(u)} |N(a) ∩ N(v)| (excluding v from N(u) consideration... 
        # Actually: u-a-b-v: a is neighbor of u, b is neighbor of v, a~b
        # count = sum_{a in N(u)} |N(a) ∩ N(v)|
        
        iu = np.triu_indices(n, 1)
        best_increase = float('inf')
        best_edge = None
        
        for k in range(len(iu[0])):
            u, v = iu[0][k], iu[1][k]
            if adj[u,v]:
                continue
            # Count new C4s if we add (u,v)
            # = number of paths u-a-b-v
            # a in N(u), b in N(v), a~b
            # = sum over a in N(u) of |N(a) ∩ N(v)|
            nu = np.where(adj[u] > 0)[0]
            nv = np.where(adj[v] > 0)[0]
            count = 0
            for a in nu:
                if a == v:
                    continue
                # |N(a) ∩ N(v)|
                count += np.sum(adj[a] * adj[v])
            if count < best_increase:
                best_increase = count
                best_edge = (u, v)
        
        if best_edge is not None:
            u, v = best_edge
            adj[u,v] = 1
            adj[v,u] = 1
        return adj
    
    # Strategy: Start with random graphs, greedily reduce C4s, then add 91st edge optimally
    random.seed(42)
    
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random graph
        adj = np.zeros((n, n), dtype=np.int32)
        # Use a structured random initialization: prefer regular-ish graphs
        # Start with random permutation-based construction
        perm = list(range(n))
        random.shuffle(perm)
        
        # Build graph by connecting vertices with some structure
        # Try: connect i to i+1, i+2, i+3 (mod n) for a base, then add random
        edges_added = 0
        # Base: 3-regular-ish ring
        for i in range(n):
            for offset in [1, 2]:
                j = (i + offset) % n
                if i < j and adj[i,j] == 0:
                    adj[i,j] = 1
                    adj[j,i] = 1
                    edges_added += 1
                    if edges_added >= 85:
                        break
            if edges_added >= 85:
                break
        
        # Add random edges to reach ~88-90 edges
        while edges_added < 88:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and adj[u,v] == 0:
                adj[u,v] = 1
                adj[v,u] = 1
                edges_added += 1
        
        # Greedy reduce
        c4 = greedy_reduce(adj, max_iter=200)
        
        # Now we have a graph with <= 88 edges and few C4s. Add edges to reach 91.
        # Add one edge at a time, choosing the one that adds fewest C4s
        current_edges = np.sum(adj) // 2
        while current_edges < target_edges:
            if time.time() > deadline:
                break
            adj = add_edge_optimal(adj)
            current_edges = np.sum(adj) // 2
        
        # Evaluate
        final_c4 = count_c4(n, adj)
        if best_c4 is None or final_c4 < best_c4:
            best_c4 = final_c4
            best_adj = adj.copy()
            iu = np.triu_indices(n, 1)
            best_edges = [(int(iu[0][k]), int(iu[1][k])) for k in range(len(iu[0])) if adj[iu[0][k], iu[1][k]]]
    
    # If we haven't found anything (shouldn't happen), fallback
    if best_edges is None or len(best_edges) != target_edges:
        # Fallback: just use the seed
        pairs = [[i, j] for i in range(n) for j in range(i + 1, n)]
        return pairs[:91]
    
    # Verify we have exactly 91 edges
    assert len(best_edges) == target_edges, f"Got {len(best_edges)} edges, expected {target_edges}"
    # Ensure all edges are distinct and valid
    edge_set = set()
    for e in best_edges:
        assert 0 <= e[0] < n and 0 <= e[1] < n
        assert e[0] != e[1]
        key = (min(e[0], e[1]), max(e[0], e[1]))
        assert key not in edge_set
        edge_set.add(key)
    
    return [[e[0], e[1]] for e in best_edges]
