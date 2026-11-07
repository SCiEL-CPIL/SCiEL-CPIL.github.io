"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        """Find minimum edge-disjoint clique partition for a split graph."""
        # Build the graph
        rng = Random(seed)
        # Adjacency as sets for fast lookup
        adj = [set() for _ in range(n)]
        edge_set = set()
        
        # Clique on first k vertices
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
        
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((u, v))
        
        # Total edges
        total_edges = len(edge_set)
        
        # Strategy: We need to cover all edges with edge-disjoint cliques.
        # A clique can contain vertices from the clique part (C) and independent part (I).
        # If a clique contains vertices from I, it can contain at most one vertex from I
        # (since I is an independent set).
        # So any clique is either:
        # 1. Entirely within C (a subset of {0..k-1})
        # 2. Contains exactly one vertex from I, say v, and some subset of C that is
        #    a clique in C (which is always true since C is a clique) AND all those
        #    vertices in C are adjacent to v.
        
        # For a vertex v in I, let N_C(v) = {u in C : u adjacent to v}.
        # Any clique containing v must be {v} ∪ S where S ⊆ N_C(v).
        # The edges covered are (v, u) for u in S.
        
        # Edges within C must be covered by cliques entirely in C.
        
        # This is a complex combinatorial optimization. Let's use a greedy approach
        # with some local search / improvement.
        
        # Greedy: 
        # 1. Start with no cliques.
        # 2. While there are uncovered edges:
        #    - Find the largest possible clique that covers the most uncovered edges.
        #    - Add it to the partition.
        
        # For efficiency, we can think of it as:
        # - Edges within C: need to be partitioned into cliques. Since C is a clique,
        #   any subset is a clique. To minimize number of cliques covering edges of C,
        #   we want large cliques. But they must be edge-disjoint.
        #   Actually, the edges of a complete graph K_k can be partitioned into 
        #   edge-disjoint cliques. The minimum number is related to graph coloring of
        #   the line graph, or more simply, we can use a greedy edge-coloring-like approach.
        #   Actually, for K_k, the minimum number of edge-disjoint cliques covering all
        #   edges is k (if k is even, use k/2 perfect matchings... no wait).
        #   Let's think: A clique of size s covers C(s,2) edges.
        #   We want to cover C(k,2) edges with minimum number of cliques.
        #   This is equivalent to decomposing K_k into cliques.
        #   The minimum number of cliques in an edge decomposition of K_k is k if k is even?
        #   No. For K_4, we can use 2 triangles? No, triangles share edges.
        #   K_4 has 6 edges. A triangle has 3 edges. 2 triangles would need 6 edges but
        #   they'd share vertices. Can we partition K_4 edges into 2 triangles? No, because
        #   each triangle uses 3 vertices, and K_4 has 4 vertices.
        #   Actually, the minimum number of cliques to edge-partition K_k is k.
        #   Wait, for K_3, it's 1 (the triangle itself).
        #   For K_4, we can use 4 edges as 2 matchings? No, matchings are not cliques of size > 2.
        #   A clique of size 2 is just an edge. So we can always use individual edges.
        #   But we want to minimize the number of cliques, so we want large cliques.
        #   
        #   This is getting complex. Let's just use a greedy heuristic that works well in practice.
        
        # Greedy algorithm:
        uncovered = set(edge_set)
        cliques = []
        
        # For each vertex in I, we can form a clique {v} ∪ S where S ⊆ N_C(v).
        # The edges covered are (v, u) for u in S.
        # For edges within C, we need cliques in C.
        
        # Let's first handle edges incident to I vertices, then handle C-C edges.
        
        # Actually, a clique can contain both I vertices? No, I is independent.
        # So each clique has at most one I vertex.
        
        # Let's greedily pick cliques:
        while uncovered:
            if time.time() - start_time > DEADLINE:
                break
            
            # Find the best clique to add
            best_clique = None
            best_gain = -1
            best_edges = None
            
            # Option 1: Clique with one I vertex
            for v in range(k, n):
                # N_C(v) ∩ endpoints of uncovered edges incident to v
                neighbors_in_C = adj[v] & set(range(k))
                # Uncovered edges incident to v
                unc_incident = [(v, u) if (v, u) in uncovered else (u, v) 
                               for u in neighbors_in_C if (v, u) in uncovered or (u, v) in uncovered]
                if not unc_incident:
                    continue
                # We can pick any subset S of neighbors_in_C such that all (v, u) for u in S are uncovered
                # To maximize gain, pick all such u
                S = set()
                for u in neighbors_in_C:
                    if (v, u) in uncovered or (u, v) in uncovered:
                        S.add(u)
                if len(S) < 1:
                    continue
                clique = [v] + list(S)
                # Edges covered: (v, u) for u in S
                edges_covered = set()
                for u in S:
                    e = (min(v, u), max(v, u))
                    if e in uncovered:
                        edges_covered.add(e)
                gain = len(edges_covered)
                if gain > best_gain:
                    best_gain = gain
                    best_clique = clique
                    best_edges = edges_covered
            
            # Option 2: Clique entirely in C
            # Find a large subset of C that covers many uncovered C-C edges
            # For simplicity, try adding the largest possible clique in C that covers uncovered edges
            # Since C is a clique, any subset is a clique.
            # We want to pick a subset S of C such that many edges within S are uncovered.
            # Greedy: start with all vertices in C that have uncovered edges, then...
            # Actually, let's just try the whole C first, then remove vertices if needed.
            # But we need edge-disjointness, so we can only use edges that are uncovered.
            
            # For C-C edges, let's find a clique (subset of C) that maximizes covered uncovered edges
            # This is equivalent to finding a dense subgraph in the graph of uncovered C-C edges.
            # For small k (up to 30), we can do this reasonably.
            
            # Build graph of uncovered C-C edges
            c_vertices = list(range(k))
            unc_cc_edges = [(u, v) for (u, v) in uncovered if u < k and v < k]
            
            if unc_cc_edges:
                # Greedy: find a vertex with high degree in unc_cc_edges, build clique around it
                deg = {v: 0 for v in c_vertices}
                for u, v in unc_cc_edges:
                    deg[u] += 1
                    deg[v] += 1
                
                # Try to build a clique greedily
                # Start with the vertex of highest degree
                if c_vertices:
                    start_v = max(c_vertices, key=lambda x: deg.get(x, 0))
                    current_clique = {start_v}
                    # Add vertices that are connected to all in current_clique in unc_cc_edges
                    candidate = list(c_vertices)
                    candidate.remove(start_v)
                    for v in candidate:
                        if all((min(v, u), max(v, u)) in unc_cc_edges for u in current_clique):
                            current_clique.add(v)
                    
                    if len(current_clique) >= 2:
                        clique = list(current_clique)
                        edges_covered = set()
                        for i in range(len(clique)):
                            for j in range(i + 1, len(clique)):
                                e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                                if e in uncovered:
                                    edges_covered.add(e)
                        gain = len(edges_covered)
                        if gain > best_gain:
                            best_gain = gain
                            best_clique = clique
                            best_edges = edges_covered
            
            if best_clique is None or best_gain <= 0:
                # Fallback: pick any uncovered edge as a clique of size 2
                e = next(iter(uncovered))
                best_clique = [e[0], e[1]]
                best_edges = {e}
                best_gain = 1
            
            cliques.append(best_clique)
            uncovered -= best_edges
        
        # Convert to list of lists of ints
        result = []
        for c in cliques:
            result.append([int(v) for v in c])
        return result
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() - start_time > DEADLINE:
            # Return empty lists to avoid timeout (will be invalid, but better than exception)
            results.append([])
            continue
        result = solve_one(n, k, p, seed)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
