"""Rewritten program for clique partition of split graphs."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_clique_in_set(vertices, adj):
        """Find a maximum clique within a set of vertices (all must be mutually adjacent)."""
        if len(vertices) <= 1:
            return list(vertices) if vertices else []
        
        # Bron-Kerbosch with pivot to find max clique
        best = []
        
        def bronk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            # Choose pivot
            pivot = max(P | X, key=lambda v: len(adj[v] & P)) if (P | X) else None
            candidates = list(P - adj.get(pivot, set())) if pivot is not None else list(P)
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bronk(set(), set(vertices), set())
        return best

    def partition_split_graph(n, k, p, seed):
        """
        Split graph: clique C = {0..k-1}, independent set I = {k..n-1}.
        Edges within C: all present.
        Edges within I: none.
        Cross edges: present with prob p.
        
        We need to cover all edges with cliques, each clique has >= 2 vertices,
        each edge in exactly one clique.
        
        Strategy: 
        - The clique C itself can be one clique if we don't split it.
        - But we might want to partition C into smaller cliques to allow
          different cross-vertex attachments.
        
        Key insight: In a split graph, any clique consists of some subset of C 
        and some subset of I. Since I is independent, a clique can contain at 
        most one vertex from I. So each clique is either:
          (a) a subset of C (size >= 2), or
          (b) a single vertex from I plus a subset of C that are all adjacent to it.
        
        Actually wait - a clique can contain multiple vertices from I only if they're 
        adjacent, but I is independent so no two I vertices are adjacent. So each 
        clique contains at most one I vertex.
        
        So the structure is: partition the edges into cliques where each clique is 
        either a sub-clique of C, or {i} ∪ S_i where i ∈ I and S_i ⊆ C is a clique 
        (subset of C, so always a clique) such that i is adjacent to all of S_i.
        
        The edges to cover:
        1. Edges within C: (u,v) for u,v in C, u<v
        2. Cross edges: (u,v) for u in C, v in I, if edge exists
        
        For cross edges: for each i in I, let N(i) = {u in C : edge (u,i) exists}.
        The edges incident to i are (u,i) for u in N(i). These must be covered by 
        cliques that include i. Each such clique is {i} ∪ S where S ⊆ N(i) and S is 
        a clique in C (always true since C is a clique). So we need to partition 
        N(i) into subsets S_1, ..., S_m such that each |S_j| >= 1 (since clique 
        {i} ∪ S_j needs |S_j| >= 1 to have at least 2 vertices). Actually the clique 
        {i} ∪ S_j has size |S_j| + 1, so we need |S_j| >= 1.
        
        For edges within C: we need to partition the edge set of C into cliques. 
        But wait - the cliques that cover cross edges also include subsets of C. 
        Do those cliques also cover edges within C? No! A clique {i} ∪ S covers 
        edges (i, s) for s in S, and edges (s1, s2) for s1, s2 in S. So it DOES 
        cover edges within S.
        
        So we need to be careful: edges within C that are in some S_j (used for 
        cross edge covering) are already covered. The remaining edges within C 
        need to be covered by cliques consisting only of vertices from C.
        
        This is complex. Let me think differently.
        
        Alternative approach: Use a greedy/iterative algorithm.
        """
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        C = set(range(k))
        I = set(range(k, n))
        
        remaining_edges = set(edges)
        cliques = []
        
        # Greedy: repeatedly find a large clique in the remaining graph
        # that covers many remaining edges
        while remaining_edges:
            if time.time() > deadline:
                break
            
            # Find a good clique to add
            # Try to find a clique that maximizes coverage of remaining edges
            
            # Candidate: take a vertex, find its neighbors in remaining graph, 
            # find max clique among those neighbors + vertex
            
            # For efficiency, pick vertices that have high degree in remaining graph
            remaining_adj = defaultdict(set)
            for u, v in remaining_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)
            
            # Sort vertices by degree in remaining graph
            verts = sorted(remaining_adj.keys(), key=lambda v: len(remaining_adj[v]), reverse=True)
            
            best_clique = None
            best_count = 0
            
            # Try starting from high-degree vertices
            for v in verts[:20]:  # limit candidates
                # Find max clique in {v} ∪ remaining_adj[v]
                # The clique must be subset of {v} ∪ N_remaining(v)
                # And all edges in the clique must be in remaining_edges
                
                # Get common neighbors: vertices w such that (v,w) in remaining 
                # and w is adjacent to all other chosen vertices
                # Simple approach: use Bron-Kerbosch on the remaining graph induced 
                # by {v} ∪ N_remaining(v)
                
                candidates = {v} | remaining_adj[v]
                # Filter: only keep edges among candidates that are in remaining_edges
                cand_edges = set()
                for u in candidates:
                    for w in candidates:
                        if u < w and (u, w) in remaining_edges:
                            cand_edges.add((u, w))
                
                # Find max clique in this subgraph
                best_local = []
                
                def bronk_local(R, P, X):
                    nonlocal best_local
                    if not P and not X:
                        if len(R) > len(best_local):
                            best_local = list(R)
                        return
                    pivot = max(P | X, key=lambda x: len({y for y in P if (min(x,y), max(x,y)) in cand_edges})) if (P | X) else None
                    if pivot is not None:
                        # Neighbors of pivot in P
                        piv_neigh = set()
                        for x in P:
                            if (min(pivot,x), max(pivot,x)) in cand_edges:
                                piv_neigh.add(x)
                        candidates_local = list(P - piv_neigh)
                    else:
                        candidates_local = list(P)
                    for x in candidates_local:
                        new_P = set()
                        for y in P:
                            if (min(x,y), max(x,y)) in cand_edges:
                                new_P.add(y)
                        new_X = set()
                        for y in X:
                            if (min(x,y), max(x,y)) in cand_edges:
                                new_X.add(y)
                        bronk_local(R | {x}, new_P, new_X)
                        P.discard(x)
                        X.add(x)
                
                bronk_local(set(), set(candidates), set())
                
                if best_local:
                    # Count how many remaining edges this clique covers
                    count = 0
                    for i in range(len(best_local)):
                        for j in range(i+1, len(best_local)):
                            a, b = sorted((best_local[i], best_local[j]))
                            if (a, b) in remaining_edges:
                                count += 1
                    if count > best_count:
                        best_count = count
                        best_clique = best_local
            
            if best_clique is None or best_count == 0:
                # Fallback: just take any edge
                e = next(iter(remaining_edges))
                best_clique = list(e)
                best_count = 1
            
            # Add this clique
            if len(best_clique) >= 2:
                cliques.append(sorted(best_clique))
                # Remove all edges in this clique from remaining
                for i in range(len(best_clique)):
                    for j in range(i+1, len(best_clique)):
                        a, b = sorted((best_clique[i], best_clique[j]))
                        remaining_edges.discard((a, b))
            
            if time.time() > deadline:
                break
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Not enough time, use seed program result as fallback
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
