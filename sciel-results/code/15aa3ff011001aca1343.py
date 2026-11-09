import time
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0 # 10s margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_maximal_clique(vertices, available_edges):
        """
        Greedy maximal clique in the graph induced by 'vertices' with edges 'available_edges'.
        vertices: set of int
        available_edges: set of tuples (min, max)
        Returns: set of int
        """
        if not vertices:
            return set()
        
        # We want to find a large clique.
        # Greedy strategy: repeatedly pick the vertex with the highest degree in the current candidate set.
        # Then restrict candidates to its neighbors.
        
        candidates = set(vertices)
        clique = set()
        
        while candidates:
            # Calculate degrees in the induced subgraph of candidates
            # To speed up, we can just count neighbors in candidates that are connected in available_edges
            # Since k is small (<= 90), O(|V|^2) is fine.
            
            best_v = None
            best_deg = -1
            
            for v in candidates:
                d = 0
                for u in candidates:
                    if u == v: continue
                    if (min(u, v), max(u, v)) in available_edges:
                        d += 1
                if d > best_deg:
                    best_deg = d
                    best_v = v
            
            if best_v is None:
                break
            
            clique.add(best_v)
            
            # Restrict candidates to neighbors of best_v in available_edges
            new_candidates = set()
            for u in candidates:
                if u == best_v: continue
                if (min(u, best_v), max(u, best_v)) in available_edges:
                    new_candidates.add(u)
            candidates = new_candidates
            
        return clique

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = []
        
        # N(v) for v in I
        N = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if (u, v) in edges:
                    nbrs.add(u)
            N[v] = nbrs
        
        # Edges within C
        cc_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                cc_edges.add((u, v))
                
        covered_cc = set()
        
        # Sort I vertices by neighborhood size (descending)
        # This allows us to form large cliques early, covering many C-C edges
        sorted_I = sorted(range(k, n), key=lambda v: len(N[v]), reverse=True)
        
        for v in sorted_I:
            if time.time() > DEADLINE:
                # Fallback: just add 2-cliques for remaining edges? 
                # But we need to be valid. 
                # If we break, we must ensure all edges are covered.
                # We will handle remaining edges at the end.
                break
            
            nbrs = N[v]
            if not nbrs:
                continue
            
            # We want to cover all edges (u, v) for u in nbrs.
            # These edges are covered by cliques {u, v, ...} where ... is a clique in C.
            # The "available" edges in C are those not yet covered.
            # However, a clique {u, w, v} covers (u,v), (w,v) and (u,w).
            # So we need to find cliques in C such that every u in nbrs is in at least one clique.
            
            remaining_nbrs = set(nbrs)
            
            while remaining_nbrs:
                if time.time() > DEADLINE:
                    break
                
                # Find a maximal clique in C using only edges not in covered_cc
                # But we also need to ensure the clique intersects remaining_nbrs?
                # Actually, if we pick a maximal clique S in C (w.r.t available edges), 
                # and S is disjoint from remaining_nbrs, it doesn't help cover edges to v.
                # So we should restrict the search to vertices in remaining_nbrs?
                # If we restrict to remaining_nbrs, we find a clique S subset of remaining_nbrs.
                # Then {S, v} is a clique. It covers edges (u,v) for u in S.
                # It also covers edges (u,w) for u,w in S.
                
                S = find_maximal_clique(remaining_nbrs, cc_edges - covered_cc)
                
                if not S:
                    # This should not happen if remaining_nbrs is not empty and there are no covered edges between them?
                    # Wait, if all edges between remaining_nbrs are covered, find_maximal_clique returns empty?
                    # My find_maximal_clique returns a set. If candidates is not empty, it picks best_v.
                    # So S will have at least 1 vertex.
                    # If S has size 1, say {u}, then {u, v} is a valid clique.
                    pass
                
                clique = sorted(S) + [v]
                cliques.append(clique)
                
                # Mark C-C edges within S as covered
                s_list = sorted(S)
                for i in range(len(s_list)):
                    for j in range(i+1, len(s_list)):
                        covered_cc.add((s_list[i], s_list[j]))
                
                remaining_nbrs -= S
        
        # Cover remaining C-C edges
        remaining_edges = cc_edges - covered_cc
        
        if remaining_edges:
            verts_involved = set()
            for u, v in remaining_edges:
                verts_involved.add(u)
                verts_involved.add(v)
                
            current_remaining = set(remaining_edges)
            
            while current_remaining:
                if time.time() > DEADLINE:
                    # Fallback: 2-cliques
                    for u, v in current_remaining:
                        cliques.append([u, v])
                    break
                
                # Find maximal clique in the graph defined by current_remaining
                S = find_maximal_clique(verts_involved, current_remaining)
                
                if not S:
                    # Pick an edge
                    u, v = next(iter(current_remaining))
                    S = {u, v}
                
                clique = sorted(S)
                cliques.append(clique)
                
                # Remove edges of S from current_remaining
                s_list = sorted(S)
                for i in range(len(s_list)):
                    for j in range(i+1, len(s_list)):
                        current_remaining.discard((s_list[i], s_list[j]))
                        
                # Update verts_involved to only those with remaining edges
                # This speeds up subsequent iterations
                new_verts = set()
                for u, v in current_remaining:
                    new_verts.add(u)
                    new_verts.add(v)
                verts_involved = new_verts

        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        result.append(cleaned)

    return result
