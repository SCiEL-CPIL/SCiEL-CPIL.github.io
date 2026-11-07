"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 145

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_clique_greedy(vertices, adj):
        """
        Find a large clique greedily.
        vertices: iterable of vertices
        adj: dict of vertex -> set of neighbors
        Returns: list of vertices forming a clique
        """
        if not vertices:
            return []
        
        # Sort by degree descending
        v_list = sorted(vertices, key=lambda x: len(adj.get(x, set())), reverse=True)
        
        clique = []
        candidates = set(v_list)
        
        for v in v_list:
            if v not in candidates:
                continue
            # Check if v connects to all in current clique
            if all(u in adj.get(v, set()) for u in clique):
                clique.append(v)
                # Prune candidates to those connected to v
                new_candidates = set()
                for w in candidates:
                    if w == v:
                        continue
                    if v in adj.get(w, set()):
                        new_candidates.add(w)
                candidates = new_candidates
                if not candidates:
                    break
                    
        return clique

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency sets
        adj = {i: set() for i in range(n)}
        for (a, b) in edges:
            adj[a].add(b)
            adj[b].add(a)
        
        # C vertices: 0..k-1, I vertices: k..n-1
        C = list(range(k))
        I = list(range(k, n))
        
        # Precompute N_C for each I vertex
        N_C = {v: adj[v].intersection(C) for v in I}
        
        # Sort I vertices by |N_C| descending
        I_sorted = sorted(I, key=lambda v: len(N_C[v]), reverse=True)
        
        cliques = []
        
        # Track remaining edges in C-C subgraph
        # We start with all C-C edges present
        # We will remove them as we form cliques containing I vertices or C-only cliques
        
        # Instead of tracking edges, we can track which C vertices are "available" for pairing
        # But a C vertex can be in multiple cliques? No, edge partition.
        # Each edge (u,v) in C must be in exactly one clique.
        # So if we form clique {v, u, w}, edges (u,v), (w,v), (u,w) are covered.
        # Wait, (u,v) and (w,v) are cross edges. (u,w) is a C-C edge.
        # So forming {v, u, w} covers cross edges (v,u), (v,w) and C-C edge (u,w).
        
        # We need to cover all C-C edges.
        # Let's maintain a set of "uncovered" C-C edges.
        uncovered_cc = set()
        for u in C:
            for w in C:
                if u < w and (u, w) in adj:
                    uncovered_cc.add((u, w))
        
        # We also need to cover all cross edges (u, v) where u in C, v in I.
        # Let's process I vertices one by one.
        # For each I vertex v, we want to form as many cliques {v, S} as possible
        # where S is a subset of N_C[v] such that S is a clique in the remaining C-C graph
        # (i.e., all pairs in S have their edge in uncovered_cc).
        # Actually, if S is a clique in the original graph, and we haven't covered the edges in S yet,
        # then {v} + S is a valid clique.
        
        # Strategy:
        # For each v in I_sorted:
        #   Let U = N_C[v]
        #   While U is not empty:
        #     Find a maximal clique S in U with respect to uncovered_cc
        #     If S has size >= 2:
        #       Add clique [v] + S
        #       Remove edges in S from uncovered_cc
        #       Remove S from U
        #     Else:
        #       # No pair in U has an uncovered edge between them?
        #       # Or just pick one?
        #       # If we pick one u, clique [v, u] covers edge (v,u) but no C-C edge.
        #       # This is suboptimal if we can pair them.
        #       # But if no pair in U has an uncovered edge, we must use size 2 cliques.
        #       # Pick arbitrary u, add [v, u], remove u from U.
        #       pass
        
        # Let's implement this.
        
        for v in I_sorted:
            if time.time() > deadline - 5:
                break
            U = set(N_C[v])
            while U:
                if time.time() > deadline - 5:
                    break
                # Find max clique in U using uncovered_cc as edge set
                # Build local adj for U based on uncovered_cc
                local_adj = {u: set() for u in U}
                for u in U:
                    for w in U:
                        if u < w:
                            if (u, w) in uncovered_cc:
                                local_adj[u].add(w)
                                local_adj[w].add(u)
                
                S = max_clique_greedy(U, local_adj)
                
                if len(S) >= 2:
                    cliques.append([v] + S)
                    # Remove edges in S from uncovered_cc
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            a, b = S[i], S[j]
                            e = (a, b) if a < b else (b, a)
                            uncovered_cc.discard(e)
                    U -= set(S)
                else:
                    # No pair in U has an uncovered edge
                    # Pick one
                    u = next(iter(U))
                    cliques.append([v, u])
                    U.discard(u)
        
        # Cover remaining uncovered_cc with C-only cliques
        while uncovered_cc:
            if time.time() > deadline - 2:
                # Fallback: individual edges
                for e in uncovered_cc:
                    cliques.append(list(e))
                uncovered_cc = set()
                break
            
            # Find max clique in vertices of uncovered_cc
            vertices = set()
            for e in uncovered_cc:
                vertices.add(e[0])
                vertices.add(e[1])
            
            # Build local adj
            local_adj = {u: set() for u in vertices}
            for e in uncovered_cc:
                local_adj[e[0]].add(e[1])
                local_adj[e[1]].add(e[0])
            
            S = max_clique_greedy(vertices, local_adj)
            
            if not S:
                e = next(iter(uncovered_cc))
                S = list(e)
            
            cliques.append(S)
            
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    a, b = S[i], S[j]
                    e = (a, b) if a < b else (b, a)
                    uncovered_cc.discard(e)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 10:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for (u, v) in edges:
                if not (u < k and v < k):
                    cliques.append([u, v])
            results.append(cliques)
        else:
            results.append(solve_instance(*spec))
    
    return results
# EVOLVE-BLOCK-END
