import random
import itertools

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = set()
        # Edges within K
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Edges between K and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        U = set(edges)
        cliques = []
        
        K = list(range(k))
        I = list(range(k, n))
        
        # Precompute N[v] for v in I
        N = {v: set() for v in I}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)

        def get_clique_edges(C):
            C_list = list(C)
            es = set()
            for i in range(len(C_list)):
                for j in range(i + 1, len(C_list)):
                    u, v = C_list[i], C_list[j]
                    if u > v: u, v = v, u
                    es.add((u, v))
            return es

        while U:
            best_clique = None
            best_score = 0
            
            # 1. Try cross cliques {v} U S
            for v in I:
                # Neighbors of v that are still connected via uncovered cross edge
                N_v_uncov = {u for u in N[v] if (min(v,u), max(v,u)) in U}
                if not N_v_uncov:
                    continue
                
                # Build graph of uncovered K-edges within N_v_uncov
                # Vertices: N_v_uncov
                # Edges: (u,w) in U if u,w in N_v_uncov
                # We want S subset of N_v_uncov maximizing |S| + |E_U(S)|
                
                # Heuristic 1: S = N_v_uncov
                S1 = N_v_uncov
                score1 = len(S1) + sum(1 for u in S1 for w in S1 if u < w and (u,w) in U)
                
                # Heuristic 2: Max clique in the graph of uncovered K-edges on N_v_uncov
                # This is a small graph.
                verts = list(N_v_uncov)
                # Simple greedy for max clique or densest subgraph?
                # Let's just use S1 for now, or try a few random subsets?
                # Actually, S1 is the largest possible. Does it always give max score?
                # Score(S) = |S| + E(S).
                # If we remove a vertex x, we lose 1 (from |S|) and deg(x) edges.
                # So we should only remove x if deg(x) < 1? i.e. deg(x)=0.
                # If deg(x) >= 1, removing it decreases score.
                # So S = N_v_uncov is optimal for this v?
                # Wait, if we remove x, we might be able to add y? No, S is subset of N_v_uncov.
                # The score function is monotonic?
                # f(S) = |S| + E(S).
                # f(S \ {x}) = |S|-1 + E(S) - deg_S(x).
                # f(S) - f(S \ {x}) = 1 + deg_S(x).
                # Since deg_S(x) >= 0, f(S) >= f(S \ {x}).
                # So yes, S = N_v_uncov is always optimal for a fixed v!
                
                if score1 > best_score:
                    best_score = score1
                    best_clique = [v] + list(S1)
            
            # 2. Try pure K cliques
            # We want S subset of K maximizing E_U(S).
            # This is the maximum edge clique in the graph U_K (uncovered K edges).
            # Vertices involved in U_K:
            verts_K = set()
            for u, v in U:
                if u < k and v < k:
                    verts_K.add(u)
                    verts_K.add(v)
            
            if verts_K:
                # Heuristic: Greedy max clique in U_K
                # Or just pick the densest part?
                # Let's try a simple greedy: pick vertex with max degree in U_K, add it, 
                # restrict to its neighbors, repeat?
                # This finds a maximal clique.
                # Is the max clique the best for score?
                # Score(S) = E_U(S).
                # Max clique in U_K has the most edges among cliques of that size?
                # Not necessarily. A non-clique set S can have many edges.
                # But S MUST be a clique in the original graph?
                # Yes, "Every listed set must be a clique".
                # Since K is a clique in G, any subset of K is a clique in G.
                # So S can be ANY subset of K.
                # We want to maximize E_U(S) where S is any subset of K.
                # This is the "Maximum Edge Subgraph" problem?
                # No, we just want the subset of vertices that induces the most edges in U_K.
                # This is the "Densest Subgraph" problem, which is NP-hard, but k is small.
                # We can use a heuristic: 
                # Start with all vertices in verts_K.
                # Iteratively remove vertex with lowest degree in current induced subgraph?
                # This is a common heuristic for densest subgraph.
                
                current_verts = set(verts_K)
                while current_verts:
                    # Calculate degrees in U_K induced by current_verts
                    degs = {v: 0 for v in current_verts}
                    for u, v in U:
                        if u < k and v < k:
                            if u in current_verts and v in current_verts:
                                degs[u] += 1
                                degs[v] += 1
                    min_deg = min(degs.values())
                    if min_deg == 0:
                        # Remove all isolated?
                        to_remove = [v for v in current_verts if degs[v] == 0]
                        current_verts.difference_update(to_remove)
                        continue
                    # Remove one with min deg
                    v_to_remove = min(current_verts, key=lambda x: degs[x])
                    current_verts.remove(v_to_remove)
                    
                    # Check score for current_verts
                    if current_verts:
                        score_k = sum(1 for u in current_verts for w in current_verts if u < w and (u,w) in U)
                        if score_k > best_score:
                            best_score = score_k
                            best_clique = list(current_verts)
                
                # Also check the initial set?
                # The loop removes vertices. The best might be an intermediate set.
                # We checked inside the loop.
                
            if best_clique is None:
                # Fallback: pick any edge
                u, v = next(iter(U))
                best_clique = [u, v]
                best_score = 1
            
            # Add clique and update U
            C_edges = get_clique_edges(best_clique)
            cliques.append([int(x) for x in best_clique])
            U.difference_update(C_edges)
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_instance(*spec))
        
    return results
