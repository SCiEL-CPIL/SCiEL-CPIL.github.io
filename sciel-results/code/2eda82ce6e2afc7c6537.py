"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    _DEADLINE = 145.0  # safety margin before 150s kill
    _T0 = time.time()

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_clique_partition(n, k, edges):
        """
        Find a minimum clique partition of a split graph.
        
        In a split graph, vertices are partitioned into a clique C (size k) and an independent set I.
        The edges are: all pairs in C, plus some cross edges between C and I.
        
        Key insight: Any clique in a split graph either:
        1. Is a subset of C (the clique part), or
        2. Contains at most one vertex from I (since I is independent)
        
        So every clique has the form {v} ∪ S where v ∈ I, S ⊆ C, and v is adjacent to all of S.
        Or it's a subset of C.
        
        Strategy: 
        - For each vertex v in I, let N_C(v) = {u in C : (u,v) is an edge}.
        - A clique containing v is {v} ∪ S where S ⊆ N_C(v).
        - We need to cover all edges with cliques of size ≥ 2.
        
        Edges to cover:
        1. Edges within C: (u, u') for u, u' in C
        2. Cross edges: (u, v) for u in C, v in I
        
        For cross edge (u, v), it must be in a clique containing both u and v.
        Since v is in I, the clique must be {v} ∪ S where u ∈ S ⊆ N_C(v).
        
        For edge (u, u') in C, it can be in:
        - A clique {v} ∪ S where v ∈ I, u, u' ∈ S ⊆ N_C(v), or
        - A clique that is a subset of C (e.g., {u, u'} itself, or larger subsets of C)
        
        Greedy approach:
        1. For each v in I, we can form cliques {v} ∪ S where S ⊆ N_C(v).
           The best is to use the largest possible S to cover more cross edges at once.
        2. But we also need to cover C-internal edges.
        
        Better approach: This is equivalent to covering all edges with cliques.
        In split graphs, the minimum clique partition can be found via a reduction.
        
        Let me think more carefully. Each clique in the partition is either:
        - Type A: {v} ∪ S where v ∈ I, S ⊆ N_C(v), |S| ≥ 1 (so clique size ≥ 2)
        - Type B: S ⊆ C, |S| ≥ 2
        
        Every cross edge (u,v) must be in exactly one Type A clique containing v and u.
        Every C-edge (u,u') must be in exactly one clique (Type A or Type B) containing both.
        
        For a fixed v ∈ I, if we assign a set S_v ⊆ N_C(v) to be the clique {v} ∪ S_v,
        then:
        - All cross edges (u,v) for u ∈ S_v are covered.
        - The C-edges within S_v are also covered by this clique.
        
        But cross edges (u,v) for u ∈ N_C(v) \\ S_v must be covered by... wait, they can only
        be covered by a clique containing v, which would be {v} ∪ S' where u ∈ S'. But v can
        only appear in ONE clique in the partition (since each edge is in exactly one clique,
        and v has multiple incident edges, all those edges must be in the same clique containing v? No!
        
        Wait, re-reading: "every edge must lie in exactly one listed clique." This means each edge
        belongs to exactly one clique. A vertex can appear in multiple cliques. So v ∈ I can appear
        in multiple cliques, each covering different edges incident to v.
        
        So for v ∈ I, its incident edges { (u,v) : u ∈ N_C(v) } must be partitioned among cliques
        of the form {v} ∪ S_1, {v} ∪ S_2, ... where S_1, S_2, ... are disjoint subsets of N_C(v).
        
        To minimize the number of cliques, for each v we want to use as few cliques as possible.
        The minimum number of cliques for v's edges is 1 if we use {v} ∪ N_C(v) (one clique covers all).
        But that clique also covers C-edges within N_C(v).
        
        The C-edges not covered by any Type A clique must be covered by Type B cliques.
        
        So the problem reduces to:
        - For each v ∈ I, choose a partition of N_C(v) into subsets S_{v,1}, ..., S_{v,m_v}
        - The cliques are {v} ∪ S_{v,j} for each v and j
        - Plus Type B cliques to cover remaining C-edges
        - Minimize total number of cliques
        
        If we use one clique per v (i.e., {v} ∪ N_C(v)), then:
        - All cross edges are covered.
        - C-edges within N_C(v) for any v are covered.
        - Remaining C-edges are those (u, u') where there is NO v ∈ I with u, u' ∈ N_C(v).
        
        Let's compute: A C-edge (u, u') is "covered" if ∃v ∈ I such that u ∈ N_C(v) and u' ∈ N_C(v).
        Otherwise it needs a Type B clique.
        
        For Type B cliques, we need to cover the remaining C-edges with cliques (subsets of C).
        The minimum number of cliques to cover a set of edges in a complete graph on C is...
        well, we can use one big clique C itself, which covers all C-edges. But if some C-edges
        are already covered, we only need to cover the remaining ones.
        
        Actually, the simplest Type B clique is just C itself (all k vertices), which covers all C-edges.
        That's 1 clique. Or we could use smaller ones.
        
        Total cliques = |I| (one per v) + (number of Type B cliques needed for uncovered C-edges).
        
        If all C-edges are covered by Type A cliques, total = |I| = n - k.
        Otherwise, we need additional Type B cliques.
        
        Can we do better by splitting some v's edges into multiple cliques? That would increase
        the count for that v but might reduce Type B cliques. Unlikely to help.
        
        So the strategy is:
        1. For each v ∈ I, form clique {v} ∪ N_C(v). (|I| cliques)
        2. Find C-edges not covered by any of these cliques.
        3. Cover remaining C-edges with minimum number of cliques (subsets of C).
        
        For step 3, the remaining C-edges form a graph on C. We need to partition its edges
        into cliques. Since C is a clique, any subset is a clique. So we just need to partition
        the edge set of this remaining graph into the minimum number of complete subgraphs.
        
        This is the edge clique partition problem, which is NP-hard in general, but for our
        specific structure it might be easier.
        
        Actually, let me reconsider. The remaining C-edges are those (u,u') where no v has both
        u and u' in N_C(v). Let's call this set E_rem.
        
        To cover E_rem with minimum cliques: we can greedily pick the largest clique from E_rem,
        remove those edges, repeat. Or since the graph is a subgraph of K_k, we can try to find
        a good partition.
        
        For small k (10, 20, 30), we can afford a decent heuristic.
        
        Let me implement this.
        """
        # Build adjacency for C part
        # C = {0, ..., k-1}, I = {k, ..., n-1}
        
        # N_C(v) for v in I
        N_C = {}
        for v in range(k, n):
            N_C[v] = set()
        
        # Build edge set
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((min(u,v), max(u,v)))
        
        # Find cross edges to determine N_C
        for (u, v) in edges:
            if u < k and v >= k:
                N_C[v].add(u)
            elif v < k and u >= k:
                N_C[u].add(v)
        
        # Type A cliques: one per v in I
        cliques = []
        covered_C_edges = set()
        
        for v in range(k, n):
            S = N_C[v]
            if len(S) >= 1:
                clique = sorted([v] + list(S))
                cliques.append(clique)
                # Mark C-edges within S as covered
                S_list = sorted(S)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        covered_C_edges.add((S_list[i], S_list[j]))
        
        # Find remaining C-edges
        remaining_C_edges = []
        for u in range(k):
            for v in range(u+1, k):
                if (u, v) not in covered_C_edges:
                    remaining_C_edges.append((u, v))
        
        # Cover remaining C-edges with minimum cliques
        if remaining_C_edges:
            # Build adjacency for the remaining C-edge graph
            rem_adj = {u: set() for u in range(k)}
            for (u, v) in remaining_C_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Greedy: repeatedly find a maximal clique in the remaining edge graph
            remaining_edges = set(remaining_C_edges)
            while remaining_edges:
                # Find a vertex with max degree
                max_deg = -1
                start_v = None
                for (u, v) in remaining_edges:
                    deg = len(rem_adj[u])
                    if deg > max_deg:
                        max_deg = deg
                        start_v = u
                
                # Grow a clique greedily from start_v
                clique = {start_v}
                candidates = set(rem_adj[start_v])
                for cand in sorted(candidates):
                    if all(cand in rem_adj[x] for x in clique):
                        clique.add(cand)
                
                # Remove edges within this clique
                clique_list = sorted(clique)
                for i in range(len(clique_list)):
                    for j in range(i+1, len(clique_list)):
                        remaining_edges.discard((clique_list[i], clique_list[j]))
                
                if len(clique) >= 2:
                    cliques.append(clique_list)
        
        return cliques

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        return solve_clique_partition(n, k, edges)

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _DEADLINE:
            break
        cliques = solve_instance(*spec)
        # Ensure all vertices are Python ints
        clean_cliques = [[int(v) for v in clique] for clique in cliques]
        result.append(clean_cliques)
    
    return result
# EVOLVE-BLOCK-END
