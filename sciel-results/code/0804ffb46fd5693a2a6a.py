def construct():
    import time
    from random import Random

    deadline = time.time() + 145.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_setcover(n, k, p, seed, variant_seed=None):
        """
        Model: Cover all edges of the split graph with cliques.
        Edges are: C-edges (within clique C), cross edges (between C and I).
        A clique can be: a subset of C (covers C-edges), or {v} ∪ S where v∈I, S⊆C
        (covers C-edges within S plus cross edges from v to S).
        
        We use greedy set cover: repeatedly pick the clique that covers the most
        uncovered edges, add it, and mark those edges as covered.
        """
        cross_edges = instance(n, k, p, seed)
        
        # All edges in the graph
        # C-edges: all pairs (u,w) with u<w, both in [0,k)
        # Cross edges: given
        all_edges = set()
        for u in range(k):
            for w in range(u+1, k):
                all_edges.add((u, w))
        for u, v in cross_edges:
            all_edges.add((min(u,v), max(u,v)))
        
        # Neighbor lists for I vertices
        N = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            N[v].add(u)
        
        # Precompute: for each v in I, which C-edges can be "absorbed" into a clique containing v
        # A clique {v} ∪ S covers: all C-edges within S, and cross edges (v, u) for u in S
        # It does NOT cover cross edges (v, u) for u not in S
        
        uncovered = set(all_edges)
        cliques = []
        
        # Greedy set cover with randomization
        rng_var = Random(variant_seed if variant_seed is not None else seed + 777)
        
        # For efficiency, we'll evaluate candidate cliques and pick the best
        # Candidate types:
        # 1. Pure C-cliques: any subset of C
        # 2. Mixed cliques: {v} ∪ S for v in I, S ⊆ N[v]
        
        # Strategy: at each step, generate candidate cliques and pick the one
        # that covers the most uncovered edges.
        
        # To keep it tractable, we limit candidate generation:
        # - For pure C: consider maximal cliques in the remaining C-edge graph
        # - For mixed: for each v in I, consider S = N[v] ∩ (vertices incident to uncovered edges)
        
        while uncovered:
            if time.time() > deadline:
                # Fallback: cover remaining edges as 2-cliques
                for e in list(uncovered):
                    cliques.append([int(e[0]), int(e[1])])
                break
            
            best_clique = None
            best_count = -1
            best_covered = set()
            
            # Generate candidates
            
            # Type 1: Pure C-cliques from remaining C-edges
            c_edges_remaining = set()
            for e in uncovered:
                if e[0] < k and e[1] < k:
                    c_edges_remaining.add(e)
            
            if c_edges_remaining:
                # Find a good clique in the C-vertex induced subgraph
                c_vertices = set()
                for e in c_edges_remaining:
                    c_vertices.add(e[0])
                    c_vertices.add(e[1])
                
                # Greedy maximal clique in C
                S = []
                cand = set(c_vertices)
                while cand:
                    best_v = None
                    best_deg = -1
                    for w in cand:
                        deg = sum(1 for x in cand if x != w and (min(w,x), max(w,x)) in c_edges_remaining)
                        if deg > best_deg:
                            best_deg = deg
                            best_v = w
                    if best_v is None:
                        break
                    S.append(best_v)
                    new_cand = set()
                    for w in cand:
                        if w == best_v:
                            continue
                        if (min(best_v, w), max(best_v, w)) in c_edges_remaining:
                            new_cand.add(w)
                    cand = new_cand
                
                if len(S) >= 2:
                    # Count how many uncovered edges this covers
                    covered = set()
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            e = (min(S[i], S[j]), max(S[i], S[j]))
                            if e in uncovered:
                                covered.add(e)
                    if len(covered) > best_count:
                        best_count = len(covered)
                        best_clique = [int(x) for x in S]
                        best_covered = covered
            
            # Type 2: Mixed cliques {v} ∪ S
            # For each v in I that has uncovered cross edges, try to form a clique
            for v in range(k, n):
                if time.time() > deadline:
                    break
                
                # Cross edges from v that are still uncovered
                v_cross_uncovered = set()
                for e in uncovered:
                    if v in e:
                        other = e[0] if e[1] == v else e[1]
                        if other < k:
                            v_cross_uncovered.add(other)
                
                if not v_cross_uncovered:
                    continue
                
                # S must be a clique in C, and S ⊆ v_cross_uncovered
                # (because {v}∪S covers cross edges (v,u) for u in S)
                # Also S should be a clique in C to make {v}∪S a clique
                
                # Find a good clique S within v_cross_uncovered
                # such that S is a clique in C and maximizes covered edges
                # Covered edges: C-edges within S (if uncovered) + cross edges (v,u) for u in S
                
                # Greedy: build S by adding vertices that are connected in C
                S = []
                cand = set(v_cross_uncovered)
                while cand:
                    best_w = None
                    best_score = -1
                    for w in cand:
                        # Score: number of C-edges from w to current S that are uncovered
                        # plus potential to extend
                        score = 0
                        for x in S:
                            e = (min(w, x), max(w, x))
                            if e in uncovered:
                                score += 1
                        # Also count C-edges from w to other candidates (potential)
                        for x in cand:
                            if x == w:
                                continue
                            e = (min(w, x), max(w, x))
                            if e in c_edges_remaining:
                                score += 0.5  # potential
                        if score > best_score:
                            best_score = score
                            best_w = w
                    if best_w is None:
                        break
                    S.append(best_w)
                    # Update cand: keep only neighbors of best_w in C
                    new_cand = set()
                    for w in cand:
                        if w == best_w:
                            continue
                        e = (min(best_w, w), max(best_w, w))
                        if e in c_edges_remaining:
                            new_cand.add(w)
                    cand = new_cand
                
                if len(S) >= 1:
                    # Evaluate this clique
                    covered = set()
                    # Cross edges
                    for u in S:
                        e = (min(v, u), max(v, u))
                        if e in uncovered:
                            covered.add(e)
                    # C-edges within S
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            e = (min(S[i], S[j]), max(S[i], S[j]))
                            if e in uncovered:
                                covered.add(e)
                    
                    if len(covered) > best_count:
                        best_count = len(covered)
                        best_clique = [int(v)] + [int(x) for x in S]
                        best_covered = covered
            
            # If no good candidate found, fallback to 2-clique
            if best_clique is None or best_count <= 0:
                e = next(iter(uncovered))
                best_clique = [int(e[0]), int(e[1])]
                best_covered = {e}
            
            cliques.append(best_clique)
            uncovered.difference_update(best_covered)
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    
    for (n, k, p, seed) in specs:
        best_cliques = None
        best_count = float('inf')
        
        # Try multiple variants with different random seeds for tie-breaking
        for i in range(15):
            if time.time() > deadline - 5.0:
                break
            v_seed = seed + 1000 + i
            cliques = solve_setcover(n, k, p, seed, v_seed)
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques
        
        # If we haven't found anything, use a simple fallback
        if best_cliques is None:
            cross_edges = instance(n, k, p, seed)
            best_cliques = [list(range(k))]
            for (u, v) in cross_edges:
                best_cliques.append([int(u), int(v)])
        
        result.append(best_cliques)

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
