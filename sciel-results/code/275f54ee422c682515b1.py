"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        start = time.time()
        deadline = start + 130  # safety margin before 150s kill

        rng = Random(seed)
        # Build C-C edges (clique among first k vertices)
        cc_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                cc_edges.add((u, v))

        # Build C-I edges
        ci_edges = []
        ci_set = set()
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    ci_edges.append((u, v))
                    ci_set.add((u, v))

        # For each I-vertex, compute its neighbor set in C
        i_neighbors = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if (u, v) in ci_set:
                    nbrs.add(u)
            i_neighbors[v] = nbrs

        cliques = []

        # 1. Handle C-C edges: group them into maximal cliques
        # Since C is a clique, we can cover C-C edges with few cliques.
        # Strategy: greedily form large cliques from C-vertices.
        # A clique in C is just a subset of C (since all C-vertices are mutually adjacent).
        # We need to cover all C-C edges. The minimum number of cliques to cover K_k edges
        # is ceil(k-1) (using stars from vertex 0, or similar). But we can do better
        # by grouping.
        # Actually, for a complete graph K_k, the minimum edge clique cover is ceil(k/2)
        # if k is even (perfect matching of cliques of size 2... no).
        # Wait: we need cliques (vertex sets) such that every edge is in exactly one.
        # For K_k, one clique {0,1,...,k-1} covers all edges. That's 1 clique!
        # But we also need to handle C-I edges, and those cliques can't include I-vertices
        # in the same clique as C-C edges unless the I-vertex is adjacent to all C-vertices in it.
        
        # Let's think differently: 
        # - C-C edges: can be covered by one big clique C = {0,...,k-1}
        # - C-I edges: each needs a clique containing that C-I edge
        # - But if we use C as a clique, then C-I edges can't be in C (since I-vertices aren't in C)
        # - So C-I edges need separate cliques.
        # 
        # Alternative: don't use the full C clique. Instead, for each I-vertex v,
        # form a clique {v} ∪ N(v) ∩ C. This covers all C-I edges incident to v.
        # But C-C edges within N(v)∩C are also covered by this clique.
        # We need to ensure every C-C edge is covered exactly once.
        
        # Strategy:
        # 1. For each I-vertex v, create clique {v} ∪ N_C(v). This covers all (c,v) edges for c in N_C(v).
        # 2. The C-C edges covered by these cliques are those within N_C(v) for various v.
        # 3. Remaining C-C edges need to be covered by additional cliques (subsets of C).
        # 4. We must ensure no C-C edge is covered twice.
        
        # This is tricky. Let me use a different approach:
        # Process edges one by one, greedily forming maximal cliques.
        
        # Better approach:
        # - Start with all edges uncovered.
        # - Repeatedly: pick an uncovered edge, find the largest clique containing it,
        #   remove all edges in that clique, add clique to result.
        # - This is a standard greedy edge clique cover.
        
        # For efficiency, let's be smarter about split graph structure.
        
        # Approach:
        # 1. For each I-vertex v with neighbors N_C(v) in C:
        #    If N_C(v) is non-empty, form clique {v} ∪ N_C(v).
        #    This covers all C-I edges incident to v and C-C edges within N_C(v).
        # 2. Track which C-C edges are now covered.
        # 3. For remaining C-C edges, cover them with additional cliques.
        
        # But we must avoid double-covering C-C edges.
        # If two I-vertices v1, v2 both have c in N_C(v1) ∩ N_C(v2), then edge (c, c') for c' in the intersection
        # might be covered twice.
        
        # To avoid this: assign each C-C edge to at most one I-vertex's clique.
        # Greedy: for each I-vertex v, its clique covers C-C edges within N_C(v), but only those
        # not already covered by previously processed I-vertices.
        
        # Let me implement:
        covered_cc = set()  # covered C-C edges
        all_cc = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_cc.add((u, v))
        
        # Sort I-vertices by size of N_C(v) descending (greedy: cover more first)
        i_vertices = [v for v in range(k, n) if i_neighbors[v]]
        i_vertices.sort(key=lambda v: len(i_neighbors[v]), reverse=True)
        
        for v in i_vertices:
            if time.time() > deadline:
                break
            nbrs = i_neighbors[v]
            if not nbrs:
                continue
            # Clique is {v} ∪ nbrs
            clique = [v] + sorted(nbrs)
            # Mark C-C edges within nbrs as covered
            nbrs_list = sorted(nbrs)
            for i_idx in range(len(nbrs_list)):
                for j_idx in range(i_idx + 1, len(nbrs_list)):
                    e = (nbrs_list[i_idx], nbrs_list[j_idx])
                    covered_cc.add(e)
            cliques.append(clique)
        
        # Remaining C-C edges
        remaining_cc = all_cc - covered_cc
        if remaining_cc:
            # Cover remaining C-C edges with cliques on C-vertices
            # Greedy: repeatedly pick largest subset of C that forms a clique
            # (which is any subset since C is a clique) and covers many remaining edges.
            # Just use: for each remaining edge, we can group them.
            # Simple: use a greedy set cover on edges.
            # For K_k, we can cover with cliques of size ~k/2.
            # Greedy: while remaining edges, pick a vertex with max degree in remaining,
            # form clique with it and its neighbors in remaining graph.
            
            rem_set = set(remaining_cc)
            while rem_set:
                if time.time() > deadline:
                    break
                # Pick vertex with max degree in remaining graph
                deg = {}
                for (a, b) in rem_set:
                    deg[a] = deg.get(a, 0) + 1
                    deg[b] = deg.get(b, 0) + 1
                # Pick vertex with max degree
                max_v = max(deg, key=lambda x: deg[x])
                # Find all vertices adjacent to max_v in remaining graph
                adj = set()
                for (a, b) in rem_set:
                    if a == max_v:
                        adj.add(b)
                    elif b == max_v:
                        adj.add(a)
                # Clique: {max_v} ∪ adj (all mutually adjacent since in C)
                clique = [max_v] + sorted(adj)
                if len(clique) < 2:
                    # Shouldn't happen, but handle
                    clique = [max_v]
                    # Find one edge
                    for (a, b) in rem_set:
                        if a == max_v:
                            clique.append(b)
                            break
                        elif b == max_v:
                            clique.append(a)
                            break
                cliques.append(clique)
                # Remove all edges within this clique
                clique_set = set(clique)
                new_rem = set()
                for (a, b) in rem_set:
                    if a in clique_set and b in clique_set:
                        continue
                    new_rem.add((a, b))
                rem_set = new_rem
        
        # Validate: check no edge is covered twice and all edges are covered
        # For now, trust the algorithm. Let's do a quick validation.
        edge_count = {}
        for clique in cliques:
            cs = set(clique)
            cl = sorted(cs)
            for i_idx in range(len(cl)):
                for j_idx in range(i_idx + 1, len(cl)):
                    e = (cl[i_idx], cl[j_idx])
                    edge_count[e] = edge_count.get(e, 0) + 1
        
        # All original edges must be covered exactly once
        all_edges = cc_edges | ci_set
        for e in all_edges:
            if edge_count.get(e, 0) != 1:
                # Fallback: return edge-by-edge (seed behavior)
                return [list(e) for e in sorted(all_edges)]
        for e, cnt in edge_count.items():
            if e not in all_edges or cnt != 1:
                return [list(e) for e in sorted(all_edges)]
        
        # Filter out cliques with < 2 vertices
        cliques = [c for c in cliques if len(c) >= 2]
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_one(*spec))
    
    # Ensure all types are correct
    final = []
    for cliques in results:
        final.append([list(c) for c in cliques])
    return final
# EVOLVE-BLOCK-END
