"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(edges, n, k):
        """
        Strategy:
        1. Identify cross edges (u, v) with u < k <= v.
        2. For each u in K, let N_I(u) be the set of independent vertices connected to u.
        3. A clique containing u and some subset S of N_I(u) is valid if S is a clique in the induced graph on I? 
           No, I is an independent set. So any clique containing u and vertices from I can contain at most one vertex from I? 
           Wait. The problem states I is an independent set. 
           If I is independent, there are NO edges between vertices in I.
           Therefore, a clique can contain at most ONE vertex from I.
           
           Let's re-read the definition of split graph and clique partition constraints.
           "Every listed set must be a clique... every edge must lie in exactly one listed clique."
           
           If I is an independent set, then for any v in I, N(v) is a subset of K.
           Any clique containing v can only contain vertices from N(v) (which are in K) and v itself.
           It cannot contain any other vertex from I because there are no edges between them.
           
           So, any clique C containing a vertex v in I is of the form {v} ∪ S_v, where S_v ⊆ N(v) ∩ K.
           For C to be a clique, S_v must be a clique in the subgraph induced by K.
           Since K is a clique, ANY subset of K is a clique.
           So, for any v in I and any subset S_v of N(v), {v} ∪ S_v is a clique.
           
           The edges covered by {v} ∪ S_v are:
           - Edges between v and each u in S_v (cross edges)
           - Edges between pairs of vertices in S_v (internal K edges)
           
           We need to partition ALL edges.
           Let E_cross be the set of cross edges.
           Let E_K be the set of edges within K.
           
           If we choose a collection of cliques C_1, ..., C_m where each C_i contains exactly one vertex v_i from I,
           then the cross edges covered are exactly the edges incident to v_i and the vertices in S_i.
           Since each cross edge (u, v) is incident to exactly one vertex v in I, it must be covered by the unique clique containing v.
           Therefore, for each v in I, the clique containing v MUST cover ALL edges incident to v.
           This implies that S_v must be exactly N(v).
           
           Wait, is that true?
           "every edge must lie in exactly one listed clique"
           If an edge (u, v) is not in the clique containing v, it must be in some other clique.
           But any clique containing (u, v) must contain both u and v.
           Since v is in I, and I is independent, any clique containing v can only contain neighbors of v from K.
           So the ONLY clique that can cover edge (u, v) is the one that contains v.
           Therefore, the clique containing v MUST include all neighbors of v in K.
           
           So, for each v in I, we are FORCED to have a clique C_v = {v} ∪ N(v).
           These cliques cover all cross edges.
           They also cover the edges within N(v) for each v.
           
           The remaining edges to cover are the edges in K that are NOT covered by any C_v.
           An edge (u, w) in K is covered by C_v if and only if u ∈ N(v) AND w ∈ N(v).
           i.e., if v is a common neighbor of u and w in I.
           
           So the problem reduces to:
           1. Create cliques C_v = {v} ∪ N(v) for all v in I.
           2. Identify the set of edges E_remaining in K that are not covered by any C_v.
           3. Partition E_remaining into the minimum number of cliques (subsets of K).
           
           Since K is a clique, any subset of K is a clique.
           To minimize the number of cliques covering E_remaining, we should use the largest possible cliques.
           The largest clique is K itself. But we can only use vertices that are still "available" to cover edges?
           No, the constraint is that every edge must be in EXACTLY one clique.
           The cliques C_v already "used up" the edges within N(v).
           We cannot use those edges again.
           
           So we have a graph G_rem on vertex set K, where an edge (u, w) exists in G_rem if and only if (u, w) is an edge in K AND there is NO v in I such that u, w ∈ N(v).
           We need to partition the edges of G_rem into the minimum number of cliques.
           
           Since the vertex set is K (size k), and G_rem is a subgraph of the complete graph K,
           we can simply cover G_rem with cliques.
           Since any subset of vertices in K forms a clique in the original graph, 
           the problem is equivalent to finding a minimum clique cover of the graph G_rem.
           
           However, G_rem is a graph on k vertices. k is at most 30.
           Minimum clique cover is NP-hard in general, but for k=30, we can use efficient heuristics or exact methods if the graph is sparse/dense.
           Actually, since K is a clique, G_rem is just a graph where some edges are missing.
           A clique in G_rem is just a set of vertices in K that are mutually adjacent in G_rem.
           
           Wait, if G_rem has an edge (u,w), it means NO v in I is adjacent to BOTH u and w.
           If G_rem has NO edge (u,w), it means there EXISTS some v in I adjacent to both.
           
           Let's check the size.
           n=30, k=10. I has 20 vertices.
           n=90, k=30. I has 60 vertices.
           
           The graph G_rem on k vertices.
           We need to partition its edges into cliques.
           Since k is small (<=30), we can use a greedy approach or branch and bound.
           A simple greedy: repeatedly pick the largest clique in G_rem, remove its edges, repeat.
           This is not optimal but good.
           
           Better: Since the vertex set is small, we can model this as an exact cover or use a simple recursive search with pruning if k is very small.
           But k=30 might be too big for naive B&B.
           
           However, note that G_rem is a subgraph of a complete graph.
           Actually, we can just cover the edges of G_rem with single edges? That would be |E_rem| cliques.
           We want to minimize the number.
           
           Let's try a greedy "Maximal Clique" removal.
           1. Find a maximal clique in G_rem.
           2. Add it to the partition.
           3. Remove all edges in that clique from G_rem.
           4. Repeat until no edges left.
           
           This is O(k^3) roughly. Very fast.
           
           Let's verify the "forced" nature of C_v.
           Is it possible to cover a cross edge (u, v) with a clique that does NOT contain v?
           No, a clique containing edge (u, v) must contain both u and v.
           So yes, each cross edge is tied to its endpoint in I.
           And since I is independent, no clique can contain two vertices from I.
           So each v in I has exactly one clique containing it.
           That clique MUST contain all neighbors of v in K to cover all edges incident to v.
           So C_v = {v} ∪ N(v) is fixed.
           
           So the only optimization is in covering the residual edges in K.
        """
        # 1. Build N(v) for v in I
        N = {v: set() for v in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
        
        cliques = []
        
        # 2. Create C_v for each v in I
        # We only need to create C_v if it has at least 2 vertices?
        # "Every listed set must be a clique with at least two vertices"
        # If N(v) is empty, C_v = {v}. This is size 1.
        # But if N(v) is empty, v has no edges. So it doesn't need to be in any clique?
        # The rule says "every edge must lie in exactly one listed clique".
        # It does NOT say every vertex must be in a clique.
        # So if N(v) is empty, we don't create a clique for v.
        # If N(v) is not empty, C_v has size >= 2.
        
        covered_K_edges = set()
        
        for v in range(k, n):
            if N[v]:
                clique_v = [v] + sorted(list(N[v]))
                cliques.append(clique_v)
                # Mark edges within N[v] as covered
                neighbors = sorted(list(N[v]))
                for i in range(len(neighbors)):
                    for j in range(i+1, len(neighbors)):
                        covered_K_edges.add((neighbors[i], neighbors[j]))
        
        # 3. Determine remaining edges in K
        remaining_edges = []
        for u, v in edges:
            if u < k and v < k:
                if (u, v) not in covered_K_edges:
                    remaining_edges.append((u, v))
        
        # 4. Partition remaining_edges into minimum number of cliques
        # The vertices involved are a subset of K.
        # We can use a greedy approach on the graph formed by remaining_edges.
        
        if remaining_edges:
            # Build adjacency for remaining graph
            rem_adj = {u: set() for u in range(k)}
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Greedy Maximal Clique Removal
            # To be slightly better than random, we can try to find the largest clique?
            # Or just any maximal clique.
            # For k=30, finding largest clique is hard, but finding a maximal one is easy.
            # Actually, we can just group edges by a vertex?
            # Star cover? No, cliques are better.
            
            # Simple greedy:
            # While edges remain:
            #   Pick a vertex with highest degree in rem_graph.
            #   Find a maximal clique containing it.
            #   Add to cliques.
            #   Remove those edges.
            
            rem_edges_set = set(remaining_edges)
            
            while rem_edges_set:
                if time.time() > deadline:
                    # Fallback: cover remaining edges as single edges (size 2 cliques)
                    for u, v in rem_edges_set:
                        cliques.append([u, v])
                    break
                
                # Find a vertex with max degree in the current remaining graph
                deg = {u: len(rem_adj[u] & {v for u,v in rem_edges_set for v in [u,v] if u!=v}) for u in rem_adj}
                # Simpler degree calc:
                deg = {}
                for u in rem_adj:
                    deg[u] = 0
                for u, v in rem_edges_set:
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
                
                if not deg:
                    break
                    
                max_v = max(deg, key=deg.get)
                
                # Find a maximal clique in the remaining graph containing max_v
                # Greedy: start with {max_v}, add neighbors that are connected to all existing in clique
                clique = {max_v}
                candidates = list(rem_adj[max_v])
                
                # Sort candidates by degree in remaining graph to prioritize high-degree nodes?
                candidates.sort(key=lambda x: deg.get(x, 0), reverse=True)
                
                for c in candidates:
                    # Check if c is connected to all in clique
                    if all((min(c, nc), max(c, nc)) in rem_edges_set or (min(nc, c), max(nc, c)) in rem_edges_set for nc in clique):
                        clique.add(c)
                
                # Convert clique to edge list and remove from rem_edges_set
                clique_list = sorted(clique)
                clique_edges = set()
                for i in range(len(clique_list)):
                    for j in range(i+1, len(clique_list)):
                        u, v = clique_list[i], clique_list[j]
                        edge = (u, v) if u < v else (v, u)
                        if edge in rem_edges_set:
                            clique_edges.add(edge)
                
                if clique_edges:
                    # Only add if clique has at least 2 vertices
                    if len(clique) >= 2:
                        cliques.append(clique_list)
                    rem_edges_set -= clique_edges
                
                # Update rem_adj? Not strictly necessary if we check rem_edges_set, 
                # but deg becomes stale. For k=30, recalculating is fine.
                # However, the `candidates` check uses `rem_edges_set`, so it's correct.
                # The loop `while rem_edges_set` ensures we terminate.
                
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
