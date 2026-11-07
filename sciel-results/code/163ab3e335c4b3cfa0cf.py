"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict
import sys

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def min_clique_partition_k_edges(remaining_edges, k):
        """
        Partition the remaining edges of the clique K into the minimum number of cliques.
        Since the vertices are 0..k-1 and the edges are a subset of the complete graph,
        this is equivalent to finding a coloring of the complement graph of the remaining edge set.
        For small k (<=30), we can use a greedy approach or a more sophisticated heuristic.
        However, a simple greedy maximal clique extraction is often sufficient and fast.
        """
        if not remaining_edges:
            return []
        
        rem_edges = set(remaining_edges)
        cliques = []
        
        # Precompute adjacency for the remaining graph
        # For speed, we'll just iterate.
        while rem_edges:
            # Find a maximal clique in the remaining graph
            # Greedy: pick a vertex with high degree, then greedily add neighbors
            degrees = defaultdict(int)
            for u, v in rem_edges:
                degrees[u] += 1
                degrees[v] += 1
            
            if not degrees:
                break
                
            # Pick vertex with max degree
            max_v = max(degrees, key=degrees.get)
            
            # Find neighbors of max_v in rem_edges
            neighbors = set()
            for u, v in rem_edges:
                if u == max_v:
                    neighbors.add(v)
                elif v == max_v:
                    neighbors.add(u)
            
            # Sort neighbors by degree to prioritize dense areas
            candidate_neighbors = list(neighbors)
            candidate_neighbors.sort(key=lambda x: degrees.get(x, 0), reverse=True)
            
            clique_vertices = [max_v]
            
            for cand in candidate_neighbors:
                # Check if cand is connected to all existing clique_vertices
                is_connected = True
                for cv in clique_vertices:
                    if cv == cand:
                        continue
                    if (min(cv, cand), max(cv, cand)) not in rem_edges:
                        is_connected = False
                        break
                if is_connected:
                    clique_vertices.append(cand)
            
            cliques.append(sorted(clique_vertices))
            
            # Remove all edges within this clique from rem_edges
            for i in range(len(clique_vertices)):
                for j in range(i + 1, len(clique_vertices)):
                    u, w = clique_vertices[i], clique_vertices[j]
                    edge = (min(u, w), max(u, w))
                    if edge in rem_edges:
                        rem_edges.remove(edge)
                        
            # Check deadline
            if time.time() > deadline - 5:
                # Fallback: cover each remaining edge individually
                for e in rem_edges:
                    cliques.append([e[0], e[1]])
                rem_edges.clear()
                break

        return cliques

    def clique_partition(edges, n, k):
        # 1. Identify cross edges and build adjacency for I vertices to K vertices
        neighbors_in_K = defaultdict(set)
        cross_edges = []
        
        for u, v in edges:
            if u < k and v >= k:
                neighbors_in_K[v].add(u)
                cross_edges.append((u, v))
            
        # 2. Strategy: Greedy Set Cover with Cost Estimation
        # For each I vertex v, if we assign it to a clique C_v (subset of N(v)),
        # we cover |C_v| cross edges.
        # The "cost" of not covering an edge is 1 (it becomes a 2-clique).
        # The "benefit" of covering an edge is 1.
        # However, the vertices in C_v are removed from the pool of K-vertices available for forming larger K-cliques?
        # No, the K-cliques are formed from the *edges* of K.
        # If we form a clique {u, w, v}, we cover the K-edge (u,w) and the cross edges (u,v), (w,v).
        # This saves 2 cliques compared to covering (u,w), (u,v), (w,v) separately.
        # If we form a clique {u, w, x, v}, we cover K-edges (u,w), (u,x), (w,x) and cross edges (u,v), (w,v), (x,v).
        # This saves 5 cliques compared to individual edges.
        
        # So, for each I vertex v, we want to choose a subset S_v of N(v) to form a clique {S_v, v}.
        # This covers all K-edges within S_v and all cross edges between S_v and v.
        # The remaining K-edges must be covered by K-only cliques.
        
        # We want to minimize:
        # (Number of I-cliques) + (Number of K-only cliques covering remaining K-edges) + (Number of uncovered cross edges)
        
        # Note: If v is assigned to a clique, it contributes 1 to "Number of I-cliques".
        # If v is NOT assigned to any clique (or assigned to an empty set, which is invalid), its cross edges are uncovered.
        # Wait, every I vertex with degree > 0 must be in at least one clique?
        # No, the cross edges must be covered. If v has edges to u1, u2, we can cover (u1,v) and (u2,v) in separate 2-cliques.
        # Or we can cover them in one clique {u1, u2, v} if (u1, u2) is an edge in K (which it is, K is a clique).
        # So it is always beneficial to group as many neighbors as possible into a single clique with v,
        # provided that the K-edges within that neighbor set are not "wasted" (i.e., they would have been covered by a large K-clique anyway).
        
        # Actually, since K is a clique, any subset of neighbors forms a clique.
        # The "cost" of using a K-edge (u,w) in an I-clique is that it cannot be used in a K-only clique.
        # But K-only cliques cover multiple K-edges.
        # So we need to balance:
        # Gain from I-clique: (|S_v| - 1) cross edges saved + (|S_v|*(|S_v|-1)/2 - 1) K-edges saved?
        # No.
        # Baseline: All edges are 2-cliques. Total = |E|.
        # If we form a clique C of size m, we replace m*(m-1)/2 edges with 1 clique.
        # Savings = m*(m-1)/2 - 1.
        
        # So we want to maximize total savings.
        
        # Let's try a greedy approach:
        # For each I vertex v, sort by degree descending.
        # For v, try to find the "best" subset of neighbors to include in its clique.
        # Ideally, we want a large subset.
        # But if we take all neighbors, we consume many K-edges.
        
        # Let's approximate the cost of consuming K-edges.
        # If we consume a set of K-edges E_K_used, the remaining K-edges E_K_rem must be partitioned.
        # The number of cliques needed for E_K_rem is roughly |V_K_rem| / log |V_K_rem|? No.
        # For a random graph, chromatic number is ~ d/2 log d / log log d.
        # Here the graph is dense.
        
        # Simplified Greedy:
        # 1. Start with all K-edges available.
        # 2. For each I vertex v (sorted by degree desc):
        #    a. Consider all neighbors of v.
        #    b. Try to form a clique with as many neighbors as possible.
        #    c. However, we should be careful not to fragment K too much.
        #    d. Heuristic: Only include a neighbor u in v's clique if u has high degree in the *remaining* K-graph?
        #       Or simply: Include all neighbors?
        
        # Let's test "Include all neighbors" first.
        # If v is connected to all K, we form a clique of size k+1.
        # This covers all K-edges.
        # Then no K-only cliques are needed.
        # Cross edges are all covered.
        # Total cliques = 1 (for v) + ... wait, other I vertices?
        # If all I vertices are connected to all K, we can form k+1 size cliques for each?
        # No, K-edges can only be covered once.
        
        # So, we must partition the K-edges among the I-cliques and K-cliques.
        
        # Revised Strategy:
        # 1. Calculate degree of each I vertex.
        # 2. Sort I vertices by degree descending.
        # 3. Maintain a set of "available" K-edges.
        # 4. For each I vertex v:
        #    - Find the largest clique in the subgraph of K induced by N(v) using *only* available K-edges.
        #    - Let this clique be C_v.
        #    - If |C_v| >= 2:
        #       - Add {C_v, v} to the list of cliques.
        #       - Remove all K-edges within C_v from the "available" set.
        #    - Else:
        #       - v will be covered by individual 2-cliques for its cross edges.
        # 5. Cover remaining available K-edges with K-only cliques.
        # 6. Cover all cross edges not covered by I-cliques with 2-cliques.
        
        available_k_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                available_k_edges.add((u, w))
                
        cliques = []
        covered_cross_edges = set()
        
        # Sort I vertices by degree descending
        i_vertices = [v for v in range(k, n) if neighbors_in_K[v]]
        i_vertices.sort(key=lambda v: len(neighbors_in_K[v]), reverse=True)
        
        for v in i_vertices:
            nv = neighbors_in_K[v]
            # Find max clique in K induced by nv, using only available_k_edges
            # Since k is small, we can do a simple search or greedy.
            # Greedy: pick vertex in nv with max degree in available_k_edges restricted to nv.
            
            # Build local adjacency
            local_deg = defaultdict(int)
            for u in nv:
                for w in nv:
                    if u < w and (u, w) in available_k_edges:
                        local_deg[u] += 1
                        local_deg[w] += 1
                        
            if not local_deg:
                continue
                
            max_v = max(local_deg, key=local_deg.get)
            clique_v = [max_v]
            candidates = [u for u in nv if u != max_v]
            candidates.sort(key=lambda x: local_deg.get(x, 0), reverse=True)
            
            for cand in candidates:
                ok = True
                for cv in clique_v:
                    if (min(cv, cand), max(cv, cand)) not in available_k_edges:
                        ok = False
                        break
                if ok:
                    clique_v.append(cand)
                    
            if len(clique_v) >= 2:
                cliques.append(sorted(clique_v + [v]))
                # Mark cross edges as covered
                for u in clique_v:
                    covered_cross_edges.add((u, v))
                # Remove K-edges
                for i in range(len(clique_v)):
                    for j in range(i + 1, len(clique_v)):
                        u, w = clique_v[i], clique_v[j]
                        e = (min(u, w), max(u, w))
                        if e in available_k_edges:
                            available_k_edges.remove(e)
            # Else, v is left for individual coverage

        # Cover remaining K-edges
        k_cliques = min_clique_partition_k_edges(available_k_edges, k)
        cliques.extend(k_cliques)
        
        # Cover remaining cross edges
        for u, v in cross_edges:
            if (u, v) not in covered_cross_edges:
                cliques.append([u, v])
                
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        
        # Convert to pure Python types
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
# EVOLVE-BLOCK-END
