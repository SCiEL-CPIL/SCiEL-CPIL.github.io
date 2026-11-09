"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from collections import defaultdict
    import random

    def instance(n, k, p, seed):
        from random import Random
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

    def get_maximal_cliques(n, k, edges):
        # In a split graph, maximal cliques are:
        # 1. The clique [0, k) if it's maximal (it is, unless some v>=k is connected to all of it, but even then [0,k) is a maximal clique? No, if v is connected to all, {0..k-1, v} is a clique. But [0..k-1] is still a clique, just not maximal if extended.
        # Actually, for split graphs, the maximal cliques are of the form {v} U N(v) for v >= k, and [0, k) itself.
        # Let's just generate candidate cliques:
        # - [0, k)
        # - For each v >= k, {v} U N(v)
        # - Also, subsets of these? No, maximal cliques are sufficient for greedy set cover usually, but maybe not optimal.
        # However, any clique is a subset of a maximal clique. If we pick a non-maximal clique, it's less efficient than picking the maximal one (covers more edges). 
        # Wait, picking a maximal clique might cover edges we don't want to cover yet? No, we just want to cover all edges exactly once.
        # Greedy set cover: pick the set that covers the most uncovered elements.
        # Using only maximal cliques is a good heuristic.
        
        cliques = []
        # The big clique
        big_clique = list(range(k))
        cliques.append(big_clique)
        
        # For each independent vertex
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        for v in range(k, n):
            # Maximal clique containing v
            # It is {v} U (neighbors of v in [0, k))
            # Note: neighbors in [k, n) are none.
            neighbors_in_clique = [u for u in range(k) if u in adj[v]]
            if len(neighbors_in_clique) >= 1:
                cliques.append([v] + neighbors_in_clique)
                
        return cliques

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
            
        # Build list of edges
        edge_list = list(edges)
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        # Get candidate cliques
        candidates = get_maximal_cliques(n, k, edges)
        
        # For each candidate, list the edges it covers
        # We need to check if the candidate is actually a clique in the graph.
        # The candidates generated above ARE cliques by construction of split graphs.
        # Let's verify and get the edge sets.
        clique_edges = []
        for c in candidates:
            c_set = set(c)
            # Check if it's a clique
            is_clique = True
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    if (c[i], c[j]) not in edges and (c[j], c[i]) not in edges:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique and len(c) >= 2:
                e_set = set()
                for i in range(len(c)):
                    for j in range(i+1, len(c)):
                        e = (min(c[i], c[j]), max(c[i], c[j]))
                        if e in edge_to_idx:
                            e_set.add(e)
                if e_set:
                    clique_edges.append(e_set)
        
        # Greedy Set Cover
        uncovered = set(edge_list)
        selected_cliques = []
        
        # Optimization: Precompute for each clique, the list of edges it covers.
        # And for each edge, which cliques cover it.
        # Since n is small, we can just iterate.
        
        # Sort candidates by size descending? Or just pick best each time.
        # To speed up, we can maintain a count of uncovered edges per clique.
        # But re-calculating is fine for small n.
        
        # Heuristic: Start with the largest cliques?
        # Greedy: pick clique with max |clique_edges[i] \cap uncovered|
        
        while uncovered:
            best_clique_idx = -1
            best_count = 0
            best_edges = None
            
            for i, e_set in enumerate(clique_edges):
                # Count how many edges in e_set are still uncovered
                # Optimization: if len(e_set) <= best_count, skip? No, intersection matters.
                # Just compute intersection size.
                count = 0
                # Small optimization: if best_count == len(uncovered), we can break? No.
                for e in e_set:
                    if e in uncovered:
                        count += 1
                        if count > best_count:
                            best_count = count
                            best_clique_idx = i
                            best_edges = e_set
                # Early termination for this clique if we already found a perfect cover?
                # If best_count == len(uncovered), we can't do better.
                if best_count == len(uncovered):
                    break
            
            if best_clique_idx == -1:
                # Should not happen if graph is covered by cliques
                # Fallback: add individual edges
                for e in list(uncovered):
                    u, v = e
                    selected_cliques.append([u, v])
                    uncovered.remove(e)
                break
            
            selected_cliques.append(clique_edges[best_clique_idx]) # Wait, we need the vertex list, not edge set
            # We stored e_set in clique_edges. We need the original vertex list.
            # Let's store tuples (vertex_list, edge_set)
            
            # Re-do storage
            # This implementation is getting messy. Let's clean up.
            pass

        # Let's rewrite partition_cliques more cleanly.
        
        return [] # Placeholder

    # Rewrite partition_cliques properly
    
    def partition_cliques_v2(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
            
        edge_list = list(edges)
        
        # Generate candidate cliques (vertex lists)
        candidates = []
        
        # 1. The main clique
        candidates.append(list(range(k)))
        
        # 2. Cliques centered at independent vertices
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        for v in range(k, n):
            neighbors_in_clique = [u for u in range(k) if u in adj[v]]
            if neighbors_in_clique:
                candidates.append([v] + neighbors_in_clique)
        
        # Precompute edge sets for candidates
        cand_info = []
        for c in candidates:
            if len(c) < 2:
                continue
            e_set = set()
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    u, v = c[i], c[j]
                    if (u, v) in edges or (v, u) in edges:
                        e_set.add((min(u, v), max(u, v)))
            if e_set:
                cand_info.append((c, e_set))
        
        uncovered = set(edge_list)
        result_cliques = []
        
        # Greedy Set Cover
        # To make it faster, we can sort candidates by size initially, but greedy picks based on uncovered count.
        
        while uncovered:
            best_idx = -1
            best_count = 0
            best_c = None
            best_e_set = None
            
            for i, (c, e_set) in enumerate(cand_info):
                # Quick check: if max possible is less than best, skip?
                # Max possible is len(e_set).
                if len(e_set) <= best_count:
                    continue
                
                count = 0
                for e in e_set:
                    if e in uncovered:
                        count += 1
                        if count > best_count:
                            best_count = count
                            best_idx = i
                            best_c = c
                            best_e_set = e_set
                if best_count == len(uncovered):
                    break
            
            if best_idx == -1:
                # No candidate covers any uncovered edge?
                # This implies an edge is not in any candidate.
                # Fallback: add edges individually
                for e in list(uncovered):
                    u, v = e
                    result_cliques.append([u, v])
                break
            
            result_cliques.append(best_c)
            uncovered.difference_update(best_e_set)
            
        # Ensure types are correct
        # result_cliques contains lists of ints.
        return result_cliques

    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        results.append(partition_cliques_v2(*spec))
    
    return results
# EVOLVE-BLOCK-END
