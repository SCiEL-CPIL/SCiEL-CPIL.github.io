def try_merge_bicliques(bicliques, adj, deadline):
    """Try to merge compatible bicliques to reduce count."""
    if len(bicliques) <= 1:
        return bicliques
    
    # Pre-compute edge sets for all bicliques for fast intersection checks
    biclique_edge_sets = []
    for L, R in bicliques:
        edges = set()
        for u in L:
            for v in R:
                edges.add(frozenset((u, v)))
        biclique_edge_sets.append(edges)
        
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            for j in range(i+1, len(bicliques)):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                L1s, R1s = set(L1), set(R1)
                L2s, R2s = set(L2), set(R2)
                
                # Check disjointness of vertices
                if L1s & L2s or R1s & R2s or L1s & R2s or L2s & R1s:
                    continue
                
                # Check all cross edges exist in the original graph
                valid = True
                new_cross_edges = set()
                for u in L1s:
                    for v in R2s:
                        e = frozenset((u, v))
                        if e not in ALL_EDGES:
                            valid = False
                            break
                        new_cross_edges.add(e)
                    if not valid:
                        break
                if valid:
                    for u in L2s:
                        for v in R1s:
                            e = frozenset((u, v))
                            if e not in ALL_EDGES:
                                valid = False
                                break
                            new_cross_edges.add(e)
                            if not valid:
                                break
                
                if not valid:
                    continue
                
                # Check if new cross edges conflict with other bicliques
                conflict = False
                for k in range(len(bicliques)):
                    if k == i or k == j:
                        continue
                    # If any new edge is already covered by biclique k, conflict
                    if new_cross_edges & biclique_edge_sets[k]:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Merge
                new_L = list(L1s | L2s)
                new_R = list(R1s | R2s)
                new_bicliques = []
                new_edge_sets = []
                for k in range(len(bicliques)):
                    if k != i and k != j:
                        new_bicliques.append(bicliques[k])
                        new_edge_sets.append(biclique_edge_sets[k])
                
                # Compute edge set for the new merged biclique
                merged_edges = biclique_edge_sets[i] | biclique_edge_sets[j] | new_cross_edges
                new_bicliques.append((new_L, new_R))
                new_edge_sets.append(merged_edges)
                
                bicliques = new_bicliques
                biclique_edge_sets = new_edge_sets
                changed = True
                break
            if changed:
                break
    
    return bicliques
