<<<<<<< SEARCH
    def local_search(bicliques):
        """
        Try to improve the partition by merging pairs of bicliques.
        """
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            n_bicliques = len(bicliques)
            for i in range(n_bicliques):
                for j in range(i + 1, n_bicliques):
                    if time.time() > deadline:
                        return bicliques
                        
                    # Get edges covered by biclique i and j
                    edges_i = get_edges_from_bicliques([bicliques[i]])
                    edges_j = get_edges_from_bicliques([bicliques[j]])
                    combined_edges = edges_i.union(edges_j)
                    
                    # Try to cover combined_edges with 1 biclique
                    # First try direct merge checks (fast), then fall back to greedy
                    
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    merged = False
                    # Try all 4 orientations of combining the two bicliques
                    orientations = [
                        (set(L1) | set(L2), set(R1) | set(R2)),
                        (set(L1) | set(R2), set(R1) | set(L2)),
                        (set(R1) | set(L2), set(L1) | set(R2)),
                        (set(R1) | set(R2), set(L1) | set(L2)),
                    ]
                    
                    for new_L_set, new_R_set in orientations:
                        if not new_L_set or not new_R_set:
                            continue
                        # Check if all cross edges exist in the original graph
                        valid = True
                        for u in new_L_set:
                            for v in new_R_set:
                                if not adj[u][v]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Verify that the combined biclique covers exactly combined_edges
                            covered = set()
                            for u in new_L_set:
                                for v in new_R_set:
                                    covered.add((min(u, v), max(u, v)))
                            if covered == combined_edges:
                                bicliques.pop(j)
                                bicliques.pop(i)
                                bicliques.append((list(new_L_set), list(new_R_set)))
                                improved = True
                                merged = True
                                break
                        if merged:
                            break
                    
                    if not merged:
                        # Fall back to greedy on combined edges
                        new_bicliques = greedy_set_cover(combined_edges, strategy='random')
                        if new_bicliques is not None and len(new_bicliques) == 1:
                            new_b = new_bicliques[0]
                            bicliques.pop(j)
                            bicliques.pop(i)
                            bicliques.append(new_b)
                            improved = True
                    
                    if improved:
                        break # Restart the outer loop as indices changed
                if improved:
                    break
                    
        return bicliques
=======
    def local_search(bicliques):
        """
        Try to improve the partition by merging pairs of bicliques.
        """
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            n_bicliques = len(bicliques)
            for i in range(n_bicliques):
                for j in range(i + 1, n_bicliques):
                    if time.time() > deadline:
                        return bicliques
                        
                    # Get edges covered by biclique i and j
                    edges_i = get_edges_from_bicliques([bicliques[i]])
                    edges_j = get_edges_from_bicliques([bicliques[j]])
                    combined_edges = edges_i.union(edges_j)
                    
                    # Try to cover combined_edges with 1 biclique
                    # First try direct merge checks (fast), then fall back to try_single_biclique
                    
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    merged = False
                    # Try all 4 orientations of combining the two bicliques
                    orientations = [
                        (set(L1) | set(L2), set(R1) | set(R2)),
                        (set(L1) | set(R2), set(R1) | set(L2)),
                        (set(R1) | set(L2), set(L1) | set(R2)),
                        (set(R1) | set(R2), set(L1) | set(L2)),
                    ]
                    
                    for new_L_set, new_R_set in orientations:
                        if not new_L_set or not new_R_set:
                            continue
                        # Check if all cross edges exist in the original graph
                        valid = True
                        for u in new_L_set:
                            for v in new_R_set:
                                if not adj[u][v]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Verify that the combined biclique covers exactly combined_edges
                            covered = set()
                            for u in new_L_set:
                                for v in new_R_set:
                                    covered.add((min(u, v), max(u, v)))
                            if covered == combined_edges:
                                bicliques.pop(j)
                                bicliques.pop(i)
                                bicliques.append((list(new_L_set), list(new_R_set)))
                                improved = True
                                merged = True
                                break
                        if merged:
                            break
                    
                    if not merged:
                        # Fall back to try_single_biclique on combined edges
                        result = try_single_biclique(combined_edges)
                        if result is not None:
                            bicliques.pop(j)
                            bicliques.pop(i)
                            bicliques.append(result)
                            improved = True
                    
                    if improved:
                        break # Restart the outer loop as indices changed
                if improved:
                    break
                    
        return bicliques
>>>>>>> REPLACE

