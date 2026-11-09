    def construct():
        best_edges = None
        best_count = float('inf')
        
        # Try n from 5 to 12
        for n in range(5, 13):
            # Generate all 5-uniform edges
            all_edges = list(itertools.combinations(range(n), 5))
            num_edges = len(all_edges)
            
            # If num_edges is too large, B&B might be slow?
            # n=12, edges=792. 2^12=4096 colorings.
            # It should still be okay with good pruning.
            
            # Generate all colorings as bitmasks
            # Coloring is a tuple of 0/1. We map it to an index 0..2^n-1.
            # For each edge, we compute a bitmask of colorings it violates (makes monochromatic).
            
            # Optimization:
            # A coloring is an integer mask of n bits.
            # Edge e = (v1, v2, v3, v4, v5).
            # It is monochromatic if bit(v1)==bit(v2)==...==bit(v5).
            
            # We iterate through all 2^n colorings.
            # For each coloring, we check which edges are monochromatic.
            # This builds the "universe" coverage.
            # Actually, we want for each edge, the set of colorings.
            # It's easier to iterate colorings and mark edges?
            # No, we need edge -> colorings mask.
            
            # Let's precompute edge_masks.
            # edge_masks[i] is an integer bitmask of length 2^n.
            # Bit j is 1 if coloring j makes edge i monochromatic.
            
            # 2^n can be up to 4096 (n=12). Python ints handle this fine.
            
            edge_masks = [0] * num_edges
            
            for c_idx in range(1 << n):
                # Decode coloring c_idx to bits
                # We can just check bits directly
                for e_idx, edge in enumerate(all_edges):
                    # Check if monochromatic
                    # Get color of first vertex
                    c0 = (c_idx >> edge[0]) & 1
                    mono = True
                    for v in edge[1:]:
                        if ((c_idx >> v) & 1) != c0:
                            mono = False
                            break
                    if mono:
                        edge_masks[e_idx] |= (1 << c_idx)
                        
            # Now we have edge_masks.
            # We want to find min number of edges such that OR of their masks == (1 << (2^n)) - 1.
            
            # B&B
            full_mask = (1 << (1 << n)) - 1
            
            # Sort edges by coverage size (popcount) descending to help heuristics
            # Actually, we can just use the list.
            
            best_for_n = float('inf')
            best_edges_for_n = None
            
            # Recursive B&B
            # state: current_mask (covered colorings), current_depth, start_idx (for exclusion branching)
            # We use a greedy choice for the first branch to get a good upper bound quickly.
            
            # Greedy initial solution
            curr_mask = 0
            used_edges = []
            remaining = full_mask
            edges_indices = list(range(num_edges))
            
            # Simple greedy to get upper bound
            while remaining:
                # Find edge with max coverage of remaining
                best_e = -1
                best_cov = 0
                for e in edges_indices:
                    cov = (edge_masks[e] & remaining).bit_count()
                    if cov > best_cov:
                        best_cov = cov
                        best_e = e
                if best_e == -1:
                    break
                used_edges.append(best_e)
                remaining &= ~edge_masks[best_e]
                edges_indices.remove(best_e)
                
            if remaining == 0:
                best_for_n = len(used_edges)
                best_edges_for_n = [all_edges[e] for e in used_edges]
            
            # Now B&B to find better
            # We need to be careful with recursion depth and time.
            # n=12 might be slow. Let's limit n to 10 or 11 first.
            # Known m(5) is 51? No, "best known construction has 51 edges".
            # If we find 51, we are good. If we find 40, better.
            
            # To ensure we finish in 120s, we should limit the search.
            # If n=12 is too slow, we skip it or use a timeout.
            
            start_time = time.time()
            
            # B&B function
            def solve_bnb(covered, depth, candidates):
                nonlocal best_for_n, best_edges_for_n
                if depth >= best_for_n:
                    return
                if covered == full_mask:
                    if depth < best_for_n:
                        best_for_n = depth
                        # Reconstruct edges? We need to track them.
                        # This is tricky in B&B without storing the path.
                        # Let's store the path in a global list.
                        pass
                if time.time() - start_time > 50: # Time limit per n
                    return
                
                # Lower bound
                # Max coverage of any single edge in candidates
                max_cov = 0
                best_e = -1
                for e in candidates:
                    cov = (edge_masks[e] & ~covered).bit_count()
                    if cov > max_cov:
                        max_cov = cov
                        best_e = e
                if max_cov == 0:
                    return
                lb = depth + (-(-((full_mask ^ covered).bit_count()) // max_cov)) # ceil division
                if lb >= best_for_n:
                    return
                
                # Branch 1: Include best_e
                new_covered = covered | edge_masks[best_e]
                new_candidates = [e for e in candidates if e != best_e]
                # Track path
                path_copy = current_path[:]
                path_copy.append(best_e)
                current_path = path_copy
                solve_bnb(new_covered, depth + 1, new_candidates)
                
                # Branch 2: Exclude best_e
                # We don't include it. We just remove it from candidates.
                # Is this sufficient? Yes, standard set cover B&B.
                new_candidates_ex = [e for e in candidates if e != best_e]
                # Note: We might prune more if we realize best_e is necessary, but simple exclusion is safe.
                solve_bnb(covered, depth, new_candidates_ex)
                
            # Initialize path
            current_path = []
            # We need to capture the path when we update best.
            # Let's modify solve_bnb to take a path argument.
            
            def solve_bnb_v2(covered, depth, candidates, path):
                nonlocal best_for_n, best_edges_for_n
                if depth >= best_for_n:
                    return
                if covered == full_mask:
                    if depth < best_for_n:
                        best_for_n = depth
                        best_edges_for_n = [all_edges[e] for e in path]
                    return
                if time.time() - start_time > 50:
                    return
                
                # Heuristic: Pick edge with max marginal coverage
                max_cov = 0
                best_e = -1
                # Optimization: If candidates is large, this loop is slow.
                # But n<=12, candidates <= 792.
                for e in candidates:
                    cov = (edge_masks[e] & ~covered).bit_count()
                    if cov > max_cov:
                        max_cov = cov
                        best_e = e
                if max_cov == 0:
                    return
                
                lb = depth + ((full_mask ^ covered).bit_count() + max_cov - 1) // max_cov
                if lb >= best_for_n:
                    return
                
                # Include
                new_covered = covered | edge_masks[best_e]
                new_candidates = [e for e in candidates if e != best_e]
                new_path = path + [best_e]
                solve_bnb_v2(new_covered, depth + 1, new_candidates, new_path)
                
                # Exclude
                # To improve pruning, we can also try excluding other edges if best_e is "dominated"?
                # Simple exclusion:
                solve_bnb_v2(covered, depth, new_candidates, path)

            solve_bnb_v2(0, 0, list(range(num_edges)), [])
            
            if best_for_n < best_count:
                best_count = best_for_n
                best_edges = best_edges_for_n
                # If we found a very good one, we can stop early?
                # Theoretical lower bound is 29. If we find 29, we can stop.
                if best_count <= 29:
                    break

        if best_edges:
            return {'n': n, 'edges': best_edges}
        else:
            # Fallback
            return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
    