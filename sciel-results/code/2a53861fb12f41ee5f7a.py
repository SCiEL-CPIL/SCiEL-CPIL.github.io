<<<<<<< SEARCH
        def find_large_clique(adj, n, rng=None, max_iter=5000):
            """Find a large clique using greedy expansion from best seed edge."""
            if not any(adj):
                return []

            # Find edge with maximum common neighbors
            best_u, best_v = -1, -1
            best_common = -1
            for u in range(n):
                if adj[u] == 0:
                    continue
                for v in iter_bits(adj[u]):
                    if v > u:
                        common = (adj[u] & adj[v]).bit_count()
                        if common > best_common:
                            best_common = common
                            best_u, best_v = u, v

            if best_u == -1:
                return []

            # Expand clique greedily
            C = [best_u, best_v]
            U = adj[best_u] & adj[best_v]

            while U:
                # Find vertex in U with max connections within U
                best_candidate = -1
                best_score = -1
                candidates = list(iter_bits(U))
                if rng is not None and len(candidates) > 1:
                    rng.shuffle(candidates)
                for u in candidates:
                    score = (adj[u] & U).bit_count()
                    if score > best_score:
                        best_score = score
                        best_candidate = u
                if best_candidate == -1:
                    break
                C.append(best_candidate)
                U &= adj[best_candidate]

            return C
=======
        def edge_clique_potential(u, v, adj):
            """Compute potential for edge (u,v) based on clique density ratio."""
            common = (adj[u] & adj[v]).bit_count()
            deg_u = adj[u].bit_count()
            deg_v = adj[v].bit_count()
            if deg_u == 0 or deg_v == 0:
                return 0.0
            # Higher potential for edges in denser regions
            return common / (deg_u + deg_v - common + 1e-9)

        def find_large_clique(adj, n, rng=None, max_iter=5000):
            """Find a large clique using greedy expansion from best seed edge by clique density."""
            if not any(adj):
                return []

            # Find edge with maximum clique density potential
            best_u, best_v = -1, -1
            best_potential = -1.0
            for u in range(n):
                if adj[u] == 0:
                    continue
                for v in iter_bits(adj[u]):
                    if v > u:
                        pot = edge_clique_potential(u, v, adj)
                        if pot > best_potential:
                            best_potential = pot
                            best_u, best_v = u, v

            if best_u == -1:
                return []

            # Expand clique greedily, prioritizing vertices with high internal density
            C = [best_u, best_v]
            U = adj[best_u] & adj[best_v]

            while U:
                # Find vertex in U with max connections within U (density-based)
                best_candidate = -1
                best_score = -1
                candidates = list(iter_bits(U))
                if rng is not None and len(candidates) > 1:
                    rng.shuffle(candidates)
                for u in candidates:
                    # Score by how well connected u is within the remaining candidate set U
                    score = (adj[u] & U).bit_count()
                    if score > best_score:
                        best_score = score
                        best_candidate = u
                if best_candidate == -1:
                    break
                C.append(best_candidate)
                U &= adj[best_candidate]

            return C
>>>>>>> REPLACE

