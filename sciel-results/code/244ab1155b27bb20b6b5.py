import random
import time

def construct():
        start_time = time.time()
        deadline = 110.0

        def instance(n, k, p, seed):
            rng = random.Random(seed)
            edges = []
            # K edges
            for u in range(k):
                for v in range(u + 1, k):
                    edges.append((u, v))
            # I-K edges
            for v in range(k, n):
                for u in range(k):
                    if rng.random() < p:
                        edges.append((u, v))
            return edges

        def solve_split(n, k, p, seed):
            # Regenerate graph to get adjacency
            rng = random.Random(seed)
            adj_k = {u: set() for u in range(k)}
            # K is a clique
            for u in range(k):
                for v in range(u+1, k):
                    adj_k[u].add(v)
                    adj_k[v].add(u)
            
            # I vertices and their neighbors in K
            neighbors_in_k = []
            for v in range(k, n):
                nks = set()
                for u in range(k):
                    if rng.random() < p:
                        nks.add(u)
                neighbors_in_k.append(nks)
            
            cliques = []
            # covered_edges_in_k: set of tuples (u, v) with u < v
            covered_in_k = set()
            
            # Step 1: Process I vertices
            for v in range(k, n):
                if time.time() > deadline:
                    # Fallback
                    # This shouldn't happen given the speed
                    break
                nks = neighbors_in_k[v - k]
                if nks:
                    clique = list(nks) + [v]
                    cliques.append(clique)
                    # Mark edges in nks as covered
                    nks_list = list(nks)
                    for i in range(len(nks_list)):
                        for j in range(i + 1, len(nks_list)):
                            u, w = nks_list[i], nks_list[j]
                            covered_in_k.add((min(u, w), max(u, w)))
            
            # Step 2: Process remaining K edges
            # Find all pairs in K not in covered_in_k
            remaining_edges = []
            for u in range(k):
                for w in range(u + 1, k):
                    if (u, w) not in covered_in_k:
                        remaining_edges.append((u, w))
            
            if remaining_edges:
                # Build graph for remaining edges
                rem_adj = {u: set() for u in range(k)}
                rem_edge_set = set(remaining_edges)
                for u, w in remaining_edges:
                    rem_adj[u].add(w)
                    rem_adj[w].add(u)
                
                # Greedy maximal clique extraction
                # Since remaining_edges is small, this is fast
                while rem_edge_set:
                    if time.time() > deadline:
                        # Fallback: add remaining edges as single-edge cliques
                        for e in rem_edge_set:
                            cliques.append([e[0], e[1]])
                        rem_edge_set.clear()
                        break
                    
                    # Find a maximal clique
                    # Start with a vertex with highest degree in rem_graph
                    # Or just pick the first edge
                    # For small graphs, Bron-Kerbosch or simple greedy works
                    # Simple greedy:
                    # Pick vertex with max degree
                    max_deg_v = -1
                    max_deg = -1
                    for v in rem_adj:
                        if len(rem_adj[v]) > max_deg:
                            max_deg = len(rem_adj[v])
                            max_deg_v = v
                    
                    if max_deg_v == -1:
                        break
                        
                    clique = [max_deg_v]
                    candidates = set(rem_adj[max_deg_v])
                    
                    while candidates:
                        # Pick candidate with max degree within candidates
                        best_c = None
                        best_d = -1
                        for c in candidates:
                            d = len(rem_adj[c] & candidates)
                            if d > best_d:
                                best_d = d
                                best_c = c
                        if best_c is None:
                            break
                        clique.append(best_c)
                        candidates.discard(best_c)
                        candidates = candidates & rem_adj[best_c]
                    
                    if len(clique) >= 2:
                        cliques.append(clique)
                        # Remove edges
                        for i in range(len(clique)):
                            for j in range(i + 1, len(clique)):
                                u, w = clique[i], clique[j]
                                rem_edge_set.discard((min(u, w), max(u, w)))
                    else:
                        # Should not happen if rem_edge_set not empty
                        # But just in case, take one edge
                        if rem_edge_set:
                            e = next(iter(rem_edge_set))
                            cliques.append([e[0], e[1]])
                            rem_edge_set.discard(e)

            # Clean up types
            clean_cliques = []
            for c in cliques:
                clean_cliques.append([int(x) for x in c])
            return clean_cliques

        results = []
        specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
        
        for spec in specs:
            if time.time() > deadline:
                # Fallback: return edges as cliques (invalid if >1 edge, but we need to return something)
                # Actually, if we time out, we should return what we have or a valid fallback.
                # Given the speed, this won't happen.
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            else:
                res = solve_split(*spec)
                results.append(res)
                
        return results
