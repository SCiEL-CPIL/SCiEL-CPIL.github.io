import time
import random

def construct():
        start_time = time.time()
        deadline = start_time + 110

        def instance(n, k, p, seed):
            rng = random.Random(seed)
            neighbors = {v: [] for v in range(k, n)}
            for v in range(k, n):
                for u in range(k):
                    if rng.random() < p:
                        neighbors[v].append(u)
            return neighbors

        def solve_instance(n, k, p, seed):
            neighbors = instance(n, k, p, seed)
            
            # Identify vertices in I with |N(v)| >= 2
            candidates = []
            for v in range(k, n):
                if len(neighbors[v]) >= 2:
                    candidates.append(v)
            
            # Sort by size descending
            candidates.sort(key=lambda v: len(neighbors[v]), reverse=True)
            
            V_grouped = []
            for v in candidates:
                valid = True
                for u in V_grouped:
                    if len(set(neighbors[v]) & set(neighbors[u])) > 1:
                        valid = False
                        break
                if valid:
                    V_grouped.append(v)
            
            # Determine covered internal edges
            covered_edges = set()
            for v in V_grouped:
                ns = neighbors[v]
                for i in range(len(ns)):
                    for j in range(i+1, len(ns)):
                        covered_edges.add((ns[i], ns[j]))
            
            # Determine remaining internal edges
            rem_edges = set()
            for u in range(k):
                for w in range(u+1, k):
                    if (u, w) not in covered_edges:
                        rem_edges.add((u, w))
            
            # Calculate kappa(rem_edges)
            # Greedy clique cover for rem_edges
            cliques_for_rem = []
            remaining = set(rem_edges)
            
            # Helper to find a maximal clique in the graph defined by 'remaining' edges on vertices 0..k-1
            def find_max_clique(edges, vertices):
                if not edges:
                    return []
                # Pick vertex with max degree
                deg = {v: 0 for v in vertices}
                for u, w in edges:
                    deg[u] += 1
                    deg[w] += 1
                # Find vertex with max degree
                max_v = max(vertices, key=lambda x: deg[x])
                # Neighbors of max_v in the graph
                neigh = set()
                for u, w in edges:
                    if u == max_v: neigh.add(w)
                    elif w == max_v: neigh.add(u)
                
                # The clique must be subset of {max_v} U neigh
                # And must be a clique in the induced subgraph on neigh
                # Since k is small, we can try to build a clique greedily
                # Start with max_v
                current_clique = {max_v}
                candidates = list(neigh)
                
                # Greedy add vertices to current_clique
                # A vertex x can be added if it is connected to all in current_clique
                # and we haven't used it yet
                added = True
                while added:
                    added = False
                    # Sort candidates by degree in remaining graph? Or just iterate
                    # To be safe, sort by number of connections to current_clique
                    def score(x):
                        count = 0
                        for y in current_clique:
                            if (min(x,y), max(x,y)) in edges:
                                count += 1
                        return count
                    candidates.sort(key=score, reverse=True)
                    
                    for x in candidates:
                        if x in current_clique: continue
                        # Check if x connects to all in current_clique
                        ok = True
                        for y in current_clique:
                            if (min(x,y), max(x,y)) not in edges:
                                ok = False
                                break
                        if ok:
                            current_clique.add(x)
                            added = True
                            break # Restart sorting? Or just continue? 
                            # Restarting is safer for "maximal"
                            # But we just need A maximal clique to remove edges.
                            # Let's just take the first valid one found in sorted order.
                            # Actually, if we break, we only add one. 
                            # We want a large clique. 
                            # Let's not break, but check all? 
                            # If we don't break, we might add non-adjacent ones? 
                            # No, we check against ALL in current_clique. 
                            # So we can add multiple in one pass? 
                            # If we add x, then y might not connect to x. 
                            # So we should restart the loop after adding one.
                            break
                
                return list(current_clique)

            while remaining:
                # Find vertices involved in remaining edges
                verts = set()
                for u, w in remaining:
                    verts.add(u)
                    verts.add(w)
                
                # Find a maximal clique
                # We need to pass the subgraph of 'remaining'
                # My find_max_clique uses 'edges' which is the global remaining?
                # Yes, it checks membership in 'edges'.
                # But it needs to know which vertices are available.
                # The logic above assumes 'vertices' are the only ones considered.
                # And 'edges' is the set of available edges.
                
                # However, find_max_clique as written might be slow if called many times.
                # k <= 30. 
                
                clique = find_max_clique(remaining, verts)
                if not clique:
                    # Should not happen if remaining is not empty
                    # Fallback: take any edge
                    u, w = next(iter(remaining))
                    clique = [u, w]
                
                cliques_for_rem.append(clique)
                
                # Remove edges covered by this clique
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        edge = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        if edge in remaining:
                            remaining.remove(edge)
            
            # Construct final list of cliques
            result = []
            
            # Add grouped cliques
            for v in V_grouped:
                result.append(neighbors[v] + [v])
                
            # Add single cross edges for non-grouped
            for v in range(k, n):
                if v not in V_grouped:
                    for u in neighbors[v]:
                        result.append([u, v])
                        
            # Add remaining internal cliques
            for c in cliques_for_rem:
                result.append(c)
                
            return result

        results = []
        specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
        for spec in specs:
            if time.time() > deadline:
                break
            cliques = solve_instance(*spec)
            results.append(cliques)
        return results
