"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for 120s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Build adjacency structure
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        cliques = []
        
        if time.time() > deadline:
            return [[list(e) for e in edge_set]]
        
        # Strategy:
        # 1. Cover cross edges (between clique part 0..k-1 and independent set k..n-1)
        #    by forming cliques for each independent vertex w: {w} U N(w)
        #    This covers all edges (u,w) for u in N(w) AND all edges (u,v) for u,v in N(w).
        # 2. After covering cross edges, remaining edges are within 0..k-1.
        #    These form a subgraph of K_k. Cover them with cliques.
        
        remaining_edges = set(edge_set)
        
        # Step 1: Cover cross edges
        # For each independent vertex w (k..n-1), find neighbors in 0..k-1 with uncovered edges
        indep_vertices = list(range(k, n))
        # Sort by degree (number of neighbors in clique part) descending to cover more internal edges first
        degs = {}
        for w in indep_vertices:
            degs[w] = len(adj[w] & set(range(k)))
        indep_vertices.sort(key=lambda w: degs[w], reverse=True)
        
        for w in indep_vertices:
            if time.time() > deadline:
                break
            
            # Find neighbors in 0..k-1 with uncovered edge to w
            neighbors = []
            for u in range(k):
                e = (u, w) if u < w else (w, u)
                if e in remaining_edges:
                    neighbors.append(u)
            
            if not neighbors:
                continue
            
            # Form clique {w} U neighbors
            clique = [w] + neighbors
            cliques.append(clique)
            
            # Remove all edges within this clique from remaining_edges
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e)
        
        # Step 2: Cover remaining edges within 0..k-1
        # These form a subgraph of K_k. Use greedy clique cover.
        # For a complete graph K_k, we can cover it with k*(k-1)/2 edges (size-2 cliques),
        # but we can do better by finding larger cliques.
        # Since it's a subgraph of a clique, any subset is a clique.
        # Greedy: repeatedly find a large clique in the remaining graph.
        
        if remaining_edges:
            # Build adjacency for the remaining graph on vertices 0..k-1
            rem_adj = {u: set() for u in range(k)}
            for u, v in remaining_edges:
                if u < k and v < k:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)
            
            # Greedy clique cover: repeatedly pick a maximal clique
            while remaining_edges:
                if time.time() > deadline:
                    break
                
                # Find a large clique in the remaining graph
                # Use a simple heuristic: start with a vertex of high degree, expand greedily
                best_clique = None
                best_size = 0
                
                # Try starting from each vertex
                for start in range(k):
                    if not rem_adj[start]:
                        continue
                    
                    # Greedily build a clique starting from start
                    current = [start]
                    candidates = list(rem_adj[start])
                    
                    while candidates:
                        # Pick candidate with highest degree within current clique
                        best_cand = None
                        best_deg = -1
                        for c in candidates:
                            deg = len(rem_adj[c] & set(current))
                            if deg > best_deg:
                                best_deg = deg
                                best_cand = c
                        
                        if best_cand is None:
                            break
                        
                        current.append(best_cand)
                        candidates = [c for c in candidates if c in rem_adj[best_cand]]
                    
                    if len(current) > best_size:
                        best_size = len(current)
                        best_clique = current
                
                if not best_clique or len(best_clique) < 2:
                    # No clique of size >= 2 found, just take individual edges
                    for e in list(remaining_edges):
                        cliques.append([e[0], e[1]])
                        remaining_edges.discard(e)
                    break
                
                # Add the clique
                cliques.append(best_clique)
                
                # Remove all edges within this clique
                for i in range(len(best_clique)):
                    for j in range(i + 1, len(best_clique)):
                        u, v = best_clique[i], best_clique[j]
                        e = (u, v) if u < v else (v, u)
                        if e in remaining_edges:
                            remaining_edges.discard(e)
                            rem_adj[u].discard(v)
                            rem_adj[v].discard(u)
        
        # Fallback: if we ran out of time, add remaining edges as individual cliques
        if remaining_edges:
            for e in remaining_edges:
                cliques.append([e[0], e[1]])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback: return edge cliques for remaining instances
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
        else:
            results.append(solve_split(*spec))
    
    return results
# EVOLVE-BLOCK-END
