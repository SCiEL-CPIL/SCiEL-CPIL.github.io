import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        # K is 0..k-1, I is k..n-1
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def greedy_clique_cover(remaining_edges, nodes_involved):
        """
        Greedily cover remaining edges with cliques.
        remaining_edges: set of (u, v) tuples
        nodes_involved: set of nodes that have remaining edges
        Returns list of cliques
        """
        cliques = []
        remaining = set(remaining_edges)
        
        while remaining:
            # Pick an edge
            u, v = next(iter(remaining))
            
            # Find maximal clique containing u, v
            # Candidates: nodes x such that (u,x) and (v,x) are in remaining
            candidates = set()
            for a, b in remaining:
                if a == u and b != v:
                    candidates.add(b)
                elif b == u and a != v:
                    candidates.add(a)
                if a == v and b != u:
                    candidates.add(b)
                elif b == v and a != u:
                    candidates.add(a)
            
            valid_c = []
            for c in candidates:
                if (min(u, c), max(u, c)) in remaining and (min(v, c), max(v, c)) in remaining:
                    valid_c.append(c)
            
            if valid_c:
                # Try to extend further? For now, just take one
                # Actually, let's try to find a larger clique greedily
                clique = [u, v]
                used = {u, v}
                # Repeatedly add a node that connects to all in clique
                while True:
                    added = False
                    # Sort candidates by degree in remaining graph for heuristic
                    cand_list = []
                    for c in valid_c:
                        if c not in used:
                            # Check if c connects to all in clique
                            ok = True
                            for node in clique:
                                e = (min(c, node), max(c, node))
                                if e not in remaining:
                                    ok = False
                                    break
                            if ok:
                                cand_list.append(c)
                    
                    if not cand_list:
                        break
                    
                    # Pick one with highest degree in remaining
                    def deg(c):
                        count = 0
                        for a, b in remaining:
                            if a == c or b == c:
                                count += 1
                        return count
                    
                    best_c = max(cand_list, key=deg)
                    clique.append(best_c)
                    used.add(best_c)
                    added = True
                    if not added:
                        break
                    
            else:
                clique = [u, v]
            
            # Remove edges covered by this clique
            edges_to_remove = set()
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    edges_to_remove.add((min(a, b), max(a, b)))
            
            remaining -= edges_to_remove
            cliques.append(clique)
            
        return cliques

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Set of available K-edges
        available_k_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        
        cliques = []
        
        # Process each v in I
        # Sort by degree to handle high degree ones first
        vs = sorted([v for v in range(k, n)], key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs:
            if time.time() > deadline:
                # If we run out of time, we must still cover remaining edges
                # Break and let the final cover handle it
                break
            
            N_v = neighbors[v]
            if not N_v:
                continue
                
            # Find maximal matching in the subgraph of available_k_edges induced by N_v
            matched_nodes = set()
            pairs = []
            
            # Create list of potential edges
            potential_edges = []
            for i in range(len(N_v)):
                for j in range(i + 1, len(N_v)):
                    u, w = N_v[i], N_v[j]
                    edge = (min(u, w), max(u, w))
                    if edge in available_k_edges:
                        potential_edges.append(edge)
                        
            # Greedy matching with heuristic: pick edges with higher degree first
            # Calculate degree of each edge in the induced subgraph
            edge_deg = {}
            for e in potential_edges:
                edge_deg[e] = 0
            for e in potential_edges:
                u, w = e
                for e2 in potential_edges:
                    if e2 != e and (u in e2 or w in e2):
                        edge_deg[e] += 1
            
            # Sort by degree descending
            potential_edges.sort(key=lambda e: edge_deg[e], reverse=True)
            
            for edge in potential_edges:
                u, w = edge
                if u not in matched_nodes and w not in matched_nodes:
                    pairs.append((u, w, v))
                    matched_nodes.add(u)
                    matched_nodes.add(w)
                    available_k_edges.remove(edge)
            
            # Create cliques for pairs
            for u, w, v_node in pairs:
                cliques.append([u, w, v_node])
                
            # Create cliques for unmatched neighbors (singles)
            for u in N_v:
                if u not in matched_nodes:
                    cliques.append([u, v])
                    
        # Cover remaining K-edges
        if available_k_edges:
            nodes_involved = set()
            for u, v in available_k_edges:
                nodes_involved.add(u)
                nodes_involved.add(v)
            
            remaining_cliques = greedy_clique_cover(available_k_edges, nodes_involved)
            cliques.extend(remaining_cliques)
                
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Emergency fallback
            pass
        cliques = solve_instance(*spec)
        # Convert to list of lists of ints
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
