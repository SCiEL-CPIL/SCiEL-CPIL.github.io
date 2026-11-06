"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    def instance(n, p, seed):
        rng = np.random.default_rng(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj
    
    def greedy_clique_partition(edges, adj, n):
        """Greedily partition edges into cliques."""
        edge_set = set(edges)
        covered = set()
        cliques = []
        remaining = set(edges)
        
        while remaining:
            # Find the edge with maximum common neighbors in remaining graph
            best_edge = None
            best_common = -1
            
            # For efficiency, sample if too many edges
            edge_list = list(remaining)
            if len(edge_list) > 200:
                # Sample from beginning, middle, end
                sample = edge_list[:50] + edge_list[len(edge_list)//2:len(edge_list)//2+50] + edge_list[-50:]
            else:
                sample = edge_list
            
            for (u, v) in sample:
                # Count common neighbors that are still connected via remaining edges
                common = adj[u] & adj[v]
                # Filter to those where edges are still remaining
                score = 0
                for w in common:
                    e1 = (min(u, w), max(u, w))
                    e2 = (min(v, w), max(v, w))
                    if e1 in remaining and e2 in remaining:
                        score += 1
                if score > best_common:
                    best_common = score
                    best_edge = (u, v)
            
            if best_edge is None:
                best_edge = edge_list[0]
            
            u, v = best_edge
            # Build clique greedily
            clique = [u, v]
            clique_set = {u, v}
            
            # Get candidates
            candidates = list(adj[u] & adj[v])
            # Sort by how many remaining edges they have
            def rem_degree(vertex):
                count = 0
                for e in remaining:
                    if vertex in e:
                        count += 1
                return count
            
            candidates.sort(key=rem_degree, reverse=True)
            
            for w in candidates:
                if w in clique_set:
                    continue
                # Check connectivity to all in clique
                if all(w in adj[x] for x in clique):
                    # Also check that edges exist in remaining
                    all_edges_remaining = True
                    for x in clique:
                        e = (min(w, x), max(w, x))
                        if e not in remaining:
                            all_edges_remaining = False
                            break
                    if all_edges_remaining:
                        clique.append(w)
                        clique_set.add(w)
            
            if len(clique) >= 2:
                cliques.append(clique)
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (min(a, b), max(a, b))
                        if e in edge_set:
                            covered.add(e)
                            remaining.discard(e)
        
        return cliques
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges, adj = instance(n, p, seed)
        cliques = greedy_clique_partition(edges, adj, n)
        results.append([list(c) for c in cliques])
    
    return results
# EVOLVE-BLOCK-END
