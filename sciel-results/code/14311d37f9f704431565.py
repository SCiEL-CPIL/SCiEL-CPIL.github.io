import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    def instance(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_maximal_clique_greedy(vertices, adj):
        """
        Find a maximal clique in the subgraph induced by 'vertices'.
        adj: dict of sets, adj[u] contains v if (u,v) is an edge.
        vertices: set of vertex indices.
        """
        if not vertices:
            return set()
        
        # Sort by degree within the induced subgraph
        def deg(v):
            return len(adj[v] & vertices)
        
        sorted_verts = sorted(vertices, key=deg, reverse=True)
        clique = set()
        for v in sorted_verts:
            # Check if v is connected to all in clique
            if all(u in adj[v] for u in clique):
                clique.add(v)
        return clique

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        candidates = []
        
        # Generate candidates from single vertices
        for v in range(n):
            if time.time() > deadline:
                break
            neighbors = adj[v]
            if not neighbors:
                continue
            # Find a clique in the neighborhood
            K = find_maximal_clique_greedy(neighbors, adj)
            C = {v} | K
            if len(C) >= 2:
                candidates.append(frozenset(C))
        
        # Generate candidates from edges
        for u, v in edges:
            if time.time() > deadline:
                break
            common_neighbors = adj[u] & adj[v]
            K = find_maximal_clique_greedy(common_neighbors, adj)
            C = {u, v} | K
            if len(C) >= 2:
                candidates.append(frozenset(C))
        
        # Ensure we have at least the single edges as fallback
        # Actually, if the greedy clique finder fails to produce good candidates,
        # we might miss edges. But maximal cliques cover all edges?
        # No, a maximal clique doesn't necessarily cover all edges incident to its vertices.
        # But the set of ALL maximal cliques covers all edges.
        # Our candidates are a SUBSET of maximal cliques (one per vertex/edge seed).
        # So we MUST have fallback single edges.
        
        single_edge_candidates = [frozenset([u, v]) for u, v in edges]
        candidates.extend(single_edge_candidates)
        
        # Greedy Set Cover
        uncovered = set(edges)
        partition = []
        
        # Precompute edges for each candidate to speed up intersection
        # candidate_edges: list of (candidate, set_of_edges)
        cand_edges = []
        for C in candidates:
            c_edges = set()
            c_list = list(C)
            for i in range(len(c_list)):
                u = c_list[i]
                for j in range(i + 1, len(c_list)):
                    v = c_list[j]
                    if u > v:
                        u, v = v, u
                    c_edges.add((u, v))
            cand_edges.append((C, c_edges))
        
        # To optimize, we can remove duplicate candidates
        seen = set()
        unique_cand_edges = []
        for C, c_edges in cand_edges:
            if C not in seen:
                seen.add(C)
                unique_cand_edges.append((C, c_edges))
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_cand = None
            best_c_edges = None
            best_count = -1
            
            for C, c_edges in unique_cand_edges:
                # Count how many edges in c_edges are still uncovered
                # Optimization: if c_edges is subset of already covered, count is 0
                # We can maintain a 'covered' set or just check intersection
                # Intersection size
                count = len(c_edges & uncovered)
                if count > best_count:
                    best_count = count
                    best_cand = C
                    best_c_edges = c_edges
                    if best_count == len(c_edges): # Can't do better than full coverage
                        # Note: another candidate might have more total edges, but this one is fully uncovered.
                        # Actually, we want max count. If we find a candidate that covers all its edges,
                        # we can't beat it with a candidate that has fewer total edges.
                        # But we should keep checking if there's a candidate with MORE edges?
                        # No, 'count' is the number of *uncovered* edges covered.
                        # If best_count == len(c_edges), it means this candidate adds 'len(c_edges)' new edges.
                        # Another candidate might add more if it has more edges.
                        # So we can't break early unless we know max possible.
                        pass
            
            if best_cand is None or best_count == 0:
                # Fallback: pick any uncovered edge
                # This should not happen if single_edge_candidates are present
                # But just in case
                remaining = list(uncovered)
                if not remaining:
                    break
                u, v = remaining[0]
                partition.append([u, v])
                uncovered.remove((u, v))
            else:
                partition.append(list(best_cand))
                uncovered -= best_c_edges
        
        return partition

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
