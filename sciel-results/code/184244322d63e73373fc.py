import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

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
        cliques = []
        
        # Track which K-edges are already covered by I-cliques
        # Use a set of frozensets or a 2D boolean array
        covered = [[False] * k for _ in range(k)]
        
        # Sort I vertices by neighborhood size (largest first) to maximize matching potential
        vs = sorted(range(k, n), key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs:
            N_v = neighbors[v]
            if not N_v:
                continue
            
            # Find maximal matching in the graph induced by N_v on K-edges not yet covered
            # Since K is a clique, the graph is complete minus already-covered edges
            # We want to pair up as many vertices as possible, but only if their edge is free
            matched = set()
            pairs = []
            
            # Greedy matching: iterate through N_v, try to match each unmatched vertex
            # with another unmatched vertex whose edge is not covered
            # To be more effective, we can try multiple orderings or just simple greedy
            # For small k (<=30), simple greedy is fast.
            
            # Optimization: sort N_v to make it deterministic
            N_v_list = sorted(N_v)
            
            for i in range(len(N_v_list)):
                u = N_v_list[i]
                if u in matched:
                    continue
                for j in range(i + 1, len(N_v_list)):
                    w = N_v_list[j]
                    if w in matched:
                        continue
                    # Check if edge (u, w) is available
                    if not covered[u][w]:
                        pairs.append((u, w))
                        matched.add(u)
                        matched.add(w)
                        covered[u][w] = True
                        covered[w][u] = True
                        break  # u is matched, move to next i
            
            # Add I-cliques
            for u, w in pairs:
                cliques.append([u, w, v])
            
            for u in N_v:
                if u not in matched:
                    cliques.append([u, v])
        
        # Cover remaining K-edges
        # The remaining graph is K_k minus the matching of edges covered by I-cliques.
        # We need to partition the remaining edges into cliques.
        # Since K is a clique, any subset is a clique in the original graph.
        # But in the remaining graph, some edges are missing.
        # However, the cliques we form must be cliques in the ORIGINAL graph.
        # And they must cover the remaining edges exactly once.
        
        # Simple strategy: Greedily form cliques from remaining edges.
        # Since the remaining graph is dense (only a matching removed), we can form large cliques.
        # In fact, the remaining graph is K_k minus a matching.
        # A clique in this graph cannot contain both endpoints of any removed edge.
        # So we can pick at most one vertex from each removed pair.
        # Let M be the matching of removed edges.
        # The maximum clique size is k - |M| (if we pick one from each pair and all singletons).
        # But we need to cover ALL remaining edges.
        
        # Greedy clique cover for remaining edges:
        remaining_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if not covered[u][w]:
                    remaining_edges.append((u, w))
        
        # We can cover the remaining edges by forming cliques.
        # Since the graph is very dense, we can just pick a large clique that covers many edges.
        # Actually, the remaining graph is K_k minus a matching.
        # The edges are all pairs (u, w) such that (u, w) is not in the matching.
        # We can cover these edges by:
        # 1. For each vertex u, the edges incident to u that are remaining.
        # But that would be many cliques.
        
        # Better: The remaining graph is a complete multipartite graph? No.
        # It's the complement of a matching.
        # We can cover it by 1 clique if no edges are removed.
        # If edges are removed, we can still cover it with few cliques.
        # Note that if we remove a matching M, we can cover the remaining edges by:
        # - One clique for each vertex? No.
        
        # Let's use a simple greedy:
        # While remaining_edges is not empty:
        #   Pick an edge (u, w).
        #   Find the largest clique in the remaining graph containing (u, w).
        #   Add it, and remove all its edges from remaining_edges.
        
        # Since k is small (<=30), we can do this efficiently.
        # But we need to be careful about the "largest clique" finding.
        # In a graph that is K_k minus a matching, the largest clique containing (u, w) is:
        # All vertices x such that (u, x) and (w, x) are not in the matching.
        # i.e., x is not the partner of u in M, and x is not the partner of w in M.
        # So the clique is V \ {partner(u), partner(w)} (excluding u and w themselves? No, u and w are in the clique).
        # Wait, if partner(u) = x, then (u, x) is removed. So x cannot be in the clique with u.
        # So the clique is {u, w} U {x : x != partner(u), x != partner(w), x != u, x != w}.
        
        # Let's implement this directly.
        # First, find the matching M from the covered edges.
        matching = {}
        for u in range(k):
            for w in range(u + 1, k):
                if covered[u][w]:
                    matching[u] = w
                    matching[w] = u
        
        # Now cover remaining edges.
        # We can simply form cliques as follows:
        # The remaining edges are all (u, w) where w != matching.get(u).
        # We can cover these by taking cliques of size k-1 or k-2.
        # Actually, we can cover all remaining edges with at most 2 cliques?
        # No.
        
        # Let's just use the greedy algorithm.
        # For each remaining edge, we know the maximum clique it can be part of.
        # But we need to partition the edges.
        
        # Alternative: Since the structure is simple, we can just output the remaining edges as 2-cliques?
        # That would be |E_remaining| cliques, which is O(k^2).
        # For k=30, that's 435 cliques.
        # The I-cliques are ~ sum |N(v)|/2 ~ 0.7 * 30 * 20 / 2 = 210.
        # Total ~ 645.
        # The seed score was 2.0, meaning the seed program had ~ 2 * value cliques?
        # If value is the number of cliques, and score is seed/value,
        # and current score is 2.0076, then value is seed/2.0076.
        # If seed is, say, 1000, value is 498.
        # My estimate of 645 is worse.
        
        # We need to cover the remaining K-edges with few cliques.
        # The remaining graph is K_k \ M.
        # We can cover it by:
        # For each vertex u, consider the edges (u, w) for w != matching.get(u).
        # These edges form a star minus one edge (if u is in M).
        # A star can be covered by 1 clique? No, a star is not a clique.
        # But we can group edges.
        
        # Let's try this:
        # The remaining graph is the complement of a matching.
        # It is a complete graph where some pairs are forbidden.
        # We can cover the edges by:
        # 1. If M is empty, 1 clique.
        # 2. If M is not empty, we can use 2 cliques?
        #    Let M = {(a1, b1), (a2, b2), ...}.
        #    Consider clique C1 = V \ {b1, b2, ...}.
        #    This covers all edges between vertices in C1.
        #    What edges are left?
        #    Edges between C1 and {b1, b2, ...}, and edges within {b1, b2, ...} (but there are no edges within Bi since they are matched? No, (bi, bj) is an edge unless they are matched together, which they aren't).
        #    Actually, (bi, bj) is an edge for i != j.
        #    So the remaining edges after C1 are:
        #    - (u, bi) for u in C1, i in 1..m.
        #    - (bi, bj) for i < j.
        #    This is not covered by 1 clique.
        
        # Let's just use the greedy algorithm. It's safe and should be fast enough for k=30.
        
        remaining_edges_set = set()
        for u in range(k):
            for w in range(u + 1, k):
                if not covered[u][w]:
                    remaining_edges_set.add((u, w))
        
        while remaining_edges_set:
            # Pick an edge
            edge = next(iter(remaining_edges_set))
            u, w = edge
            
            # Find the maximal clique in the remaining graph containing u and w.
            # A vertex x can be in the clique if (u, x) and (w, x) are in remaining_edges_set (or u/x, w/x not removed).
            # Since we're building from the remaining graph, we check if edges exist in remaining_edges_set.
            # But remaining_edges_set changes as we remove edges.
            # It's safer to check against the "covered" array.
            # If covered[u][x] is False and covered[w][x] is False, then x can be added.
            
            clique = {u, w}
            for x in range(k):
                if x == u or x == w:
                    continue
                if not covered[u][x] and not covered[w][x]:
                    clique.add(x)
            
            # Add this clique
            cliques.append(list(clique))
            
            # Remove all edges within this clique from remaining_edges_set
            to_remove = set()
            clique_list = list(clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    e = (min(a, b), max(a, b))
                    if e in remaining_edges_set:
                        to_remove.add(e)
            remaining_edges_set -= to_remove
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: shouldn't happen, but just in case
            # Return a trivially valid result (e.g., all edges as 2-cliques)
            # But we need to generate the graph first.
            # For safety, if we run out of time, we can't compute.
            # Let's assume it finishes.
            pass
        cliques = solve_instance(*spec)
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
