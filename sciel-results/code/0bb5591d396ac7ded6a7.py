"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return sorted(edges + cross_edges)

    def solve_instance(edges, n, k):
        # 1. Build adjacency and edge list
        edge_set = set()
        adj = [set() for _ in range(n)]
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
            adj[u].add(v)
            adj[v].add(u)
        
        if not edges:
            return []

        # 2. Generate Candidate Cliques
        # We restrict candidates to edges (size 2) and triangles (size 3).
        # Larger cliques are rarely optimal in split graphs for minimizing count 
        # because the "clique on K" part is dense, but cross edges are sparse.
        # However, if a vertex in I is connected to all K, a clique of size k+1 
        # is very powerful. 
        # Strategy: 
        # - Always include the full K clique if k>=2? 
        #   Actually, the ILP will decide. But the K clique covers k(k-1)/2 edges 
        #   with 1 clique. This is always better than covering them individually 
        #   unless it forces other edges to be covered inefficiently? 
        #   No, in a set cover, adding a set that covers many elements is good.
        #   But we must ensure the set is a valid clique. K is a clique.
        
        candidates = []
        cand_indices = {}
        
        # Add K as a candidate if k >= 2
        if k >= 2:
            k_clique = list(range(k))
            # Verify it's a clique (it is by definition of split graph generator)
            # Add to candidates
            cand_indices[(tuple(k_clique))] = len(candidates)
            candidates.append(k_clique)
            
        # Add all edges as candidates
        for u, v in edges:
            edge = [u, v]
            key = (u, v)
            if key not in cand_indices:
                cand_indices[key] = len(candidates)
                candidates.append(edge)
                
        # Add all triangles as candidates
        # Only consider triangles that include at least one vertex from I (>=k)
        # because triangles within K are subsets of the K clique (redundant if K clique is used,
        # but useful if K clique is not used or if we want finer control? 
        # Actually, if we use K clique, we don't need triangles in K.
        # If we don't use K clique, we need to cover K edges. 
        # Using K clique is almost always optimal for the K part.
        # Let's add triangles involving I vertices.
        
        # To speed up triangle detection, iterate over edges (u,v) and find common neighbors w
        # such that {u,v,w} is a clique.
        # We only care about triangles that cover cross edges.
        # A triangle (u, v, w) where u,v in K, w in I covers (u,v), (u,w), (v,w).
        # A triangle (u, w, x) where u in K, w,x in I is impossible (w,x not connected).
        # So only triangles of form (u, v, w) with u,v < k, w >= k.
        
        # For each w in I, find pairs (u,v) in K such that w is connected to both.
        # This creates triangle (u, v, w).
        
        for w in range(k, n):
            # Neighbors of w in K
            nbrs_w_k = [u for u in range(k) if u in adj[w]]
            # For each pair u, v in nbrs_w_k, if (u,v) is an edge (it is, K is clique),
            # then (u, v, w) is a triangle.
            # We add this triangle as a candidate.
            # Note: This might generate many candidates. 
            # Number of such triangles is sum over w of C(deg_K(w), 2).
            # For n=90, k=30, p=0.7, deg ~ 21. C(21,2) ~ 210 per w. 60 * 210 = 12600 triangles.
            # Plus ~1500 edges. Total ~14000 candidates.
            # Edges to cover: ~1500.
            # ILP with 14k vars and 1500 constraints might be slow.
            
            # Optimization: Only add triangles if they seem beneficial.
            # A triangle covers 3 edges. An edge covers 1.
            # Using a triangle is better if the 3 edges are not covered by other "cheap" sets.
            # The K clique covers all K-K edges.
            # So, if we use K clique, we only need to cover cross edges.
            # Cross edges (u,w) can be covered by:
            # 1. Edge (u,w)
            # 2. Triangle (u,v,w) where v is another neighbor of w in K.
            #    But (u,v) is covered by K clique.
            #    So triangle (u,v,w) covers (u,w) and (v,w) "for free" (since (u,v) is already covered).
            #    Wait, if (u,v) is covered by K clique, we don't need to cover it again.
            #    In Set Cover formulation:
            #    Constraint for edge e: sum(x_s for s containing e) >= 1.
            #    If K clique is selected, it satisfies all K-K edges.
            #    Then for cross edges, we just need to satisfy them.
            #    A triangle (u,v,w) covers (u,w) and (v,w).
            #    So it's a 2-cover for cross edges (effectively).
            #    An edge (u,w) covers 1 cross edge.
            #    So triangles are "better" for covering cross edges if they share the K-K edge which is already covered.
            
            # Let's just add them. The ILP solver (HiGHS) is quite fast for small/medium instances.
            for i in range(len(nbrs_w_k)):
                u = nbrs_w_k[i]
                for j in range(i + 1, len(nbrs_w_k)):
                    v = nbrs_w_k[j]
                    # (u, v, w) is a triangle
                    tri = [min(u, v), max(u, v), w]
                    key = tuple(tri)
                    if key not in cand_indices:
                        cand_indices[key] = len(candidates)
                        candidates.append(tri)

        num_candidates = len(candidates)
        num_edges = len(edges)
        
        if num_candidates == 0:
            return []

        # 3. Build ILP
        # Variables x_j in {0,1} for each candidate j
        # Minimize sum(x_j)
        # Subject to: for each edge e_i, sum(x_j for j containing e_i) >= 1
        
        c = np.ones(num_candidates)
        
        # A matrix: num_edges x num_candidates
        # A[i, j] = 1 if candidate j contains edge i
        
        # To save memory and time, we can build A sparsely or just densely if small.
        # Max candidates ~ 15000. Max edges ~ 1500.
        # 1500 * 15000 = 22.5 million floats. ~180 MB. Acceptable.
        
        A = np.zeros((num_edges, num_candidates), dtype=np.float64)
        
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        for j, cand in enumerate(candidates):
            # For each pair in cand, check if it's an edge
            # cand is a list of vertices forming a clique
            # All pairs in cand are edges.
            for a_idx in range(len(cand)):
                for b_idx in range(a_idx + 1, len(cand)):
                    u, v = cand[a_idx], cand[b_idx]
                    e = (min(u, v), max(u, v))
                    if e in edge_to_idx:
                        i = edge_to_idx[e]
                        A[i, j] = 1.0

        # Constraints: A @ x >= 1
        # milp expects A_ub @ x <= b_ub or A_eq @ x == b_eq
        # So we use -A @ x <= -1
        A_ub = -A
        b_ub = -np.ones(num_edges)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: 1 for all
        integrality = np.ones(num_candidates)
        
        # Solve
        try:
            res = milp(c=c, constraints=LinearConstraint(A_ub, ub=b_ub), 
                       integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                # Extract solution
                selected = np.where(res.x > 0.5)[0]
                cliques = []
                for j in selected:
                    cliques.append(candidates[j])
                return cliques
            else:
                # Fallback: Greedy Set Cover
                return greedy_set_cover(edges, candidates, edge_to_idx)
        except Exception:
            return greedy_set_cover(edges, candidates, edge_to_idx)

    def greedy_set_cover(edges, candidates, edge_to_idx):
        # Simple greedy: pick candidate covering most uncovered edges
        uncovered = set(edges)
        cliques = []
        # Pre-compute edges covered by each candidate
        cand_edges = []
        for cand in candidates:
            es = set()
            for a in range(len(cand)):
                for b in range(a+1, len(cand)):
                    u, v = cand[a], cand[b]
                    e = (min(u, v), max(u, v))
                    if e in edge_to_idx:
                        es.add(e)
            cand_edges.append(es)
            
        # Sort candidates by size descending? Or just greedy step by step.
        # To be fast, we can just iterate.
        # Since n is small, even O(N^2) is fine.
        
        available_indices = list(range(len(candidates)))
        
        while uncovered:
            if time.time() > deadline:
                # Emergency: return remaining edges as singletons
                for e in uncovered:
                    cliques.append(list(e))
                return cliques
                
            best_idx = -1
            best_count = 0
            for idx in available_indices:
                count = len(cand_edges[idx].intersection(uncovered))
                if count > best_count:
                    best_count = count
                    best_idx = idx
                    if best_count == len(uncovered):
                        break
            
            if best_idx == -1:
                break
                
            cliques.append(candidates[best_idx])
            uncovered -= cand_edges[best_idx]
            available_indices.remove(best_idx)
            
        # If any edges remain (shouldn't happen if candidates cover all edges), add them
        for e in uncovered:
            cliques.append(list(e))
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        if time.time() > deadline - 5:
            # Not enough time, use simplest valid partition
            cliques = [list(e) for e in edges]
            # Ensure K is a clique if possible to save count?
            # Just return edges as cliques. Valid but high count.
            results.append(cliques)
            continue
            
        cliques = solve_instance(edges, n, k)
        
        # Verify types
        final_cliques = []
        for c in cliques:
            # Ensure list of ints
            final_cliques.append([int(v) for v in c])
        results.append(final_cliques)
        
    return results
# EVOLVE-BLOCK-END
