"""Clique partition of split graphs using structural decomposition."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Generate edges exactly as in the seed program to ensure same graph
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        """
        Returns a list of cliques (lists of int vertices) partitioning the edges of the split graph.
        Structure:
        - Vertices 0..k-1 form a Clique C.
        - Vertices k..n-1 form an Independent Set I.
        - Edges exist within C, and between C and I with probability p.
        
        Strategy:
        1. The clique C itself covers all edges within C.
        2. For each v in I, let N_v be its neighbors in C.
           The set {v} U N_v is a clique in the graph.
           It covers all edges (u, v) for u in N_v.
           Since I is independent, no edges exist between vertices in I.
           Thus, these cliques for different v in I are disjoint in terms of edges covered.
           Specifically, an edge (u, v) with u in C, v in I is covered ONLY by the clique corresponding to v.
        3. We must ensure each listed clique has at least 2 vertices.
           - C has k vertices. If k >= 2, it's valid.
           - {v} U N_v has size 1 + |N_v|. It is valid if |N_v| >= 1.
           - If |N_v| == 0, v has no edges, so we don't need to list any clique for it.
        
        This minimizes the number of cliques because:
        - We cannot merge {v} U N_v with {v'} U N_v' because v and v' are not adjacent (I is independent).
        - We cannot split {v} U N_v into smaller cliques without increasing the count, because any clique covering an edge (u,v) must contain v and u. If we split N_v into multiple sets, we get multiple cliques. Using the whole N_v is one clique.
        - We cannot merge C with any {v} U N_v because v is not in C and v is not adjacent to all of C (only N_v). Even if v were adjacent to all of C, the union would be a clique, but we already established v is not in C. Wait, if we merged C and {v} U N_v, we'd get C U {v}. This is a clique if v is connected to all C. But then we'd cover edges (u,v) for all u in C. However, other vertices in I would still need their own cliques. Merging C with one v's neighborhood doesn't help cover other v's edges. And it doesn't reduce the count for v (still 1 clique for v's edges, but now it's part of a bigger clique). Actually, if we merge C and {v} U N_v, we still need cliques for other w in I. The total count would be 1 (big clique) + (|I|-1) (other v's). The separate strategy is 1 (C) + |I| (each v). 
        Let's check: 
        Strategy A (Separate): 1 (C) + sum(1 for v in I if deg(v)>0).
        Strategy B (Merge v into C): If v is connected to ALL C, then C U {v} is a clique. 
        Does this save anything? 
        In Strategy A, C is one clique. {v} U N_v is another. Total 2 for these edges.
        In Strategy B, C U {v} is one clique. It covers edges in C and edges (u,v). 
        But what about edges (u, w) for w in I, w != v? They still need cliques.
        So if we merge v into C, we save 1 clique for v, but we "use up" C. 
        Wait, the clique C U {v} covers edges in C. So we don't need a separate C clique?
        If we use C U {v} as the clique for C's internal edges AND v's external edges, that's 1 clique.
        Then for other w in I, we use {w} U N_w.
        Total: 1 + (|I| - 1) = |I|.
        Strategy A Total: 1 + |I| (assuming all connected).
        So Strategy B is better by 1 if there exists a v connected to all of C?
        Actually, if ANY v is connected to all of C, we can form C U {v}. 
        But we can only merge ONE v into C? 
        If we merge v1 and v2 into C, we get C U {v1, v2}. But v1 and v2 are NOT adjacent (I is independent). So C U {v1, v2} is NOT a clique.
        So we can merge at most ONE vertex from I into C.
        Which one? Any v that is connected to all of C.
        If such a v exists, we save 1 clique.
        If no such v exists, we cannot merge any v into C.
        
        So the optimal strategy is:
        1. Check if there is any v in I such that N(v) == C (i.e., connected to all k vertices).
        2. If yes, pick one such v*. 
           - Create clique C U {v*}. This covers all internal C edges and all edges incident to v*.
           - For all other v in I \ {v*}, create clique {v} U N(v) if N(v) is not empty.
           - Note: If v* is the only vertex in I, we just have C U {v*}.
        3. If no such v exists:
           - Create clique C.
           - For all v in I, create clique {v} U N(v) if N(v) is not empty.
        
        Let's verify the "at least 2 vertices" constraint.
        - C has k vertices. k=10, 20, 30. All >= 2.
        - C U {v*} has k+1 vertices. >= 2.
        - {v} U N(v) has size 1 + |N(v)|. Valid if |N(v)| >= 1.
        
        This logic holds.
        """
        # Re-generate graph structure to get neighbors
        rng = Random(seed)
        neighbors_in_C = [set() for _ in range(n)]
        
        # Edges within C are complete, but we don't need to store them for neighbor sets of I
        # We need neighbors in C for each v in I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors_in_C[v].add(u)
        
        cliques = []
        
        # Check for a universal vertex in I (connected to all C)
        universal_v = -1
        for v in range(k, n):
            if len(neighbors_in_C[v]) == k:
                universal_v = v
                break
        
        if universal_v != -1:
            # Merge universal_v into C
            clique_main = list(range(k)) + [universal_v]
            cliques.append(clique_main)
            
            # Handle other vertices in I
            for v in range(k, n):
                if v == universal_v:
                    continue
                if len(neighbors_in_C[v]) > 0:
                    clique_v = [v] + list(neighbors_in_C[v])
                    cliques.append(clique_v)
        else:
            # No universal vertex, use C separately
            if k >= 2:
                cliques.append(list(range(k)))
            
            # Handle all vertices in I
            for v in range(k, n):
                if len(neighbors_in_C[v]) > 0:
                    clique_v = [v] + list(neighbors_in_C[v])
                    cliques.append(clique_v)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Emergency fallback: return valid but potentially non-optimal result?
            # Better to just return what we have if we are late, but we should finish fast.
            # This loop is O(n*k) which is tiny.
            pass
        n, k, p, seed = spec
        cliques = partition_split_graph(n, k, p, seed)
        # Convert to list of lists of ints
        # The function already returns lists of ints.
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
