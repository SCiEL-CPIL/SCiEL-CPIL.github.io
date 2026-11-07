"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    # Deadline guard: evaluator kills after 150s; keep a safety margin.
    t_start = time.time()
    DEADLINE = 140.0  # 10s safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        """
        Find a clique partition of the split graph.
        
        Structure:
        - Clique C = {0, ..., k-1} (all edges among these are present)
        - Independent set I = {k, ..., n-1} (no edges among these)
        - Cross edges: between C and I, each present with probability p
        
        Key insight: In a split graph, any clique must be of the form {subset of C} ∪ {subset of I that are all connected to every vertex in the subset of C}.
        
        Strategy:
        1. The clique C itself covers all k*(k-1)/2 edges within C.
        2. For cross edges, we need to cover them with cliques. A clique containing vertices from I must only include vertices from I that are all neighbors of every vertex in the C-part of the clique.
        
        Greedy approach:
        - Start with the full clique C as one clique (covers all internal C edges).
        - For remaining edges (all cross edges), use a greedy set cover:
          * Each potential clique is defined by a subset S of C and the set N(S) ∩ I (vertices in I connected to all of S).
          * But this is too many. Instead, for each vertex v in I, the neighbors in C form a set. 
          * A clique can be {v} ∪ (neighbors of v in C) if we want cliques centered at I vertices.
          * Or we can have larger cliques: {v1, v2} ∪ (common neighbors in C) if v1 and v2 share all those neighbors.
        
        Better greedy:
        - Represent the residual graph (cross edges only, since C-internal edges are covered by C).
        - Each edge is (u, v) where u in C, v in I.
        - A clique in the residual graph can include at most one vertex from I (since I is independent) and any subset of C.
        - So each clique is of the form {v} ∪ S where v in I, S ⊆ C, and every u in S is connected to v.
        - The edges covered by such a clique are {(u, v) : u in S}.
        
        So the problem reduces to: cover all cross edges, where each "set" is {v} ∪ S for v in I, S ⊆ N(v) ∩ C, and it covers the edges (u, v) for u in S.
        
        To minimize the number of cliques, for each v in I, we should use the largest possible S, which is N(v) ∩ C. So the natural clique for v is {v} ∪ N(v).
        
        But wait: can we do better by having cliques that include multiple I vertices? No, because I is an independent set, so no two I vertices can be in the same clique unless... actually, two I vertices v1, v2 can be in the same clique only if they have a common neighbor set S ⊆ C such that both are connected to all of S. But since I is independent, the clique would be {v1, v2} ∪ S. This is valid if v1 and v2 are both connected to every vertex in S.
        
        However, using cliques with multiple I vertices might not help because each such clique covers edges from both v1 and v2 to S, but we could have covered those separately with {v1} ∪ S and {v2} ∪ S using 2 cliques instead of 1. So it's better to use single-I-vertex cliques.
        
        Therefore, the optimal strategy is:
        1. One clique for C: {0, 1, ..., k-1}
        2. For each v in I, one clique: {v} ∪ N(v) where N(v) = {u in C : (u,v) is an edge}
        
        But we only need to create a clique for v if N(v) is non-empty (i.e., v has at least one cross edge). Also, each such clique must have at least 2 vertices, which is satisfied if N(v) is non-empty (since it contains v and at least one u).
        
        This gives 1 + |{v in I : deg_cross(v) > 0}| cliques.
        
        Is this optimal? Let's think: each cross edge (u, v) must be covered by some clique. That clique must contain v (since it's the only I vertex that can be with u in a clique that covers (u,v)... wait, no, the clique could contain other I vertices too, but as argued, that doesn't help). Actually, the clique covering (u,v) must contain both u and v. Since u in C and v in I, the clique is {v} ∪ S where S ⊆ C and u in S. So v must be in that clique. Therefore, each v with cross edges must appear in at least one clique that covers its cross edges. And a single clique {v} ∪ N(v) covers all cross edges incident to v. So we need at least one clique per v with cross edges, plus the C clique.
        
        So the total is 1 + number of I vertices with at least one cross edge.
        """
        edges = instance(n, k, p, seed)
        
        # Build adjacency for cross edges
        cross_edges = {}  # v -> set of u in C connected to v
        for u, v in edges:
            if u < k and v >= k:
                cross_edges.setdefault(v, set()).add(u)
            elif v < k and u >= k:
                cross_edges.setdefault(u, set()).add(v)
        
        cliques = []
        
        # Clique 1: the full C
        if k >= 2:
            cliques.append(list(range(k)))
        
        # For each v in I with cross edges, add clique {v} ∪ N(v)
        for v in range(k, n):
            if v in cross_edges and len(cross_edges[v]) > 0:
                clique = [v] + sorted(cross_edges[v])
                if len(clique) >= 2:
                    cliques.append(clique)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            # Fallback: return trivial valid answer if we're out of time
            # But this shouldn't happen for these small inputs
            break
        cliques = solve_split_graph(*spec)
        results.append(cliques)

    # Ensure we have 3 results
    while len(results) < 3:
        results.append([])

    # Convert to ensure pure Python types
    final_result = []
    for cliques in results:
        clean_cliques = []
        for clique in cliques:
            clean_cliques.append([int(x) for x in clique])
        final_result.append(clean_cliques)

    return final_result
# EVOLVE-BLOCK-END
