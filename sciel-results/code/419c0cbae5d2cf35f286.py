"""Clique partition of split graphs using structural properties."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        """
        Strategy using the structure of split graphs:
        
        1. The clique part C = {0, ..., k-1} forms a complete subgraph.
           We create one clique containing all C vertices: covers all C-C edges.
        
        2. For cross edges (between C and I), we use a matching-based approach:
           - Model cross edges as a bipartite graph between C and I.
           - Find a maximum matching to pair up cross edges efficiently.
           - Each matched pair (c1-i1, c2-i2) where c1≠c2 and i1≠i2 
             can be covered by clique {c1, c2, i1, i2} IF i1-i2 is NOT an edge 
             (which is true since I is independent) AND all cross edges present.
           - Actually, simpler: for each I-vertex i, create ONE clique 
             containing i and ALL its C-neighbors. Since any subset of C is 
             a clique and I is independent, this is valid.
           - But this might over-cover: if we use one clique per I-vertex, 
             the cross edges are covered, but we need to ensure C-C edges 
             are only covered once.
        
        Better approach:
        - One clique for all C vertices: covers all C-C edges exactly once.
        - For cross edges: group them by I-vertex. For each I-vertex i, 
          create a clique {i} ∪ N(i) where N(i) is the set of C-neighbors of i.
          This clique is valid because N(i) ⊆ C is a clique and i is 
          adjacent to all of N(i).
          Each cross edge (c, i) is covered exactly by the clique for i.
          The C-C edges within N(i) are ALSO covered by this clique, which 
          is a problem since they're already covered by the main C clique.
        
        Correction: We cannot have overlapping edge coverage. Each edge must 
        be in EXACTLY one clique.
        
        Revised strategy:
        - Option A: One big clique for all C, then for each I-vertex i, 
          create 2-cliques {c, i} for each c ∈ N(i). This gives 1 + |E_cross| cliques.
          But this is what the current program does.
          
        - Option B: Don't use one big C clique. Instead, partition C into 
          smaller cliques and use I-vertices to "extend" them.
          For each I-vertex i, create clique {i} ∪ S_i where S_i ⊆ C and 
          S_i is a clique (any subset works) and all vertices in S_i are 
          adjacent to i.
          Then cover remaining C-C edges with C-only cliques.
          The key is to minimize total cliques.
        
        - Option C: Use the bipartite matching idea from the directive.
          For cross edges, find a matching M. For each matched edge (c, i), 
          we can't just use {c, i} because that's the same as option A.
          Instead, note that if we have two cross edges (c1, i1) and (c2, i2) 
          with c1≠c2 and i1≠i2, then {c1, c2, i1, i2} is a clique (since 
          c1-c2 is an edge, i1-i2 is not an edge... wait, I is independent 
          so i1-i2 is NOT an edge. So {c1, c2, i1, i2} is NOT a clique 
          because i1 and i2 are not adjacent.
          
        So the 4-vertex set is not a clique. The best we can do per I-vertex 
        is one clique containing i and all its C-neighbors, but that over-covers 
        C-C edges.
        
        Final approach (simple and valid):
        - One clique for all C vertices.
        - For each cross edge (c, i), a 2-clique {c, i}.
        Total: 1 + |E_cross| cliques.
        """
        cliques = []
        
        # 1. One clique for the entire C part
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. 2-cliques for each cross edge
        for u, v in edges:
            if (u < k and v >= k) or (u >= k and v < k):
                cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        cleaned_cliques = []
        for c in cliques:
            cleaned_cliques.append([int(x) for x in c])
        results.append(cleaned_cliques)
    
    return results
# EVOLVE-BLOCK-END
