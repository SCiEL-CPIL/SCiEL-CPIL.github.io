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
        # Build adjacency for cross edges
        # For each independent vertex v (>= k), find its neighbors in clique part
        cross_adj = {}  # v -> list of u in [0, k) such that (u, v) is an edge
        for u, v in edges:
            if u < k and v >= k:
                if v not in cross_adj:
                    cross_adj[v] = []
                cross_adj[v].append(u)
        
        # Strategy:
        # 1. For each independent vertex v, create a clique {v} ∪ N_C(v) if |N_C(v)| >= 1
        #    This covers all cross edges incident to v.
        # 2. For edges within the clique part that are NOT covered by any step-1 clique,
        #    we need to cover them. An edge (u, w) within C is covered by step-1 clique of v
        #    iff both u and w are in N_C(v).
        #    So we need to cover C-edges where no independent vertex has both endpoints as neighbors.
        
        # First, determine which C-edges are covered by step-1 cliques
        covered_c_edges = set()
        for v in range(k, n):
            neighbors = cross_adj.get(v, [])
            if len(neighbors) >= 2:
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        u1, u2 = neighbors[i], neighbors[j]
                        if u1 > u2:
                            u1, u2 = u2, u1
                        covered_c_edges.add((u1, u2))
        
        # Now find C-edges not covered
        uncovered_c_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_c_edges:
                    uncovered_c_edges.append((u, w))
        
        # Build cliques
        cliques = []
        
        # Step 1: cliques for each independent vertex
        for v in range(k, n):
            neighbors = cross_adj.get(v, [])
            if len(neighbors) >= 1:
                clique = [v] + sorted(neighbors)
                cliques.append(clique)
            # If |N_C(v)| == 0, v is isolated, no clique needed for it
        
        # Step 2: cover uncovered C-edges
        # Simple approach: each uncovered C-edge becomes its own 2-clique
        # But we can do better: group uncovered C-edges into larger cliques
        # Since all C-vertices form a clique, any subset is a clique.
        # We just need to ensure each uncovered edge is in exactly one clique.
        # The simplest is: one big clique containing all C-vertices that participate
        # in at least one uncovered edge. But wait - that would cover ALL edges among
        # those vertices, including ones already covered by step-1 cliques.
        # That's not allowed - each edge must be in EXACTLY one clique.
        
        # So we need a clique partition of the uncovered C-edge subgraph.
        # Since the uncovered C-edges form a subgraph of a clique (complete graph on C),
        # we can just put all vertices involved in uncovered edges into one clique,
        # BUT that would cover edges among them that might already be covered.
        
        # Actually, re-reading the constraint: "every edge must lie in exactly one listed clique"
        # This means we cannot have an edge appearing in two cliques.
        # So if we put vertices {a, b, c} in a clique, edges (a,b), (b,c), (a,c) are all covered by that clique.
        # If (a,b) was already covered by a step-1 clique, we can't cover it again.
        
        # Therefore, for uncovered C-edges, we need a clique partition of that subgraph
        # where no edge in the clique partition was already covered.
        
        # Since the uncovered edges form a subgraph H of the complete graph on C,
        # and we need to partition E(H) into cliques (where each clique is a set of vertices
        # all mutually adjacent in H... wait, no. A clique in the partition means a set of vertices
        # such that all pairs are edges in H, and we're covering exactly the edges of H.)
        
        # Actually, for a clique partition of a graph H, we partition the edge set E(H) into
        # cliques, where each "clique" is a complete subgraph of H.
        
        # The simplest valid approach: each uncovered edge is its own 2-clique.
        # This is always valid and uses |uncovered_c_edges| cliques.
        
        # Can we do better? We can group edges into larger cliques if they form triangles, etc.
        # For now, let's use the simple approach and see if we can improve.
        
        for u, w in uncovered_c_edges:
            cliques.append([u, w])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        results.append(cliques)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return results
# EVOLVE-BLOCK-END
