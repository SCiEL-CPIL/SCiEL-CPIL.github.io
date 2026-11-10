"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    _start = time.time()
    _DEADLINE = 130.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_partition(n, k, p, seed):
        rng = Random(seed)
        # Build adjacency
        C_adj = {}
        for u in range(k):
            C_adj[u] = set(range(u + 1, k))
        I_adj = {}
        for v in range(k, n):
            I_adj[v] = set()
            for u in range(k):
                if rng.random() < p:
                    I_adj[v].add(u)
        
        # All edges
        all_edges = set()
        for u in range(k):
            for v in C_adj[u]:
                all_edges.add((u, v))
        for v in range(k, n):
            for u in I_adj[v]:
                all_edges.add((u, v))
        
        # Greedy: for each I-vertex, assign a subset of its C-neighbors
        # to form a mixed clique. We want to cover as many C-edges as possible
        # within mixed cliques to reduce the number of pure C-cliques needed.
        
        # For each I-vertex v, the neighbors in C form a set N(v).
        # A mixed clique {v} ∪ S covers all edges within S (C-edges) 
        # and all edges (v, s) for s in S (cross edges).
        # Since S must be a clique in C (which it always is since C is a clique),
        # any subset S of N(v) works.
        
        # Strategy: For each I-vertex, pick S = N(v) (all neighbors) to maximize
        # C-edge coverage. But we need to ensure C-edges are covered exactly once.
        # If two I-vertices both claim the same C-edge, that's a problem.
        
        # Alternative: assign each C-edge to at most one I-vertex.
        # For I-vertex v, let S_v be the set of C-vertices in N(v) that we assign to v.
        # The mixed clique is {v} ∪ S_v. This covers C-edges within S_v and cross edges (v, s) for s in S_v.
        # Remaining cross edges (v, u) where u not in S_v must be covered by... 
        # Wait, they can't be in a pure C-clique. They must be in a mixed clique with v.
        # So actually ALL cross edges incident to v must be in the mixed clique for v.
        # That means S_v MUST = N(v). There's no choice!
        
        # So each I-vertex v has exactly one mixed clique: {v} ∪ N(v).
        # This covers all cross edges and all C-edges within N(v).
        # Remaining C-edges are those NOT within any N(v) for v in I.
        # We need to cover those with pure C-cliques.
        
        # Compute which C-edges are covered by mixed cliques
        covered_C_edges = set()
        mixed_cliques = []
        for v in range(k, n):
            S = I_adj[v]
            if len(S) >= 1:
                clique = [v] + sorted(S)
                mixed_cliques.append(clique)
                for a in S:
                    for b in S:
                        if a < b:
                            covered_C_edges.add((a, b))
        
        # Remaining C-edges
        remaining_C_edges = []
        for u in range(k):
            for v in C_adj[u]:
                if (u, v) not in covered_C_edges:
                    remaining_C_edges.append((u, v))
        
        # Cover remaining C-edges with minimum number of cliques
        # Since C is a clique, any subset is a clique. We need to partition
        # the remaining edges into minimum number of cliques.
        # This is equivalent to: find minimum number of subsets of C such that
        # each remaining edge is in exactly one subset, and each subset is a clique
        # (always true). So we just need to partition the edge set into minimum
        # cliques, which for a complete graph on the involved vertices means
        # we can group edges.
        
        # The remaining edges form a graph H on vertices C. We need to cover edges
        # of H with minimum number of cliques. Since H is a subgraph of K_k,
        # this is the clique cover number of H.
        
        # Greedy clique cover of H:
        H_edges = set(remaining_C_edges)
        H_vertices = set()
        for e in H_edges:
            H_vertices.add(e[0])
            H_vertices.add(e[1])
        
        # Build adjacency of H
        H_adj = {v: set() for v in H_vertices}
        for (a, b) in H_edges:
            H_adj[a].add(b)
            H_adj[b].add(a)
        
        # Greedy: repeatedly find a large clique in H, remove its edges
        pure_cliques = []
        remaining = set(H_edges)
        vertices = set(H_vertices)
        
        while remaining:
            # Find a greedy clique
            # Start with vertex of highest degree
            if not vertices:
                break
            start = max(vertices, key=lambda x: len(H_adj[x] & vertices))
            clique = {start}
            candidates = set(H_adj[start] & vertices)
            while candidates:
                # Pick vertex connected to all in clique
                valid = candidates & set.intersection(*[set(H_adj[c]) for c in clique]) if clique else candidates
                if not valid:
                    break
                v = max(valid, key=lambda x: len(H_adj[x] & candidates))
                clique.add(v)
                candidates = candidates & H_adj[v]
            
            # Remove all edges within clique
            clique_edges = set()
            clique_list = sorted(clique)
            for i, a in enumerate(clique_list):
                for b in clique_list[i+1:]:
                    if (a, b) in remaining or (b, a) in remaining:
                        clique_edges.add((min(a,b), max(a,b)))
            
            if clique_edges:
                pure_cliques.append(clique_list)
                remaining -= clique_edges
                # Update vertices
                for v in clique:
                    # Check if v still has edges
                    has_edges = False
                    for u in H_adj[v]:
                        if (min(v,u), max(v,u)) in remaining:
                            has_edges = True
                            break
                    if not has_edges:
                        vertices.discard(v)
            else:
                # Shouldn't happen, but break to avoid infinite loop
                break
        
        return mixed_cliques + pure_cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _DEADLINE:
            break
        part = build_partition(*spec)
        # Convert to lists of ints
        clean = []
        for c in part:
            clean.append([int(x) for x in c])
        result.append(clean)
    
    return result
# EVOLVE-BLOCK-END
