"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def bipartite_max_matching(adj, left_nodes):
        """
        Find maximum matching in bipartite graph using Kuhn's algorithm.
        adj: dict mapping left node -> list of right nodes it connects to
        left_nodes: list of left nodes
        Returns: dict mapping left node -> matched right node (or None)
        """
        match_r = {}  # right node -> left node
        match_l = {}  # left node -> right node
        
        def dfs(u, visited):
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    if v not in match_r or dfs(match_r[v], visited):
                        match_r[v] = u
                        match_l[u] = v
                        return True
            return False
        
        for u in left_nodes:
            visited = set()
            dfs(u, visited)
        
        return match_l

    def clique_partition(n, k, p, seed):
        """
        Find a clique partition for a split graph using bipartite matching.
        
        Strategy:
        - Type-A cliques: {u, v} where u,v in K, covering C-C edge (u,v)
        - Type-B cliques: {u, v, w} where u,v in K, w in I, covering C-C edge (u,v) 
          and cross-edges (u,w) and (v,w)
        - Type-C cliques: {u, w} where u in K, w in I, covering cross-edge (u,w)
        
        We want to maximize Type-B cliques (they cover 3 edges with 1 clique) 
        and minimize total cliques.
        
        Formulation:
        - Left side: C-C edges (u,v) where u,v in K
        - Right side: I-vertices w
        - Edge in bipartite graph: C-C edge (u,v) connects to I-vertex w 
          if both (u,w) and (v,w) are cross-edges
        
        Each match creates a Type-B clique {u,v,w} covering:
        - C-C edge (u,v)
        - Cross-edge (u,w)  
        - Cross-edge (v,w)
        
        After matching:
        - Unmatched C-C edges get Type-A cliques
        - Cross-edges not covered by Type-B cliques get Type-C cliques
        """
        # Generate cross-edges
        rng = Random(seed)
        cross_edges = set()
        # cross_edges[u] = set of w in I such that (u,w) is an edge
        cross_adj = defaultdict(set)  # u in K -> set of w in I
        for w in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.add((u, w))
                    cross_adj[u].add(w)
        
        # C-C edges: all pairs (u,v) with u < v, both in K
        cc_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        
        # Build bipartite graph
        # Left: C-C edges indexed by (u,v)
        # Right: I-vertices
        # Edge exists if w is adjacent to both u and v
        left_nodes = list(range(len(cc_edges)))
        adj = defaultdict(list)
        
        for i, (u, v) in enumerate(cc_edges):
            # Find I-vertices adjacent to both u and v
            common = cross_adj[u] & cross_adj[v]
            for w in common:
                adj[i].append(w)
        
        # Find maximum matching
        match_l = bipartite_max_matching(adj, left_nodes)
        
        # Build cliques
        cliques = []
        
        # Track which cross-edges are covered by Type-B cliques
        covered_cross = set()
        
        # Type-B cliques from matching
        for i, w in match_l.items():
            u, v = cc_edges[i]
            cliques.append([u, v, w])
            covered_cross.add((u, w))
            covered_cross.add((v, w))
        
        # Type-A cliques for unmatched C-C edges
        matched_left = set(match_l.keys())
        for i, (u, v) in enumerate(cc_edges):
            if i not in matched_left:
                cliques.append([u, v])
        
        # Type-C cliques for uncovered cross-edges
        for (u, w) in cross_edges:
            if (u, w) not in covered_cross:
                cliques.append([u, w])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        results.append(cliques)

    # Ensure we have exactly 3 results
    while len(results) < 3:
        results.append([])

    return results
# EVOLVE-BLOCK-END
