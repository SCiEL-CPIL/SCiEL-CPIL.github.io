"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        """Partition edges of a split graph into minimum cliques."""
        edges = instance(n, k, p, seed)
        
        # Get neighbor sets for independent vertices
        I_neighbors = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if (u, v) in edges:
                    nbrs.add(u)
            I_neighbors[v] = nbrs
        
        cliques = []
        covered_edges = set()
        
        # Phase 1: For each I vertex with neighbors, create clique {v} ∪ N(v)
        for v in range(k, n):
            nbrs = I_neighbors[v]
            if len(nbrs) >= 1:
                clique = [v] + sorted(nbrs)
                cliques.append(clique)
                # Mark all edges within this clique as covered
                clique_set = set(clique)
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, w = clique[i], clique[j]
                        covered_edges.add((min(u, w), max(u, w)))
        
        # Phase 2: Cover remaining C-C edges
        remaining = set()
        for e in edges:
            u, v = e
            if u < k and v < k and e not in covered_edges:
                remaining.add(e)
        
        if remaining:
            # Greedy maximal clique extraction
            rem = set(remaining)
            while rem:
                # Pick an edge, build maximal clique
                e0 = next(iter(rem))
                clique_set = {e0[0], e0[1]}
                # Expand greedily
                changed = True
                while changed:
                    changed = False
                    for v in range(k):
                        if v not in clique_set:
                            if all((min(v, u), max(v, u)) in rem for u in clique_set):
                                clique_set.add(v)
                                changed = True
                # Remove all edges within clique_set from rem
                for i, u in enumerate(sorted(clique_set)):
                    for w in sorted(clique_set)[i + 1:]:
                        rem.discard((u, w))
                if len(clique_set) >= 2:
                    cliques.append(sorted(clique_set))
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback: edges as 2-cliques
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            n, k, p, seed = spec
            cliques = partition_cliques(n, k, p, seed)
            cleaned = []
            for c in cliques:
                cleaned.append([int(v) for v in c])
            results.append(cleaned)
    
    return results
# EVOLVE-BLOCK-END
