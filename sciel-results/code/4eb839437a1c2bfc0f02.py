"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 110.0

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
        edges = instance(n, k, p, seed)
        
        # Build the bipartite graph of cross edges
        # Left side: clique vertices [0, k)
        # Right side: independent set vertices [k, n)
        # We want to partition the cross edges into as few cliques as possible.
        # A clique in the bipartite graph is a complete bipartite subgraph K_{a,b}.
        # So we are looking for a partition of the edge set into complete bipartite subgraphs.
        # This is equivalent to finding a low-rank decomposition of the adjacency matrix,
        # but with the constraint that the decomposition is exact (no overlaps, no gaps)
        # and each "rank 1" term corresponds to a complete bipartite subgraph.
        
        # A simple greedy approach:
        # 1. While there are edges left:
        #    a. Find the largest complete bipartite subgraph (biclique) in the remaining graph.
        #    b. Add it to the partition.
        #    c. Remove its edges from the remaining graph.
        
        # Finding the maximum biclique is NP-hard, but for these small graphs (n <= 90, k <= 30)
        # we can use a heuristic.
        
        # Let's represent the remaining edges as a set of (u, v) where u < k and v >= k.
        remaining_edges = set()
        for (u, v) in edges:
            if u < k and v >= k:
                remaining_edges.add((u, v))
            elif u >= k and v < k:
                remaining_edges.add((v, u))
        
        cliques = []
        
        # First, add the big clique of the clique part
        cliques.append(list(range(k)))
        
        # Now, partition the cross edges
        while remaining_edges:
            # Find a maximal biclique
            # Pick an edge (u, v)
            u, v = next(iter(remaining_edges))
            
            # Find all u' that are connected to v (in remaining_edges)
            # and all v' that are connected to u (in remaining_edges)
            us = set()
            vs = set()
            for (uu, vv) in remaining_edges:
                if uu == u:
                    vs.add(vv)
                if vv == v:
                    us.add(uu)
            
            # Now, we want to find the largest subset U' of us and V' of vs
            # such that U' x V' is a subset of remaining_edges.
            # This is a bipartite clique problem.
            # For small graphs, we can try to maximize the size.
            
            # Let's start with U' = us and V' = vs, and remove vertices that break the biclique property.
            U_prime = set(us)
            V_prime = set(vs)
            
            changed = True
            while changed:
                changed = False
                for u_prime in list(U_prime):
                    for v_prime in V_prime:
                        if (u_prime, v_prime) not in remaining_edges:
                            # Remove either u_prime or v_prime
                            # Heuristic: remove the one with fewer connections
                            # Count connections of u_prime to V_prime
                            count_u = sum(1 for v_p in V_prime if (u_prime, v_p) in remaining_edges)
                            # Count connections of v_prime to U_prime
                            count_v = sum(1 for u_p in U_prime if (u_p, v_prime) in remaining_edges)
                            
                            if count_u <= count_v:
                                U_prime.remove(u_prime)
                            else:
                                V_prime.remove(v_prime)
                            changed = True
                            break
                    if changed:
                        break
            
            if not U_prime or not V_prime:
                # This shouldn't happen if we started with an edge
                continue
            
            # The biclique is U_prime x V_prime
            # Add it as a clique: list(U_prime) + list(V_prime)
            clique = list(U_prime) + list(V_prime)
            cliques.append(clique)
            
            # Remove all edges in U_prime x V_prime from remaining_edges
            for u_prime in U_prime:
                for v_prime in V_prime:
                    remaining_edges.discard((u_prime, v_prime))
        
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        # Ensure every clique is a list of distinct Python ints, len >= 2
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        result.append(cleaned)

    # If deadline exceeded before finishing all three, return what we have
    # (but we must return three lists; if incomplete, the result is invalid,
    #  so we rely on the fast computation finishing well before the deadline)
    return result
# EVOLVE-BLOCK-END
