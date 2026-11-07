"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        """Find minimum edge-clique partition for a split graph."""
        # Generate cross edges
        rng = Random(seed)
        cross_nbrs = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            cross_nbrs[v] = nbrs
        
        cliques = []
        covered = set()
        
        # Step 1: For each independent vertex with edges, create clique {v} ∪ N(v)
        for v in range(k, n):
            nbrs = sorted(cross_nbrs[v])
            if len(nbrs) >= 1:
                clique = [v] + nbrs
                cliques.append(clique)
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        covered.add((clique[i], clique[j]))
        
        # Step 2: Cover remaining clique-clique edges
        remaining = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered:
                    remaining.add((u, w))
        
        # Build adjacency for remaining edges
        rem_adj = {u: set() for u in range(k)}
        for (u, w) in remaining:
            rem_adj[u].add(w)
            rem_adj[w].add(u)
        
        # Greedy: repeatedly find a maximal clique in remaining, add it, remove its edges
        while remaining:
            if time.time() - start_time > DEADLINE:
                for (u, w) in remaining:
                    cliques.append([u, w])
                break
            
            # Pick a vertex with max degree
            max_deg = -1
            best_v = -1
            for u in range(k):
                if len(rem_adj[u]) > max_deg:
                    max_deg = len(rem_adj[u])
                    best_v = u
            
            if best_v == -1:
                break
            
            # Build a clique starting from best_v
            clique = {best_v}
            # Try to add neighbors in order of degree
            candidates = sorted(rem_adj[best_v], key=lambda x: -len(rem_adj[x]))
            for c in candidates:
                # Check if c is adjacent to all vertices in current clique
                ok = True
                for v2 in clique:
                    if v2 == best_v:
                        continue
                    if (min(v2, c), max(v2, c)) not in remaining:
                        ok = False
                        break
                if ok:
                    clique.add(c)
            
            if len(clique) < 2:
                # Take best_v and one of its neighbors
                for c in rem_adj[best_v]:
                    clique.add(c)
                    break
            
            clique_list = sorted(clique)
            cliques.append(clique_list)
            
            # Remove all edges within this clique from remaining
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    remaining.discard((clique_list[i], clique_list[j]))
                    rem_adj[clique_list[i]].discard(clique_list[j])
                    rem_adj[clique_list[j]].discard(clique_list[i])
        
        return cliques

    # Solve all three instances
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() - start_time > DEADLINE:
            break
        cliques = solve_split(*spec)
        results.append(cliques)
    
    # Ensure all are Python lists of lists of Python ints
    final_result = []
    for cliques in results:
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        final_result.append(final_cliques)
    
    return final_result
# EVOLVE-BLOCK-END
