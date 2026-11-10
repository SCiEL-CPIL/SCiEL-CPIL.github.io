"""Rewritten: structural clique partition for split graphs (one clique per independent vertex + cover leftover intra edges)."""
# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition(n, k, p, seed):
        rng = Random(seed)
        # Generate neighbors for independent vertices
        neighbors = []
        for v in range(k, n):
            sv = []
            for u in range(k):
                if rng.random() < p:
                    sv.append(u)
            neighbors.append((v, sv))

        cliques = []
        covered_intra = set()
        
        # Process independent vertices
        for v, sv in neighbors:
            if not sv:
                continue
            
            # Check for conflict with already covered intra edges
            conflict = False
            sv_set = set(sv)
            for i in range(len(sv)):
                for j in range(i + 1, len(sv)):
                    a, b = sv[i], sv[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) in covered_intra:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                # Use a large clique {v} U sv
                clique = [v] + sv
                cliques.append(clique)
                # Mark intra edges as covered
                for i in range(len(sv)):
                    for j in range(i + 1, len(sv)):
                        a, b = sv[i], sv[j]
                        if a > b:
                            a, b = b, a
                        covered_intra.add((a, b))
            else:
                # Use star cliques {v, u} for each u in sv
                for u in sv:
                    cliques.append([v, u])
        
        # Cover remaining intra edges in K
        for a in range(k):
            for b in range(a + 1, k):
                if (a, b) not in covered_intra:
                    cliques.append([a, b])
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            result.append([list(e) for e in edges])
        else:
            result.append(partition(*spec))

    final = []
    for cliques in result:
        cl = []
        for c in cliques:
            cl.append([int(x) for x in c])
        final.append(cl)
    return final
# EVOLVE-BLOCK-END
