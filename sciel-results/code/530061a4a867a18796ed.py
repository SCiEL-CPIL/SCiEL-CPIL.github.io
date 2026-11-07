"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        # Regenerate the graph to identify cross edges
        rng = Random(seed)
        # Generate cross edges (u < k <= v)
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append([u, v])
        
        # The k-clique: all first k vertices form one clique covering all internal edges
        clique_list = [list(range(k))]
        # Each cross edge is its own 2-clique (no two cross edges share a v, and 
        # grouping multiple u's with same v would require those u's to be mutually adjacent,
        # which they are, but v is not adjacent to other v's, so we can group u's per v)
        # Actually, for a fixed v, all u in 0..k-1 that connect to v form a clique {u1,u2,...,v}
        # since all u's are mutually adjacent. So we can group by v!
        
        # Let's redo: group cross edges by v
        from collections import defaultdict
        by_v = defaultdict(list)
        rng2 = Random(seed)
        # Regenerate to be safe - need to replay in same order
        for v in range(k, n):
            for u in range(k):
                if rng2.random() < p:
                    by_v[v].append(u)
        
        clique_list = [list(range(k))]
        for v in range(k, n):
            us = by_v[v]
            if len(us) >= 1:
                # {u1, u2, ..., v} is a clique since all u's are in the k-clique and v connects to all of them
                clique_list.append(sorted(us + [v]))
        
        results.append(clique_list)
    
    return results
# EVOLVE-BLOCK-END
