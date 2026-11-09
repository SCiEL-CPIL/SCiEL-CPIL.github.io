"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve_instance(n, k, p, seed, deadline):
        # Generate cross adjacencies
        rng = Random(seed)
        cross_adj = {}
        for v in range(k, n):
            adj = frozenset(u for u in range(k) if rng.random() < p)
            cross_adj[v] = adj
        
        # All edges
        cc_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        ci_edges = set((u, v) for v in range(k, n) for u in cross_adj[v])
        remaining = cc_edges | ci_edges
        
        cliques = []
        rng2 = Random(seed + 42)
        
        # Precompute: for each I vertex, its neighborhood
        # Greedy edge cover
        for step in range(100000):
            if not remaining:
                break
            if time.time() > deadline:
                break
            
            edge = rng2.choice(list(remaining))
            u, w = edge
            if u > w:
                u, w = w, u
            
            if u < k and w < k:
                # C-C edge
                S = frozenset([u, w])
                T = frozenset(v for v in range(k, n) if S.issubset(cross_adj[v]))
                clique_set = set(S) | set(T)
                if len(clique_set) < 2:
                    continue
                covered = set()
                S_list = sorted(S)
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        e = (S_list[i], S_list[j])
                        if e in remaining:
                            covered.add(e)
                for v in T:
                    for u2 in S:
                        e = (u2, v)
                        if e in remaining:
                            covered.add(e)
                if covered:
                    cliques.append(sorted(clique_set))
                    remaining -= covered
            else:
                # C-I edge
                u2, w2 = u, w
                S = cross_adj[w2]
                if u2 not in S:
                    S = S | frozenset([u2])
                T = frozenset(v for v in range(k, n) if S.issubset(cross_adj[v]))
                clique_set = set(S) | set(T)
                if len(clique_set) < 2:
                    continue
                covered = set()
                S_list = sorted(S)
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        e = (S_list[i], S_list[j])
                        if e in remaining:
                            covered.add(e)
                for v in T:
                    for u2b in S:
                        e = (u2b, v)
                        if e in remaining:
                            covered.add(e)
                if covered:
                    cliques.append(sorted(clique_set))
                    remaining -= covered
        
        # Handle leftover
        if remaining:
            rem_cc = set(e for e in remaining if e[0] < k and e[1] < k)
            rem_ci = set(e for e in remaining if e[0] < k and e[1] >= k)
            if rem_cc:
                inv = set()
                for e in rem_cc:
                    inv.add(e[0]); inv.add(e[1])
                if len(inv) >= 2:
                    cliques.append(sorted(inv))
            if rem_ci:
                v2u = defaultdict(set)
                for e in rem_ci:
                    a, b = e
                    v2u[b].add(a)
                sig2v = defaultdict(set)
                for v, us in v2u.items():
                    sig2v[frozenset(us)].add(v)
                for sig, vs in sig2v.items():
                    c = sorted(set(sig) | set(vs))
                    if len(c) >= 2:
                        cliques.append(c)
        
        return cliques
    
    start = time.time()
    deadline = start + 100
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        n, k, p, seed = spec
        if time.time() > deadline - 5:
            cliques = [list(e) for e in instance(n, k, p, seed)]
        else:
            cliques = solve_instance(n, k, p, seed, deadline - 5)
        results.append(cliques)
    
    return [[list(c) for c in cliques] for cliques in results]

# EVOLVE-BLOCK-END
