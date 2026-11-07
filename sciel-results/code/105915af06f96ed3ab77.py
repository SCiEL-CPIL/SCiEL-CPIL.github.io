import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 145

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed, deadline):
        rng = random.Random(seed)
        
        # Build neighbor sets for I vertices
        adj_I = [set() for _ in range(n - k)]
        for i in range(n - k):
            v = k + i
            for u in range(k):
                if rng.random() < p:
                    adj_I[i].add(u)
        
        # Track which K-edges are covered
        covered_K_edges = set()
        cliques = []
        
        # Process I-vertices
        i_indices = list(range(n - k))
        rng.shuffle(i_indices)
        
        for i in i_indices:
            v = k + i
            U = set(adj_I[i])
            if not U:
                continue
            
            # Find max subset S of U such that no edge in S is already covered
            S = max_independent_in_covered(U, covered_K_edges)
            
            if S:
                clique = [v] + list(S)
                cliques.append(clique)
                for u, w in combinations(sorted(S), 2):
                    covered_K_edges.add((min(u, w), max(u, w)))
            
            # Cover remaining neighbors
            remaining = U - S
            if remaining:
                rem_list = list(remaining)
                # Greedily pair up remaining vertices
                j = 0
                while j < len(rem_list):
                    u = rem_list[j]
                    # Try to find a partner w in remaining such that (u,w) not covered
                    partner = None
                    for w in rem_list[j+1:]:
                        e = (min(u, w), max(u, w))
                        if e not in covered_K_edges:
                            partner = w
                            break
                    
                    if partner is not None:
                        w = partner
                        clique = [v, u, w]
                        cliques.append(clique)
                        covered_K_edges.add((min(u, w), max(u, w)))
                        # Remove w from rem_list
                        rem_list.remove(w)
                    else:
                        # Find any K-vertex to pair with u
                        paired = False
                        for w in range(k):
                            if w == u:
                                continue
                            e = (min(u, w), max(u, w))
                            if e not in covered_K_edges:
                                clique = [v, u, w]
                                cliques.append(clique)
                                covered_K_edges.add(e)
                                paired = True
                                break
                        if not paired:
                            # All edges from u are covered, just make 2-clique
                            clique = [v, u]
                            cliques.append(clique)
                        rem_list.pop(j)
                        continue
                    j += 1
        
        # Cover remaining K-edges
        rem_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_K_edges:
                    rem_edges.append((u, w))
        
        if rem_edges:
            k_cliques = cover_k_edges_greedy(rem_edges, k, deadline)
            cliques.extend(k_cliques)
        
        return cliques

    def max_independent_in_covered(verts, covered_edges):
        if not verts:
            return set()
        
        conflict = {v: set() for v in verts}
        for v in verts:
            for w in verts:
                if v != w:
                    e = (min(v, w), max(v, w))
                    if e in covered_edges:
                        conflict[v].add(w)
        
        IS = set()
        cand = set(verts)
        
        while cand:
            best_v = min(cand, key=lambda v: len(conflict[v] & cand))
            IS.add(best_v)
            cand = cand - conflict[best_v] - {best_v}
        
        return IS

    def cover_k_edges_greedy(rem_edges, k, deadline):
        if not rem_edges:
            return []
        
        adj = [set() for _ in range(k)]
        for u, w in rem_edges:
            adj[u].add(w)
            adj[w].add(u)
        
        remaining = set(rem_edges)
        cliques = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            verts = set()
            for u, w in remaining:
                verts.add(u)
                verts.add(w)
            
            C = greedy_max_clique(verts, adj)
            
            if not C:
                C = {next(iter(verts))}
            
            cliques.append(list(C))
            
            for u, w in combinations(sorted(C), 2):
                e = (min(u, w), max(u, w))
                if e in remaining:
                    remaining.remove(e)
                    adj[u].discard(w)
                    adj[w].discard(u)
        
        return cliques

    def greedy_max_clique(verts, adj):
        if not verts:
            return set()
        
        C = set()
        cand = set(verts)
        
        while cand:
            best_v = max(cand, key=lambda v: len(adj[v] & cand))
            C.add(best_v)
            cand = cand & adj[best_v]
        
        return C

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed, deadline)
        clean_cliques = []
        for c in cliques:
            clean_c = [int(v) for v in c]
            clean_cliques.append(clean_c)
        results.append(clean_cliques)

    while len(results) < 3:
        results.append([])

    return results
