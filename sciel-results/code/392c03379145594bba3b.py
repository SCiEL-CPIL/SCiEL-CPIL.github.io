import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_graph(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection
            if P:
                pivot = max(P, key=lambda v: len(P & adj[v]))
            elif X:
                pivot = max(X, key=lambda v: len(P & adj[v]))
            else:
                return
                
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.remove(v)
                P.remove(v)
                X.add(v)
                
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        adj = build_graph(n, edges)
        cliques = bron_kerbosch(adj)
        
        if not cliques:
            return []

        # Greedy Set Cover
        # Map vertex to list of indices of cliques it covers
        # Or just keep a list of uncovered cliques
        
        uncovered = list(range(len(cliques)))
        # count[v] = number of uncovered cliques containing v
        count = [0] * n
        for idx in uncovered:
            for v in cliques[idx]:
                count[v] += 1
                
        S = set()
        
        while uncovered:
            # Find vertex with max count
            best_v = -1
            max_count = -1
            for v in range(n):
                if count[v] > max_count:
                    max_count = count[v]
                    best_v = v
            
            if best_v == -1:
                break
                
            S.add(best_v)
            
            # Update uncovered
            new_uncovered = []
            for idx in uncovered:
                C = cliques[idx]
                if best_v in C:
                    # Covered
                    for v in C:
                        if v != best_v:
                            count[v] -= 1
                    # best_v count becomes 0 effectively as we remove it from consideration
                else:
                    new_uncovered.append(idx)
            uncovered = new_uncovered
            count[best_v] = 0 # Mark as used/removed from pool
            
        # Local Search: Remove redundant vertices
        # A vertex v in S is redundant if for all C in cliques, if v in C then (C \ {v}) & S is not empty.
        # Equivalently, v is essential if there exists C such that C & S == {v}.
        
        changed = True
        while changed:
            changed = False
            # Iterate over a copy since we might modify S
            for v in list(S):
                is_essential = False
                for C in cliques:
                    if v in C:
                        # Check if any other vertex in S is in C
                        found_other = False
                        for u in C:
                            if u != v and u in S:
                                found_other = True
                                break
                        if not found_other:
                            is_essential = True
                            break
                if not is_essential:
                    S.remove(v)
                    changed = True
                    
        return sorted(S)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return empty or greedy only? 
            # Better to just return what we have or empty if not started.
            # But we should finish the current one if possible.
            pass
        
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
