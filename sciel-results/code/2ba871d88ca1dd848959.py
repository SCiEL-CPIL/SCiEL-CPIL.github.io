import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            P_X = P.union(X)
            if not P_X:
                return
            # Pivot: choose u in P union X with max degree in P
            u = max(P_X, key=lambda v: len(adj[v].intersection(P)))
            
            # Iterate over P \ N(u)
            for v in list(P - adj[u]):
                R_new = R.union({v})
                P_new = P.intersection(adj[v])
                X_new = X.intersection(adj[v])
                bk(R_new, P_new, X_new)
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(adj):
        cliques = bron_kerbosch_maximal_cliques(adj)
        if not cliques:
            return []
        
        n = len(adj)
        num_cliques = len(cliques)
        
        # Precompute which cliques each vertex is in
        # vertex_clique_indices[v] = list of clique indices containing v
        vertex_clique_indices = [[] for _ in range(n)]
        for i, c in enumerate(cliques):
            for v in c:
                vertex_clique_indices[v].append(i)
        
        # Greedy + Local Search
        best_S = None
        best_size = n + 1
        
        # Number of restarts depends on time
        # Estimate time per restart: 
        # Greedy: ~n * C * n operations? No, optimized.
        # Let's just run as many as possible within a sub-deadline.
        
        max_restarts = 1000 # Cap
        restarts_done = 0
        
        while time.time() - start_time < deadline - 5:
            # 1. Greedy Initialization
            S = set()
            uncovered = set(range(num_cliques))
            
            while uncovered:
                best_v = -1
                max_cov = -1
                for v in range(n):
                    if v in S:
                        continue
                    # Count how many uncovered cliques v covers
                    cov = 0
                    for c_idx in vertex_clique_indices[v]:
                        if c_idx in uncovered:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_v = v
                
                if best_v == -1 or max_cov == 0:
                    break
                
                S.add(best_v)
                # Update uncovered
                # Remove all cliques containing best_v
                for c_idx in vertex_clique_indices[best_v]:
                    if c_idx in uncovered:
                        uncovered.remove(c_idx)
            
            # 2. Local Search: Remove redundant vertices
            # Initialize cover counts
            cover_counts = [0] * num_cliques
            for v in S:
                for c_idx in vertex_clique_indices[v]:
                    cover_counts[c_idx] += 1
            
            changed = True
            # Shuffle S to remove vertices in random order
            S_list = list(S)
            random.shuffle(S_list)
            
            while changed:
                changed = False
                for v in S_list:
                    # Check if v can be removed
                    can_remove = True
                    for c_idx in vertex_clique_indices[v]:
                        if cover_counts[c_idx] == 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        # Remove v
                        S.remove(v)
                        for c_idx in vertex_clique_indices[v]:
                            cover_counts[c_idx] -= 1
                        changed = True
            
            # Try swaps to potentially escape local minima?
            # For now, just keep the best size found.
            if len(S) < best_size:
                best_size = len(S)
                best_S = set(S)
            
            restarts_done += 1
            if restarts_done >= max_restarts:
                break
            
            # Small sleep to avoid 100% CPU if too fast? No, let it run.
            
        return sorted(best_S) if best_S else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > deadline:
            # Fallback: simple greedy
            adj = random_graph(n, p, seed)
            cliques = bron_kerbosch_maximal_cliques(adj)
            S = set()
            uncovered = set(range(len(cliques)))
            # Quick greedy
            while uncovered:
                best_v = -1
                max_cov = -1
                for v in range(n):
                    if v in S: continue
                    cov = sum(1 for c in cliques if v in c and cliques.index(c) in uncovered) # Slow, but fallback
                    if cov > max_cov:
                        max_cov = cov
                        best_v = v
                if best_v == -1: break
                S.add(best_v)
                new_uncovered = set()
                for i, c in enumerate(cliques):
                    if i in uncovered and best_v not in c:
                        new_uncovered.add(i)
                uncovered = new_uncovered
            out.append(sorted(S))
            continue
        
        adj = random_graph(n, p, seed)
        res = solve_instance(adj)
        out.append(res)
        
    return out
