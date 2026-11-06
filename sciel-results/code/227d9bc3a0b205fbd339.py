import time
from random import Random

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        # Generate edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection
            if P:
                # Choose pivot u from P union X with max |P \cap N(u)|
                # For simplicity, just pick first, or iterate
                # Better: pick u that maximizes |P & N(u)|
                best_u = None
                max_count = -1
                for u in P.union(X):
                    count = len(P & adj[u])
                    if count > max_count:
                        max_count = count
                        best_u = u
                
                if best_u is not None:
                    # Iterate over P \ N(best_u)
                    for v in list(P - adj[best_u]):
                        P.remove(v)
                        X.remove(v) if v in X else None
                        bk(R | {v}, P & adj[v], X & adj[v])
                        P.add(v)
                        X.add(v)
            else:
                # If P is empty, loop is skipped, but we handled base case
                pass

        bk(set(), set(range(n)), set())
        return cliques

    def greedy_hitting_set(cliques, n):
        remaining = list(cliques)
        solution = set()
        while remaining:
            # Count frequency of each vertex in remaining cliques
            counts = {v: 0 for v in range(n)}
            for c in remaining:
                for v in c:
                    counts[v] += 1
            
            # Pick vertex with max count
            best_v = max(counts, key=counts.get)
            solution.add(best_v)
            
            # Remove cliques containing best_v
            remaining = [c for c in remaining if best_v not in c]
        
        return solution

    def min_hitting_set(cliques, n, time_limit):
        if not cliques:
            return set()
            
        start_time = time.time()
        best_solution = greedy_hitting_set(cliques, n)
        best_size = len(best_solution)
        
        # Sort cliques by size for B&B
        cliques_sorted = sorted(cliques, key=len)
        
        def lower_bound(remaining):
            # Greedy disjoint set approximation
            if not remaining:
                return 0
            # If time is tight, return 0
            if time.time() - start_time > time_limit - 0.5:
                return 0
            
            rem_set = list(remaining)
            count = 0
            while rem_set:
                # Pick smallest
                idx = 0
                min_size = len(rem_set[0])
                for i in range(1, len(rem_set)):
                    if len(rem_set[i]) < min_size:
                        min_size = len(rem_set[i])
                        idx = i
                c = rem_set.pop(idx)
                count += 1
                # Remove intersecting
                new_rem = []
                for other in rem_set:
                    if not c.intersection(other):
                        new_rem.append(other)
                rem_set = new_rem
            return count

        def dfs(selected, remaining):
            nonlocal best_solution, best_size
            
            if time.time() - start_time > time_limit:
                return
                
            if not remaining:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
                return
                
            if len(selected) >= best_size:
                return
                
            if len(selected) + lower_bound(remaining) >= best_size:
                return
            
            # Branch on the smallest remaining clique
            # Find smallest
            min_clique = min(remaining, key=len)
            
            for v in min_clique:
                new_selected = selected | {v}
                new_remaining = [c for c in remaining if v not in c]
                dfs(new_selected, new_remaining)

        dfs(set(), cliques_sorted)
        
        return best_solution

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        adj = build_adj(edges, n)
        cliques = bron_kerbosch(adj, n)
        
        # Filter size >= 2 (bron_kerbosch already does, but safe)
        cliques = [c for c in cliques if len(c) >= 2]
        
        sol = min_hitting_set(cliques, n, time_limit=35)
        results.append(sorted(sol))
        
    return results
