import time
from random import Random

def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin before 150s

    def build_graph(n, p, seed):
        rng = Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def bron_kerbosch_cliques(adj, n):
        """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Choose pivot u from P ∪ X to minimize branching
            if P:
                u = max(P, key=lambda v: bin((adj[v] & (P | X)).count() if False else 0))
                # Better: pick u that maximizes |N(u) ∩ (P∪X)|
                best_u = None
                best_cnt = -1
                for v in P:
                    cnt = bin(adj[v] & (P | X)).count('1')
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_u = v
                if best_u is None and X:
                    for v in X:
                        cnt = bin(adj[v] & (P | X)).count('1')
                        if cnt > best_cnt:
                            best_cnt = cnt
                            best_u = v
                if best_u is not None:
                    # Branch on P \ N(u)
                    candidates = [v for v in P if not (adj[best_u] >> v) & 1]
                else:
                    candidates = list(P)
            else:
                candidates = list(P)
            
            for v in candidates:
                bk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        
        # Enumerate all maximal cliques of size >= 2
        cliques = bron_kerbosch_cliques(adj, n)
        
        if not cliques:
            return []
        
        # Filter: only cliques of size >= 2 (already done in bk)
        # Greedy hitting set
        uncovered = set(range(len(cliques)))
        S = set()
        
        # Precompute for each vertex, which cliques it hits
        vertex_cliques = {v: [] for v in range(n)}
        for i, C in enumerate(cliques):
            for v in C:
                vertex_cliques[v].append(i)
        
        while uncovered:
            if time.time() > DEADLINE - 10:
                break
            # Pick vertex that hits most uncovered cliques
            best_v = -1
            best_cnt = -1
            for v in range(n):
                if v in S:
                    continue
                cnt = sum(1 for i in vertex_cliques[v] if i in uncovered)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_cnt <= 0:
                break
            S.add(best_v)
            # Remove covered cliques
            for i in vertex_cliques[best_v]:
                if i in uncovered:
                    uncovered.discard(i)
        
        # Local search: try to remove vertices from S
        improved = True
        while improved:
            improved = False
            # Recompute uncovered
            uncovered = set()
            for i, C in enumerate(cliques):
                if not (C & S):
                    uncovered.add(i)
            if not uncovered:
                break
            
            # Try removing each vertex in S
            for v in list(S):
                S_test = S - {v}
                # Check if still valid
                valid = True
                for i, C in enumerate(cliques):
                    if not (C & S_test):
                        valid = False
                        break
                if valid:
                    S = S_test
                    improved = True
                    break
        
        # Try adding vertices to potentially allow removals
        # Simpler: just return S
        return sorted(S)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE - 5:
            # Fallback: vertex cover via greedy
            adj = build_graph(n, p, seed)
            covered = set()
            cover = set()
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append((u, v))
            while edges:
                best_v = -1
                best_cnt = -1
                for v in range(n):
                    cnt = sum(1 for (a, b) in edges if a == v or b == v)
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_v = v
                if best_cnt <= 0:
                    break
                cover.add(best_v)
                edges = [(a, b) for (a, b) in edges if a != best_v and b != best_v]
            results.append(sorted(cover))
            continue
        
        result = solve_instance(n, p, seed)
        results.append(result)
    
    return results
