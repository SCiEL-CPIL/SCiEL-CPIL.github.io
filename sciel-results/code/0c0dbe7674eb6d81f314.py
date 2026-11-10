import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with margin
    
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def bron_kerbosch(adj):
        """Enumerate all maximal cliques of size >= 2."""
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Pivot selection
            if P:
                pivot = max(P, key=lambda v: len(adj[v] & P))
                candidates = [v for v in P if v not in adj[pivot]]
            else:
                candidates = []
            
            for v in candidates:
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(frozenset(), set(range(n)), set())
        return cliques
    
    def solve_hitting_set(vertices, cliques, time_limit):
        """Find minimum hitting set for the hypergraph."""
        n = len(vertices)
        if not cliques:
            return set()
        
        # Map vertex labels to indices 0..n-1
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        clique_indices = [tuple(sorted(vertex_to_idx[v] for v in c)) for c in cliques]
        
        # Precompute for each vertex, which cliques it hits
        vertex_cliques = [set() for _ in range(n)]
        for i, c in enumerate(clique_indices):
            for v in c:
                vertex_cliques[v].add(i)
        
        # Greedy initial solution
        uncovered = set(range(len(clique_indices)))
        solution = set()
        while uncovered:
            best_v = None
            best_count = -1
            for v in range(n):
                if v in solution:
                    continue
                count = len(vertex_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None:
                break
            solution.add(best_v)
            uncovered -= vertex_cliques[best_v]
        
        best_size = len(solution)
        best_solution = set(solution)
        
        # Branch and bound to try to improve
        # Sort vertices by frequency (descending)
        vertex_order = sorted(range(n), key=lambda v: len(vertex_cliques[v]), reverse=True)
        
        # State: (current_set, remaining_cliques, next_vertex_index)
        # Use DFS with pruning
        
        import sys
        sys.setrecursionlimit(10000)
        
        best = [best_size]
        best_sol = [best_solution]
        
        def dfs(current_set, remaining, v_idx, time_check):
            if time.time() > time_check:
                return
            if not remaining:
                if len(current_set) < best[0]:
                    best[0] = len(current_set)
                    best_sol[0] = set(current_set)
                return
            if v_idx >= len(vertex_order):
                return
            # Lower bound: need at least ceil(len(remaining) / max_coverage) more vertices
            # Simple bound: if len(current_set) + 1 >= best[0], we can still try (since we might find equal)
            # Actually, if len(current_set) >= best[0], prune
            if len(current_set) >= best[0]:
                return
            
            v = vertex_order[v_idx]
            
            # Option 1: Include v
            if vertex_cliques[v] & remaining:
                new_remaining = remaining - vertex_cliques[v]
                dfs(current_set | {v}, new_remaining, v_idx + 1, time_check)
            
            # Option 2: Exclude v
            # But we must ensure remaining cliques can still be covered
            dfs(current_set, remaining, v_idx + 1, time_check)
        
        time_check = min(deadline, time.time() + 30)  # 30 sec per instance
        dfs(set(), set(range(len(clique_indices))), 0, time_check)
        
        # Convert back to vertex labels
        result = {vertices[i] for i in best_sol[0]}
        return result
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            # Not enough time, return what we have or fallback
            break
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        vertices = list(range(n))
        hitting_set = solve_hitting_set(vertices, cliques, deadline - 5)
        results.append(sorted(hitting_set))
    
    # If we didn't finish all, fill with greedy fallback
    while len(results) < 3:
        n, p, seed = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))[len(results)]
        adj = build_graph(n, p, seed)
        # Fallback: all vertices in any edge
        touched = set()
        for u in range(n):
            for v in adj[u]:
                touched.add(u)
                touched.add(v)
        results.append(sorted(touched))
    
    return results
