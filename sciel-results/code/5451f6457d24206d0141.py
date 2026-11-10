# EVOLVE-BLOCK-START
import random

def construct():
    import sys
    from itertools import combinations

    def build_graph(n, p, seed):
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
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            pivot_pool = P | X
            if not pivot_pool:
                return
            pivot = max(pivot_pool, key=lambda v: len(P & adj[v]))
            candidates = P - adj[pivot]
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        expand(set(), set(range(n)), set())
        return maximal_cliques

    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        if not cliques:
            return []
        
        universe = set()
        for clique in cliques:
            universe.update(clique)
            
        # Precompute vertex to cliques mapping
        vertex_to_cliques = {v: set() for v in universe}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(i)
        
        def greedy_randomized(cliques, universe, vertex_to_cliques, rng):
            uncovered = set(range(len(cliques)))
            selected = set()
            
            while uncovered:
                best_v = None
                best_count = -1
                candidates = []
                
                for v in universe:
                    if v in selected:
                        continue
                    count = len(vertex_to_cliques[v] & uncovered)
                    if count > best_count:
                        best_count = count
                        candidates = [v]
                    elif count == best_count and count > 0:
                        candidates.append(v)
                
                if not candidates or best_count == 0:
                    break
                    
                # Randomly pick from best candidates
                v = rng.choice(candidates)
                selected.add(v)
                uncovered -= vertex_to_cliques[v]
                
            return selected

        def local_search_reduce(selected, cliques, vertex_to_cliques):
            # Remove redundant vertices
            improved = True
            while improved:
                improved = False
                for v in list(selected):
                    # Check if v is still needed
                    needed = False
                    for i in vertex_to_cliques[v]:
                        clique = cliques[i]
                        other_covered = False
                        for u in clique:
                            if u != v and u in selected:
                                other_covered = True
                                break
                        if not other_covered:
                            needed = True
                            break
                    if not needed:
                        selected.remove(v)
                        improved = True
            return selected

        best_solution = None
        
        # Run multiple randomized greedy instances
        rng = random.Random(seed + 1000)
        for _ in range(100):
            sol = greedy_randomized(cliques, universe, vertex_to_cliques, rng)
            sol = local_search_reduce(sol, cliques, vertex_to_cliques)
            
            if best_solution is None or len(sol) < len(best_solution):
                best_solution = sol
                
        return sorted(best_solution)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        solution = solve_instance(n, p, seed)
        out.append(solution)
        
    return out
# EVOLVE-BLOCK-END
