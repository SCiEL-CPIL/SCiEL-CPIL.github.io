"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
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
            # Pivot selection: choose vertex in P∪X with max neighbors in P
            pivot_candidates = list(P | X)
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda v: len(P & adj[v]))
                candidates = [v for v in P if v not in adj[pivot]]
            else:
                candidates = list(P)
            
            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        expand(frozenset(), frozenset(range(n)), frozenset())
        return maximal_cliques
    
    def greedy_set_cover(cliques, adj, n):
        """
        Greedy set cover with local search refinement.
        Each vertex 'covers' the cliques that contain it.
        """
        # Build vertex -> cliques mapping
        vertex_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].append(i)
        
        # Count how many times each clique is covered (initially 0)
        covered = [0] * len(cliques)
        selected = set()
        
        # Greedy: repeatedly pick vertex covering most uncovered cliques
        while True:
            # Check if all cliques are covered
            if all(c > 0 for c in covered):
                break
            
            best_v = None
            best_gain = 0
            for v in range(n):
                if v in selected:
                    continue
                gain = sum(1 for ci in vertex_cliques[v] if covered[ci] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v is None or best_gain == 0:
                break
            
            selected.add(best_v)
            for ci in vertex_cliques[best_v]:
                covered[ci] += 1
        
        # Local search: try to remove vertices that don't uncover any clique
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                # Check if removing v would uncover any clique
                would_uncover = False
                for ci in vertex_cliques[v]:
                    if covered[ci] == 1:
                        would_uncover = True
                        break
                if not would_uncover:
                    selected.remove(v)
                    for ci in vertex_cliques[v]:
                        covered[ci] -= 1
                    improved = True
        
        # Local search: try swapping one vertex for another
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for v in list(selected):
                # Find cliques only covered by v
                exclusive_cliques = [ci for ci in vertex_cliques[v] if covered[ci] == 1]
                if not exclusive_cliques:
                    continue
                
                # Try to find a vertex that covers all exclusive_cliques and is not in selected
                for w in range(n):
                    if w in selected:
                        continue
                    covers_all = all(w in cliques[ci] for ci in exclusive_cliques)
                    if covers_all:
                        # Check if w adds coverage to other cliques (bonus)
                        # Remove v, add w
                        selected.remove(v)
                        for ci in vertex_cliques[v]:
                            covered[ci] -= 1
                        selected.add(w)
                        for ci in vertex_cliques[w]:
                            covered[ci] += 1
                        improved = True
                        break
                if improved:
                    break
        
        return sorted(selected)
    
    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        if not cliques:
            return []
        
        # Greedy set cover
        result = greedy_set_cover(cliques, adj, n)
        
        # Verify the solution covers all cliques
        for clique in cliques:
            if not any(v in clique for v in result):
                # This shouldn't happen, but if it does, add a vertex from the clique
                for v in clique:
                    if v not in result:
                        result.append(v)
                        break
        
        return sorted(result)
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10.0:
            # Not enough time, return best effort (empty for remaining)
            results.append([])
            continue
        
        result = solve_instance(n, p, seed)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
