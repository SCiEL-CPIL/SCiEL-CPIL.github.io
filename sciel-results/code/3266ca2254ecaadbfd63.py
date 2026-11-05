"""Improved program: Greedy Set Cover on maximal cliques with local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return adj, edges

    def bron_kerbosch(adj, n):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            if P:
                u = max(P, key=lambda v: len(adj[v] & P))
            elif X:
                u = max(X, key=lambda v: len(adj[v] & X))
            else:
                return
            
            # Branch on vertices in P \\ N(u)
            for v in list(P - adj[u]):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        bk(frozenset(), set(range(n)), set())
        return cliques

    def greedy_set_cover(cliques, n):
        """Greedy set cover with 2-for-1 improvement: repeatedly pick vertex covering most uncovered cliques."""
        if not cliques:
            return set()
        
        # Map each clique to its index for tracking
        uncovered = set(range(len(cliques)))
        # For each vertex, compute which cliques it covers
        vertex_cliques = {v: set() for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].add(i)
        
        selected = set()
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                count = len(vertex_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count == 0:
                break
            
            selected.add(best_v)
            uncovered -= vertex_cliques[best_v]
        
        # 2-for-1 improvement: try to replace two selected vertices with one that covers all their cliques
        improved = True
        while improved:
            improved = False
            for i, v1 in enumerate(list(selected)):
                for v2 in list(selected):
                    if v1 == v2:
                        continue
                    # Find cliques covered by v1 or v2
                    v1_cliques = vertex_cliques[v1]
                    v2_cliques = vertex_cliques[v2]
                    union_cliques = v1_cliques | v2_cliques
                    
                    # Try to find a single vertex that covers all cliques in union_cliques
                    for v3 in range(n):
                        if v3 in selected:
                            continue
                        if union_cliques <= vertex_cliques[v3]:
                            # v3 covers all cliques that v1 and v2 cover
                            selected.discard(v1)
                            selected.discard(v2)
                            selected.add(v3)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return selected

    def local_search(selected, cliques, n, time_limit=5.0):
        """Try to improve the solution by swapping vertices."""
        if not cliques:
            return selected
        
        selected = set(selected)
        
        # Precompute for each vertex, which cliques it covers
        vertex_cliques = {v: set() for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].add(i)
        
        # Current coverage
        covered = set()
        for v in selected:
            covered |= vertex_cliques[v]
        
        # Check if all cliques are covered
        all_cliques = set(range(len(cliques)))
        
        def is_valid(sel):
            cov = set()
            for v in sel:
                cov |= vertex_cliques[v]
            return all_cliques <= cov
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removing each vertex
            for v in list(selected):
                if len(selected) <= 1:
                    break
                new_sel = selected - {v}
                if is_valid(new_sel):
                    selected = new_sel
                    improved = True
                    break
            
            # Try swapping: remove one, add another
            if not improved:
                for v_remove in list(selected):
                    if len(selected) <= 1:
                        break
                    for v_add in range(n):
                        if v_add in selected:
                            continue
                        new_sel = (selected - {v_remove}) | {v_add}
                        if len(new_sel) < len(selected):
                            if is_valid(new_sel):
                                selected = new_sel
                                improved = True
                                break
                    if improved:
                        break
        
        return selected

    def solve_instance(n, p, seed):
        adj, edges = instance(n, p, seed)
        
        # Find all maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        # Greedy set cover
        selected = greedy_set_cover(cliques, n)
        
        # Local search to improve
        remaining_time = deadline - time.time()
        if remaining_time > 2.0:
            ls_time = min(remaining_time - 1.0, 8.0)
            selected = local_search(selected, cliques, n, time_limit=ls_time)
        
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5.0:
            # Not enough time; return empty lists as fallback (will be invalid but avoid crash)
            results.append([])
            continue
        result = solve_instance(n, p, seed)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
