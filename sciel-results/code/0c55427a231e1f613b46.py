"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    
    def bron_kerbosch_maximal_cliques(adj, n):
        """Yield all maximal cliques using Bron-Kerbosch with pivot selection."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex in P ∪ X with maximum degree in P
            union_px = P | X
            if union_px:
                pivot = max(union_px, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                P.discard(v)
                X.add(v)
                
                bk(R, P_new, X_new)
                
                P.add(v)
                X.discard(v)
                R.discard(v)
        
        bk(set(), set(range(n)), set())
        return cliques
    
    def greedy_maximal_clique_cover(cliques, n):
        """
        Greedy set cover: select vertices to hit all maximal cliques (size >= 2).
        Use a priority-based approach: at each step, pick the vertex that covers
        the most currently-uncovered cliques, with tie-breaking by frequency.
        """
        # Only consider cliques of size >= 2
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []
        
        # Build mapping: vertex -> list of clique indices it belongs to
        vertex_to_cliques = defaultdict(list)
        for i, clique in enumerate(relevant_cliques):
            for v in clique:
                vertex_to_cliques[v].append(i)
        
        # Track which cliques are still uncovered
        uncovered = set(range(len(relevant_cliques)))
        
        selected = []
        
        while uncovered:
            # Find vertex that covers most uncovered cliques
            best_v = -1
            best_count = -1
            best_freq = -1
            
            for v in vertex_to_cliques:
                count = sum(1 for i in vertex_to_cliques[v] if i in uncovered)
                freq = len(vertex_to_cliques[v])
                if count > best_count or (count == best_count and freq > best_freq):
                    best_count = count
                    best_freq = freq
                    best_v = v
            
            if best_count <= 0:
                break
            
            selected.append(best_v)
            # Remove all cliques containing best_v from uncovered
            for i in vertex_to_cliques[best_v]:
                uncovered.discard(i)
        
        return sorted(selected)
    
    def improved_greedy_clique_cover(cliques, n):
        """
        More sophisticated approach: use LP relaxation or iterative improvement.
        Here we use a greedy with lookahead and then local search.
        """
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []
        
        # Start with greedy solution
        solution = set(greedy_maximal_clique_cover(cliques, n))
        
        # Local search: try to remove vertices if possible
        changed = True
        while changed:
            changed = False
            for v in list(solution):
                # Check if removing v still covers all cliques
                test_sol = solution - {v}
                covered = True
                for clique in relevant_cliques:
                    if not (clique & test_sol):
                        covered = False
                        break
                if covered:
                    solution = test_sol
                    changed = True
        
        # Local search: try to swap - remove one, add another if it helps
        # Try adding a vertex and then see if we can remove more
        for v in range(n):
            if v in solution:
                continue
            # Try adding v
            new_sol = solution | {v}
            # Try removing vertices
            improved = False
            for w in list(new_sol):
                if w == v:
                    continue
                test_sol = new_sol - {w}
                covered = True
                for clique in relevant_cliques:
                    if not (clique & test_sol):
                        covered = False
                        break
                if covered and len(test_sol) < len(solution):
                    solution = test_sol
                    improved = True
                    break
            if improved:
                break
        
        return sorted(solution)
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        
        # Find minimum vertex cover for cliques
        cover = improved_greedy_clique_cover(cliques, n)
        
        out.append(cover)
    
    return out
# EVOLVE-BLOCK-END
