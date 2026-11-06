"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import random
import time
from itertools import combinations

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    
    def bron_kerbosch_maximal_cliques(adj, n):
        """
        Enumerate all maximal cliques using Bron-Kerbosch with pivot.
        adj: list of sets, adj[i] contains neighbors of i
        Returns list of frozensets (or lists) representing maximal cliques.
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                cliques.append(list(R))
                return
            
            # Pivot selection: choose vertex in P ∪ X with max |P ∩ adj[pivot]|
            # This reduces the number of recursive calls
            pivot_candidates = list(P | X)
            if not pivot_candidates:
                return
            
            # Find pivot with max intersection with P
            best_pivot = None
            best_count = -1
            for v in pivot_candidates:
                count = len(P & adj[v])
                if count > best_count:
                    best_count = count
                    best_pivot = v
            
            # Recurse on P \ N(pivot)
            for v in list(P - adj[best_pivot]):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.remove(v)
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques
    
    def greedy_set_cover(cliques, n):
        """
        Greedy algorithm for set cover: select vertices to hit all cliques.
        """
        # Convert cliques to sets for efficient operations
        clique_sets = [set(c) for c in cliques]
        
        # Filter out cliques of size 1 (we only need to hit cliques with >= 2 vertices)
        # The task says "meet every maximal clique with at least two vertices"
        # So we only need to hit cliques of size >= 2
        clique_sets = [c for c in clique_sets if len(c) >= 2]
        
        if not clique_sets:
            return []
        
        # Start with all cliques unhit
        unhit = list(clique_sets)
        selected = set()
        
        while unhit:
            # Find vertex that hits the most unhit cliques
            best_vertex = None
            best_count = 0
            best_indices = []
            
            # Count how many unhit cliques each vertex is in
            vertex_counts = {}
            for idx, clique in enumerate(unhit):
                for v in clique:
                    vertex_counts[v] = vertex_counts.get(v, 0) + 1
            
            for v, count in vertex_counts.items():
                if count > best_count:
                    best_count = count
                    best_vertex = v
            
            if best_vertex is None:
                break
            
            # Select this vertex
            selected.add(best_vertex)
            
            # Remove all cliques that contain this vertex
            new_unhit = []
            for clique in unhit:
                if best_vertex not in clique:
                    new_unhit.append(clique)
            unhit = new_unhit
        
        return sorted(selected)
    
    def local_search(initial_solution, cliques, n, max_iterations=1000):
        """
        Local search to improve the solution by trying to remove vertices
        or swap vertices.
        """
        solution = set(initial_solution)
        clique_sets = [set(c) for c in cliques if len(c) >= 2]
        
        if not clique_sets:
            return []
        
        def is_valid(sol):
            return all(any(v in sol for v in clique) for clique in clique_sets)
        
        # Try to remove vertices
        improved = True
        while improved:
            improved = False
            for v in list(solution):
                trial = solution - {v}
                if is_valid(trial):
                    solution = trial
                    improved = True
        
        # Try swaps: remove one, add another
        for v in list(solution):
            for w in range(n):
                if w in solution:
                    continue
                trial = (solution - {v}) | {w}
                if is_valid(trial) and len(trial) < len(solution):
                    solution = trial
                    improved = True
                    break
            if improved:
                break
        
        return sorted(solution)
    
    def solve_instance(n, p, seed, time_limit=30):
        edges = instance(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Enumerate maximal cliques
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        
        # Greedy set cover
        solution = greedy_set_cover(cliques, n)
        
        # Local search improvement
        solution = local_search(solution, cliques, n)
        
        return solution
    
    out = []
    start_time = time.time()
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            # Fallback: just use a heuristic
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            out.append(sorted(touched))
        else:
            sol = solve_instance(n, p, seed, remaining)
            out.append(sol)
    
    return out
# EVOLVE-BLOCK-END
