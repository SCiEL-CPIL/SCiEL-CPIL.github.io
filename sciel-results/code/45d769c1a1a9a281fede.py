"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    import random
    import sys
    from itertools import combinations
    
    DEADLINE = 140.0
    start_time = time.time()
    
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
            # Pivot selection
            if P:
                pivot = max(P, key=lambda v: len(P & adj[v]))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        expand(frozenset(), set(range(n)), set())
        return maximal_cliques
    
    def greedy_set_cover(cliques, n):
        """Greedy set cover: repeatedly pick the vertex that covers the most uncovered cliques."""
        uncovered = list(range(len(cliques)))
        covered_by = [set() for _ in range(n)]  # covered_by[v] = set of clique indices containing v
        
        for i, clique in enumerate(cliques):
            for v in clique:
                covered_by[v].add(i)
        
        solution = set()
        uncovered_set = set(uncovered)
        
        while uncovered_set:
            # Find vertex that covers most uncovered cliques
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in solution:
                    continue
                count = len(covered_by[v] & uncovered_set)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                break
            
            solution.add(best_v)
            uncovered_set -= covered_by[best_v]
        
        return solution
    
    def local_search_remove(solution, cliques, n, max_iterations=1000):
        """Try to remove vertices from the solution while maintaining coverage."""
        solution = set(solution)
        # Build coverage map: for each clique, which vertices in solution cover it
        clique_cover = {}
        for i, clique in enumerate(cliques):
            covers = [v for v in solution if v in clique]
            if covers:
                clique_cover[i] = set(covers)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            for v in list(solution):
                # Check if removing v still covers all cliques
                can_remove = True
                for i, covers in clique_cover.items():
                    if v in covers and len(covers) == 1:
                        can_remove = False
                        break
                if can_remove:
                    solution.remove(v)
                    for i, covers in clique_cover.items():
                        if v in covers:
                            covers.remove(v)
                    improved = True
        
        return solution
    
    def local_search_swap(solution, cliques, n, max_iterations=500):
        """Try to swap one vertex for another to maintain coverage with same or smaller size.
           Actually, for minimization, we just want to remove. Let's do a more sophisticated local search."""
        solution = set(solution)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            # Try removing each vertex
            for v in list(solution):
                # Check if v is essential (only vertex in solution covering some clique)
                essential = False
                for clique in cliques:
                    if v in clique:
                        covers_in_clique = [u for u in solution if u in clique]
                        if len(covers_in_clique) == 1 and covers_in_clique[0] == v:
                            essential = True
                            break
                if not essential:
                    solution.remove(v)
                    improved = True
        
        return solution
    
    def solve_instance(n, p, seed, time_budget):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        if not cliques:
            return []
        
        best_solution = None
        best_size = float('inf')
        
        # Multiple random restarts with different greedy orderings
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() - start_time > time_budget:
                break
            
            # Shuffle the order in which we consider vertices for greedy
            # We'll use a randomized greedy
            uncovered_set = set(range(len(cliques)))
            solution = set()
            
            # Precompute which cliques each vertex covers
            vertex_cliques = [set() for _ in range(n)]
            for i, clique in enumerate(cliques):
                for v in clique:
                    vertex_cliques[v].add(i)
            
            while uncovered_set:
                # Weighted greedy: pick vertex with highest weight * coverage
                # Use random weights to diversify
                if restart == 0:
                    # Standard greedy
                    best_v = -1
                    best_count = -1
                    for v in range(n):
                        if v in solution:
                            continue
                        count = len(vertex_cliques[v] & uncovered_set)
                        if count > best_count:
                            best_count = count
                            best_v = v
                else:
                    # Randomized greedy: add noise
                    best_v = -1
                    best_score = -float('inf')
                    for v in range(n):
                        if v in solution:
                            continue
                        count = len(vertex_cliques[v] & uncovered_set)
                        if count == 0:
                            continue
                        # Add random perturbation
                        noise = random.Random(restart * 1000 + v).random() * 0.5
                        score = count + noise
                        if score > best_score:
                            best_score = score
                            best_v = v
                
                if best_v == -1:
                    break
                
                solution.add(best_v)
                uncovered_set -= vertex_cliques[best_v]
            
            # Local search: remove redundant vertices
            solution = local_search_remove(solution, cliques, n)
            
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = set(solution)
        
        # Try additional local search passes on the best solution
        if best_solution is not None:
            # Multiple passes of local search with different orderings
            for _ in range(5):
                if time.time() - start_time > time_budget:
                    break
                best_solution = local_search_remove(best_solution, cliques, n)
        
        return sorted(best_solution) if best_solution is not None else []
    
    # Solve each instance
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    total_budget = DEADLINE - (time.time() - start_time)
    per_instance_budget = total_budget / len(instances) - 5  # leave 5s margin
    
    for n, p, seed in instances:
        if time.time() - start_time > DEADLINE - 10:
            # Not enough time, return what we have
            break
        sol = solve_instance(n, p, seed, per_instance_budget)
        results.append(sol)
    
    # If we didn't solve all instances, return empty lists for unsolved ones
    while len(results) < 3:
        results.append([])
    
    return results
# EVOLVE-BLOCK-END
