"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
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
            
            # Choose pivot to minimize branching
            if P:
                pivot_candidates = P
                max_neighbors = -1
                pivot = None
                for v in pivot_candidates:
                    cnt = len(adj[v] & P)
                    if cnt > max_neighbors:
                        max_neighbors = cnt
                        pivot = v
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
    
    def greedy_set_cover(cliques, vertices):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        # Map each clique to its set of vertices
        clique_to_vertices = {i: set(cliques[i]) for i in range(len(cliques))}
        
        # For each vertex, count how many cliques it belongs to
        vertex_coverage = {v: set() for v in vertices}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_coverage[v].add(i)
        
        uncovered = set(range(len(cliques)))
        selected = []
        
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_vertex = None
            best_count = -1
            for v in vertices:
                count = len(vertex_coverage[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_vertex = v
            
            if best_vertex is None or best_count == 0:
                break
            
            selected.append(best_vertex)
            uncovered -= vertex_coverage[best_vertex]
        
        return selected
    
    def local_search_improve(selected, cliques, vertices, max_iterations=500):
        """Local search: try to swap out a vertex for a better one, or remove redundant ones."""
        selected_set = set(selected)
        
        # Build coverage map
        def get_covered_cliques(sel):
            covered = set()
            for i, clique in enumerate(cliques):
                if clique & sel:
                    covered.add(i)
            return covered
        
        current_covered = get_covered_cliques(selected_set)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try removing a vertex if it doesn't uncover any clique
            for v in list(selected_set):
                trial = selected_set - {v}
                if len(trial) < len(selected_set):
                    new_covered = get_covered_cliques(trial)
                    if new_covered == current_covered:
                        selected_set = trial
                        current_covered = new_covered
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping a vertex for another
            for v_out in list(selected_set):
                trial_without = selected_set - {v_out}
                uncovered_after_removal = set(range(len(cliques))) - get_covered_cliques(trial_without)
                
                if not uncovered_after_removal:
                    # Can just remove
                    selected_set = trial_without
                    current_covered = get_covered_cliques(selected_set)
                    improved = True
                    break
                
                # Find best vertex to add to cover the uncovered
                best_v_in = None
                best_gain = 0
                for v_in in vertices:
                    if v_in in selected_set:
                        continue
                    gain = 0
                    for i in uncovered_after_removal:
                        if v_in in cliques[i]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_v_in = v_in
                
                if best_v_in is not None and best_gain >= len(uncovered_after_removal):
                    # This swap covers all previously uncovered
                    # But we need to check if overall size decreases or stays same
                    # Actually, we're replacing one vertex with another, so size stays same
                    # Only do it if it helps future removals - skip for simplicity
                    pass
            
            if not improved:
                break
        
        return sorted(selected_set)
    
    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        if not cliques:
            return []
        
        vertices = set(range(n))
        
        # Greedy set cover
        selected = greedy_set_cover(cliques, vertices)
        
        # Local search improvement
        if time.time() < deadline:
            selected = local_search_improve(selected, cliques, vertices)
        
        return selected
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            results.append([])
            continue
        
        result = solve_instance(n, p, seed)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
