"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        n = len(adj)
        maximal_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(tuple(sorted(R)))
                return
            
            # Pivot selection: choose vertex in P ∪ X with maximum degree in P
            pivot = max(P | X, key=lambda v: len(adj[v] & P))
            candidates = P - adj[pivot]
            
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return maximal_cliques

    def find_min_hitting_set(cliques, n, time_budget):
        """
        Find a minimum hitting set for the given list of cliques.
        Uses a greedy set cover approach with local search refinement.
        """
        if not cliques:
            return set()
        
        # Build incidence: for each vertex, which cliques it hits
        vertex_to_cliques = {v: [] for v in range(n)}
        for idx, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].append(idx)
        
        # Greedy set cover: repeatedly pick vertex that covers most uncovered cliques
        uncovered = set(range(len(cliques)))
        selected = set()
        
        while uncovered:
            if time.time() > time_budget:
                break
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                count = sum(1 for c in vertex_to_cliques[v] if c in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            selected.add(best_v)
            for c in vertex_to_cliques[best_v]:
                if c in uncovered:
                    uncovered.discard(c)
        
        # Local search: try to remove vertices and see if coverage still holds
        def is_valid(selected_set):
            for idx, clique in enumerate(cliques):
                if not any(v in selected_set for v in clique):
                    return False
            return True
        
        # Try removing each selected vertex
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                if time.time() > time_budget:
                    return selected
                trial = selected - {v}
                if is_valid(trial):
                    selected = trial
                    improved = True
        
        # Try swapping: remove one, add another
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                if time.time() > time_budget:
                    return selected
                trial = selected - {v}
                # Find cliques not covered by trial
                missing_cliques = [idx for idx, clique in enumerate(cliques) 
                                   if not any(u in trial for u in clique)]
                if not missing_cliques:
                    selected = trial
                    improved = True
                else:
                    # Try adding a single vertex to cover all missing
                    for w in range(n):
                        if w in selected:
                            continue
                        covers_all = all(any(u == w for u in cliques[idx]) 
                                        for idx in missing_cliques)
                        if covers_all:
                            selected = trial | {w}
                            improved = True
                            break
        
        return selected

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Fallback: return empty or best available
            results.append([])
            continue
        
        adj = generate_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        
        # Time budget for this instance
        remaining = deadline - time.time()
        time_budget = max(remaining * 0.8, 1.0)
        
        hitting_set = find_min_hitting_set(cliques, n, time_budget)
        results.append(sorted(hitting_set))
    
    return results
# EVOLVE-BLOCK-END
