"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from random import Random

def _generate_graph(n, p, seed):
    """Generate a random graph using the same RNG as the seed program."""
    rng = Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    # Build adjacency as sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj, edges

def _bron_kerbosch(adj, n):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        union_px = P | X
        if not union_px:
            return
        # Find pivot
        best_u = None
        best_count = -1
        for u in union_px:
            count = len(P & adj[u])
            if count > best_count:
                best_count = count
                best_u = u
        # Iterate over P \ N(pivot)
        candidates = list(P - adj[best_u]) if best_u is not None else list(P)
        for v in candidates:
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bk(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _filter_cliQUES(cliques):
    """Filter to cliques with at least 2 vertices."""
    return [c for c in cliques if len(c) >= 2]

def _greedy_set_cover(cliques, n):
    """Greedy set cover: repeatedly pick the vertex covering the most uncovered cliques."""
    if not cliques:
        return []
    
    # Map each vertex to the set of clique indices it covers
    vertex_cliques = {v: set() for v in range(n)}
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].add(i)
    
    uncovered = set(range(len(cliques)))
    selected = []
    
    while uncovered:
        # Find vertex covering most uncovered cliques
        best_v = None
        best_count = -1
        for v in range(n):
            count = len(vertex_cliques[v] & uncovered)
            if count > best_count:
                best_count = count
                best_v = v
        selected.append(best_v)
        uncovered -= vertex_cliques[best_v]
    
    return sorted(selected)

def _local_search(cliques, n, initial_solution, max_iterations=10000):
    """Local search: try removing vertices and adding vertices to improve."""
    solution = set(initial_solution)
    covered = set()
    for i, clique in enumerate(cliques):
        if clique & solution:
            covered.add(i)
    
    best_solution = set(solution)
    best_size = len(solution)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try removing each vertex
        for v in list(solution):
            if len(solution) <= 1:
                break
            # Check if removing v still covers all cliques
            new_solution = solution - {v}
            all_covered = all(clique & new_solution for clique in cliques)
            if all_covered:
                solution = new_solution
                best_solution = set(solution)
                best_size = len(solution)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one vertex, add another
        for v_remove in list(solution):
            for v_add in range(n):
                if v_add in solution:
                    continue
                new_solution = (solution - {v_remove}) | {v_add}
                if len(new_solution) >= len(solution):
                    continue
                all_covered = all(clique & new_solution for clique in cliques)
                if all_covered:
                    solution = new_solution
                    best_solution = set(solution)
                    best_size = len(solution)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return sorted(best_solution)

def _branch_and_bound(cliques, n, time_limit=5.0):
    """Branch and bound with disjoint clique lower bound."""
    start_time = time.time()
    
    # Precompute: for each vertex, which cliques it covers
    vertex_cliques = [set() for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].add(i)
    
    # Find a greedy solution first
    greedy_sol = _greedy_set_cover(cliques, n)
    best_solution = set(greedy_sol)
    best_size = len(best_solution)
    
    # Disjoint clique lower bound: find a set of pairwise disjoint cliques
    # The size of the maximum such set is a lower bound
    def compute_lower_bound(uncovered_cliques):
        if not uncovered_cliques:
            return 0
        # Greedy: pick clique, remove all cliques that intersect it
        lb = 0
        remaining = list(uncovered_cliques)
        while remaining:
            # Pick the smallest clique (hardest to cover)
            remaining.sort(key=len)
            clique = remaining[0]
            lb += 1
            # Remove all cliques that share a vertex with this clique
            clique_set = set(clique)
            remaining = [c for c in remaining if not (set(c) & clique_set)]
        return lb
    
    # Recursive search
    def search(selected, uncovered_cliques, depth):
        nonlocal best_solution, best_size
        if time.time() - start_time > time_limit:
            return
        
        if not uncovered_cliques:
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = set(selected)
            return
        
        # Lower bound: need at least lb more vertices
        lb = compute_lower_bound(uncovered_cliques)
        if len(selected) + lb >= best_size:
            return
        
        # Choose the next clique to cover
        # Pick the uncovered clique with fewest candidate vertices
        candidate_clique = min(uncovered_cliques, key=lambda c: len(c))
        
        # Try each vertex in this clique
        for v in candidate_clique:
            new_selected = selected | {v}
            new_uncovered = [c for c in uncovered_cliques if v not in c]
            search(new_selected, new_uncovered, depth + 1)
    
    search(set(), list(range(len(cliques))), 0)
    
    return sorted(best_solution)

def construct():
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Fallback: greedy solution
            adj, edges = _generate_graph(n, p, seed)
            cliques = _bron_kerbosch(adj, n)
            cliques = _filter_cliQUES(cliques)
            results.append(_greedy_set_cover(cliques, n))
            continue
        
        adj, edges = _generate_graph(n, p, seed)
        cliques = _bron_kerbosch(adj, n)
        cliques = _filter_cliQUES(cliques)
        
        if not cliques:
            results.append([])
            continue
        
        # Start with greedy
        solution = _greedy_set_cover(cliques, n)
        
        # Local search improvement
        if time.time() < deadline - 10:
            solution = _local_search(cliques, n, solution, max_iterations=5000)
        
        # Branch and bound if time allows
        if time.time() < deadline - 15:
            bb_solution = _branch_and_bound(cliques, n, time_limit=min(10, deadline - time.time() - 5))
            if len(bb_solution) < len(solution):
                solution = bb_solution
        
        # Final local search polish
        if time.time() < deadline - 5:
            solution = _local_search(cliques, n, solution, max_iterations=2000)
        
        results.append(solution)
    
    return results
# EVOLVE-BLOCK-END
