"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = 145.0  # safety margin before 150s kill

def _build_graph(n, p, seed):
    """Build adjacency list from random_graph(n, p, seed) matching the seed's RNG order."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        union_px = P | X
        if union_px:
            # Pick pivot with max degree in P
            best_pivot = max(union_px, key=lambda u: len(adj[u] & P))
            candidates = P - adj[best_pivot]
        else:
            candidates = P
        
        for v in list(candidates):
            bk(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}
    
    bk(frozenset(), frozenset(range(n)), frozenset())
    return cliques

def _solve_instance(n, p, seed, time_budget):
    """Find a small set of vertices meeting every maximal clique of size >= 2."""
    adj = _build_graph(n, p, seed)
    cliques = _bron_kerbosch(adj)
    
    if not cliques:
        return []
    
    # Filter: only cliques with at least 2 vertices (already done in BK)
    # Build incidence: for each vertex, which cliques it covers
    # For each clique, which vertices are in it
    
    # Greedy set cover: repeatedly pick vertex covering most uncovered cliques
    # But we can do better with local search / simulated annealing
    
    # First, get a greedy solution
    uncovered = list(range(len(cliques)))
    # Map each clique index to set of vertices in it
    clique_verts = [set(c) for c in cliques]
    
    # Vertex -> list of clique indices it belongs to
    vert_cliques = {v: [] for v in range(n)}
    for ci, cv in enumerate(clique_verts):
        for v in cv:
            vert_cliques[v].append(ci)
    
    selected = set()
    uncovered_set = set(range(len(cliques)))
    
    # Greedy: pick vertex covering most uncovered cliques
    while uncovered_set:
        best_v = None
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            count = sum(1 for ci in vert_cliques[v] if ci in uncovered_set)
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None or best_count <= 0:
            break
        selected.add(best_v)
        for ci in vert_cliques[best_v]:
            uncovered_set.discard(ci)
    
    greedy_solution = list(selected)
    greedy_size = len(greedy_solution)
    
    if greedy_size == 0:
        return []
    
    # Now try to improve via local search: remove vertices if possible, then try swaps
    # Check if a vertex can be removed
    def can_remove(v, current_set):
        """Check if removing v still covers all cliques."""
        for ci, cv in enumerate(clique_verts):
            # Does this clique have another selected vertex?
            if v in cv:
                others = cv & current_set
                others.discard(v)
                if not others:
                    return False
        return True
    
    # Remove unnecessary vertices
    improved = True
    current_set = set(greedy_solution)
    while improved:
        improved = False
        for v in list(current_set):
            if can_remove(v, current_set):
                current_set.discard(v)
                improved = True
    
    # Try 1-1 swaps: remove one, add another, if it reduces size or same size but different
    # Actually for minimization, we want to try: remove v, then greedily add back minimal to cover
    # Better: try all pairs (remove v, add w) where w not in set, and check validity
    best_set = set(current_set)
    best_size = len(best_set)
    
    start_time = time.time()
    
    # Local search: try removing one vertex and adding one (or just removing)
    # Also try: for each vertex in set, try to replace it with a vertex not in set
    for iteration in range(200):
        if time.time() - start_time > time_budget * 0.8:
            break
        
        made_progress = False
        
        # Try single removals
        for v in list(best_set):
            new_set = best_set - {v}
            if not new_set:
                continue
            # Check coverage
            valid = True
            for ci, cv in enumerate(clique_verts):
                if cv & new_set:
                    continue
                else:
                    valid = False
                    break
            if valid:
                best_set = new_set
                best_size -= 1
                made_progress = True
                break
        
        if made_progress:
            continue
        
        # Try 1-for-1 swaps: remove v, add w
        for v in list(best_set):
            if time.time() - start_time > time_budget * 0.8:
                break
            for w in range(n):
                if w in best_set:
                    continue
                new_set = (best_set - {v}) | {w}
                if len(new_set) > best_size:
                    continue
                # Check coverage
                valid = True
                for ci, cv in enumerate(clique_verts):
                    if cv & new_set:
                        continue
                    else:
                        valid = False
                        break
                if valid:
                    best_set = new_set
                    best_size = len(best_set)
                    made_progress = True
                    break
            if made_progress:
                break
        
        if not made_progress:
            # Try 2-for-1: remove two vertices, add one
            if best_size >= 2:
                verts_list = list(best_set)
                for i in range(len(verts_list)):
                    if time.time() - start_time > time_budget * 0.8:
                        break
                    for j in range(i + 1, len(verts_list)):
                        v1, v2 = verts_list[i], verts_list[j]
                        for w in range(n):
                            if w in best_set:
                                continue
                            new_set = (best_set - {v1, v2}) | {w}
                            if len(new_set) > best_size - 1:
                                continue
                            valid = True
                            for ci, cv in enumerate(clique_verts):
                                if cv & new_set:
                                    continue
                                else:
                                    valid = False
                                    break
                            if valid:
                                best_set = new_set
                                best_size = len(best_set)
                                made_progress = True
                                break
                        if made_progress:
                            break
                    if made_progress:
                        break
            if not made_progress:
                break
    
    return sorted(best_set)

def construct():
    start_time = time.time()
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 5:
            # Not enough time, return best effort
            results.append([])
            continue
        
        # Allocate time budget proportionally, but ensure each gets at least some
        time_budget = remaining * 0.9  # leave 10% for subsequent instances
        if idx < len(instances) - 1:
            time_budget = min(time_budget, 40)  # cap per instance
        
        result = _solve_instance(n, p, seed, time_budget)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
