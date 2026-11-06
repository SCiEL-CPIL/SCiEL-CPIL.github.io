"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _build_graph(n, p, seed):
    """Build adjacency sets for a random graph."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj, n):
    """Find all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        candidates = list(P) + list(X)
        if not candidates:
            return
        best_pivot = max(candidates, key=lambda u: len(P & adj[u]))
        
        for v in list(P - adj[best_pivot]):
            P.remove(v)
            X.remove(v)
            bk(R | {v}, P & adj[v], X & adj[v])
            X.add(v)
    
    bk(set(), set(range(n)), set())
    return cliques

def _greedy_set_cover(cliques, n):
    """Greedy set cover: pick vertex covering most uncovered cliques."""
    uncovered = set(range(len(cliques)))
    selected = set()
    
    # Precompute which cliques each vertex is in
    vertex_to_cliques = {v: set() for v in range(n)}
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_to_cliques[v].add(i)
    
    while uncovered:
        best_v = None
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            count = len(vertex_to_cliques[v] & uncovered)
            if count > best_count:
                best_count = count
                best_v = v
        if best_count == 0:
            break
        selected.add(best_v)
        uncovered -= vertex_to_cliques[best_v]
    
    return selected

def _local_search(selected, cliques, n, max_iters=500):
    """Local search: try removing a vertex or swapping to reduce size."""
    selected = set(selected)
    
    def is_valid(sel):
        for clique in cliques:
            if not (clique & sel):
                return False
        return True
    
    if not is_valid(selected):
        return selected
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Try removing a vertex
        for v in list(selected):
            new_sel = selected - {v}
            if is_valid(new_sel):
                selected = new_sel
                improved = True
                break
        
        # Try swapping: remove v, add w
        if not improved:
            for v in list(selected):
                for w in range(n):
                    if w in selected:
                        continue
                    new_sel = (selected - {v}) | {w}
                    if is_valid(new_sel):
                        # Check if it's better (same size, so just accept for diversity)
                        selected = new_sel
                        improved = True
                        break
                if improved:
                    break
    
    return selected

def _solve_instance(n, p, seed, time_limit=15):
    """Solve a single instance with time management."""
    start = time.time()
    adj = _build_graph(n, p, seed)
    cliques = _bron_kerbosch(adj, n)
    
    if not cliques:
        return []
    
    # Initial greedy solution
    selected = _greedy_set_cover(cliques, n)
    
    # Local search improvement
    remaining = time_limit - (time.time() - start)
    if remaining > 2:
        selected = _local_search(selected, cliques, n, max_iters=200)
    
    # Verify validity
    for clique in cliques:
        if not (clique & selected):
            # Fallback: add a vertex from this clique
            for v in clique:
                selected.add(v)
                break
    
    return sorted(selected)

def construct():
    import time
    total_start = time.time()
    deadline = total_start + 140  # Safety margin before 150s kill
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for i, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start
        remaining = deadline - elapsed
        if remaining < 5:
            # Not enough time, use quick fallback
            adj = _build_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            results.append(sorted(touched))
            continue
        
        time_limit = max(5, remaining / (len(instances) - i) - 2)
        result = _solve_instance(n, p, seed, time_limit=time_limit)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
