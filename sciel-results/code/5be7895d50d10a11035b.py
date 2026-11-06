"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _generate_graph(n, p, seed):
    """Generate the graph matching the verifier's random_graph."""
    rng = random.Random(seed)
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bron_kerbosch(adj, n):
    """Find all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        candidates = list(P) + list(X)
        pivot = None
        max_count = -1
        for u in candidates:
            count = len(P & adj[u])
            if count > max_count:
                max_count = count
                pivot = u
        if pivot is not None:
            # Iterate over P \ N(pivot)
            to_iterate = P - adj[pivot]
        else:
            to_iterate = P
        for v in list(to_iterate):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            P.remove(v)
            X.add(v)
            bk(new_R, new_P, new_X)
            P.add(v)
            X.remove(v)
    bk(frozenset(), frozenset(range(n)), frozenset())
    return cliques

def _greedy_transversal(cliques, n):
    """Greedy set cover: pick vertex covering most uncovered cliques."""
    # Map each vertex to the set of clique indices it covers
    vertex_cliques = defaultdict(set)
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].add(i)
    
    uncovered = set(range(len(cliques)))
    selected = []
    
    while uncovered:
        best_v = None
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            count = len(vertex_cliques[v] & uncovered)
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None or best_count <= 0:
            break
        selected.append(best_v)
        uncovered -= vertex_cliques[best_v]
    
    return selected

def _local_search(selected, cliques, n, max_iters=5000):
    """Local search: try to remove vertices, then swap to reduce size."""
    selected = list(selected)
    
    def coverage_counts(sel):
        """Count how many cliques each vertex covers (among uncovered)."""
        counts = [0] * n
        for clique in cliques:
            covered = False
            for v in clique:
                if v in sel:
                    covered = True
                    break
            if not covered:
                # This clique is uncovered - shouldn't happen in valid solution
                pass
        # Better: count for each selected vertex, how many cliques ONLY it covers
        # and how many it shares
        cliq_covered = [False] * len(cliques)
        for i, clique in enumerate(cliques):
            for v in clique:
                if v in sel:
                    cliq_covered[i] = True
                    break
        # For each vertex in sel, count cliques where it's the only cover
        only_cover = [0] * n
        for i, clique in enumerate(cliques):
            if not cliq_covered[i]:
                continue
            covers = [v for v in clique if v in sel]
            if len(covers) == 1:
                only_cover[covers[0]] += 1
        return only_cover
    
    sel_set = set(selected)
    
    # Phase 1: Try to remove vertices
    changed = True
    while changed:
        changed = False
        only_cover = coverage_counts(selected)
        for v in list(selected):
            if only_cover[v] == 0:
                selected.remove(v)
                sel_set.discard(v)
                changed = True
    
    # Phase 2: Try swaps (remove one, add one if it helps)
    for _ in range(max_iters):
        improved = False
        only_cover = coverage_counts(selected)
        for v in list(selected):
            if only_cover[v] > 0:
                # This vertex is critical; try to find a replacement
                # Find cliques that only v covers
                critical_cliques = []
                for i, clique in enumerate(cliques):
                    if v in clique:
                        covers = [u for u in clique if u in sel_set]
                        if len(covers) == 1:
                            critical_cliques.append(i)
                # Try to find a vertex w not in selected that covers all critical_cliques
                for w in range(n):
                    if w in sel_set:
                        continue
                    ok = True
                    for ci in critical_cliques:
                        if w not in cliques[ci]:
                            ok = False
                            break
                    if ok:
                        # Check if w doesn't break other coverage
                        # Remove v, add w, check all cliques still covered
                        new_sel = sel_set - {v} | {w}
                        valid = True
                        for clique in cliques:
                            if not (clique & new_sel):
                                valid = False
                                break
                        if valid:
                            selected.remove(v)
                            selected.append(w)
                            sel_set = set(selected)
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Return best effort: just all vertices for remaining
            results.append(list(range(n)))
            continue
        
        adj = _generate_graph(n, p, seed)
        cliques = _bron_kerbosch(adj, n)
        
        # Greedy transversal
        selected = _greedy_transversal(cliques, n)
        
        # Local search improvement
        if time.time() < deadline:
            selected = _local_search(selected, cliques, n, max_iters=2000)
        
        results.append(sorted(selected))
    
    return results
# EVOLVE-BLOCK-END
