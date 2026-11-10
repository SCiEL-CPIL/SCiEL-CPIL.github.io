"""Stronger algorithm for hitting all maximal cliques (size >= 2) in three random graphs."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            pivot_candidates = P | X
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
            else:
                return
            # Only recurse on vertices in P that are NOT adjacent to pivot
            candidates = [v for v in P if v not in adj[pivot]] if pivot is not None else list(P)
            for v in candidates:
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def remove_redundant(selected, cliques):
        """Remove vertices that are not needed (their removal still leaves all cliques hit)."""
        if not selected:
            return selected
        result = set(selected)
        
        for v in list(result):
            if time.time() > deadline:
                break
            test_set = result - {v}
            valid = True
            for c in cliques:
                if not (c & test_set):
                    valid = False
                    break
            if valid:
                result.discard(v)
        
        return result

    def greedy_hitting(cliques, adj, n, rng=None, max_iters=10000):
        """Greedy set cover with optional random tie-breaking."""
        if not cliques:
            return set()
        
        remaining = set(cliques)
        selected = set()
        
        while remaining and len(selected) < max_iters:
            if time.time() > deadline:
                break
            
            # Build coverage count for each vertex
            counts = {}
            for c in remaining:
                for v in c:
                    counts[v] = counts.get(v, 0) + 1
            
            if not counts:
                break
            
            max_count = max(counts.values())
            best_vertices = [v for v, cnt in counts.items() if cnt == max_count]
            
            if rng:
                v = rng.choice(best_vertices)
            else:
                v = best_vertices[0]
            
            selected.add(v)
            remaining = {c for c in remaining if v not in c}
        
        return selected

    def local_search(selected, cliques, adj, n, max_rounds=100):
        """2-exchange local search: try removing one vertex and adding one or two others."""
        if not selected:
            return selected
        
        selected_set = set(selected)
        best_size = len(selected_set)
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try 1-removal: can we drop a vertex?
            for v in list(selected_set):
                test = selected_set - {v}
                if all(c & test for c in cliques):
                    selected_set.discard(v)
                    best_size = len(selected_set)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-1 swap: remove v, add w
            for v in list(selected_set):
                if time.time() > deadline:
                    break
                
                # Find cliques that only v covers
                critical = [c for c in cliques if v in c and not (c - {v}) & selected_set]
                if not critical:
                    # v is redundant
                    selected_set.discard(v)
                    best_size = len(selected_set)
                    improved = True
                    break
                
                # Candidate w must be in all critical cliques
                if len(critical) == 1:
                    candidates = [w for w in critical[0] if w not in selected_set]
                else:
                    inter = set(critical[0])
                    for c in critical[1:]:
                        inter &= c
                    candidates = [w for w in inter if w not in selected_set]
                
                if not candidates:
                    continue
                
                # Pick w that maximizes additional coverage
                best_w = None
                best_gain = -1
                for w in candidates:
                    # Count cliques that w covers which are not covered by (selected - v)
                    gain = 0
                    for c in cliques:
                        if w in c and not (c - {w}) & (selected_set - {v}):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_w = w
                
                if best_w is not None and best_gain > 0:
                    selected_set.discard(v)
                    selected_set.add(best_w)
                    if len(selected_set) < best_size:
                        best_size = len(selected_set)
                    improved = True
                    break
            
            if not improved:
                break
        
        return selected_set

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        
        if not cliques:
            return []
        
        # Precompute: for each clique, its member set as a frozenset for fast intersection
        # Also build vertex -> cliques incidence for efficiency
        vertex_to_cliques = {}
        for c in cliques:
            for v in c:
                vertex_to_cliques.setdefault(v, []).append(c)
        
        best_solution = None
        best_size = float('inf')
        
        rng = random.Random(seed + 1000)
        num_restarts = 50
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Greedy with random tie-breaking
            if rng:
                selected = greedy_hitting(cliques, adj, n, rng=rng)
            else:
                selected = greedy_hitting(cliques, adj, n)
            
            # Remove redundant vertices
            selected = remove_redundant(selected, cliques)
            
            # Local search
            selected = local_search(selected, cliques, adj, n, max_rounds=50)
            
            # Final redundancy removal
            selected = remove_redundant(selected, cliques)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = set(selected)
            
            if best_size == 0:
                break
            
            # If we've found a small solution, do more intensive local search
            if best_size <= 12 and time.time() < deadline - 10:
                selected = local_search(selected, cliques, adj, n, max_rounds=200)
                selected = remove_redundant(selected, cliques)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
        
        if best_solution is None:
            best_solution = greedy_hitting(cliques, adj, n)
            best_solution = remove_redundant(best_solution, cliques)
        
        return sorted(best_solution)

    # Solve each instance
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Emergency: return minimal valid set (all vertices in any edge)
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u, v)}
            results.append(sorted(touched))
            continue
        
        sol = solve_instance(n, p, seed)
        results.append(sol)
    
    return results
