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

    def greedy_hitting_set(cliques, adj, n):
        """Greedy: repeatedly pick vertex covering most uncovered cliques."""
        uncovered = list(cliques)
        if not uncovered:
            return set()
        
        # Build incidence: for each vertex, which cliques it's in
        # We'll maintain a count of how many uncovered cliques each vertex covers
        vertex_clique_count = {}
        for c in uncovered:
            for v in c:
                vertex_clique_count[v] = vertex_clique_count.get(v, 0) + 1
        
        selected = set()
        remaining = set(uncovered)
        
        while remaining:
            if time.time() > deadline:
                break
            # Find vertex with max coverage among remaining cliques
            best_v = None
            best_count = -1
            # Only consider vertices that appear in at least one remaining clique
            candidates = set()
            for c in remaining:
                candidates.update(c)
            
            for v in candidates:
                count = sum(1 for c in remaining if v in c)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None:
                break
            
            selected.add(best_v)
            # Remove all cliques containing best_v
            remaining = {c for c in remaining if best_v not in c}
        
        return selected

    def remove_redundant(selected, cliques):
        """Remove vertices that are not needed (their removal still leaves all cliques hit)."""
        if not selected:
            return selected
        selected_list = list(selected)
        result = set(selected)
        
        for v in selected_list:
            if time.time() > deadline:
                break
            # Try removing v: check if all cliques are still hit
            test_set = result - {v}
            valid = True
            for c in cliques:
                if not (c & test_set):
                    valid = False
                    break
            if valid:
                result.discard(v)
        
        return result

    def local_search_2exchange(selected, cliques, adj, n, max_iters=2000):
        """Try 1-for-1 exchanges: remove one selected vertex, add one unselected vertex."""
        if not selected:
            return selected
        selected_set = set(selected)
        best_size = len(selected_set)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
            
            # For each vertex in selected, try removing it and adding a better replacement
            for v_remove in list(selected_set):
                if time.time() > deadline:
                    break
                # After removing v_remove, find cliques that become unhittable
                # These are cliques that contain v_remove and no other selected vertex
                critical_cliques = [c for c in cliques if v_remove in c and not (c - {v_remove}) & selected_set]
                
                if not critical_cliques:
                    # Safe to remove v_remove
                    selected_set.discard(v_remove)
                    best_size = len(selected_set)
                    improved = True
                    break
                
                # Find a vertex that can cover all critical cliques
                # The replacement must be in the intersection of all critical cliques
                # (or at least in each critical clique)
                # Actually, the replacement just needs to hit each critical clique.
                # A single vertex w hits critical clique c if w in c.
                # So w must be in ALL critical cliques? No, w just needs to be in each one.
                # If |critical_cliques| == 1, any vertex in that clique works.
                # If |critical_cliques| > 1, w must be in the intersection.
                
                if len(critical_cliques) == 1:
                    c = critical_cliques[0]
                    candidates = [w for w in c if w not in selected_set]
                else:
                    # Intersection of all critical cliques
                    intersection = set(critical_cliques[0])
                    for c in critical_cliques[1:]:
                        intersection &= c
                    candidates = [w for w in intersection if w not in selected_set]
                
                if not candidates:
                    continue
                
                # Pick candidate that maximizes additional coverage
                best_w = None
                best_gain = -1
                for w in candidates:
                    # Count how many currently-unhit-by-other-selected cliques w covers
                    # Actually, we need w to cover all critical cliques (guaranteed by construction)
                    # and ideally also cover other cliques to allow further reductions
                    # Simple heuristic: pick w with highest degree in the graph among candidates
                    gain = len(adj[w])
                    if gain > best_gain:
                        best_gain = gain
                        best_w = w
                
                if best_w is not None:
                    selected_set.discard(v_remove)
                    selected_set.add(best_w)
                    if len(selected_set) < best_size:
                        best_size = len(selected_set)
                        improved = True
                        break
                    # Same size, keep going
                    improved = True
        
        return selected_set

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        
        if not cliques:
            return []
        
        # Multiple greedy restarts with different random orderings
        best_solution = None
        best_size = float('inf')
        
        rng = random.Random(seed + 1000)
        num_restarts = 30
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Greedy with random tie-breaking
            uncovered = list(cliques)
            if not uncovered:
                break
            
            selected = set()
            remaining = set(uncovered)
            
            while remaining:
                if time.time() > deadline:
                    break
                
                # Find all vertices and their coverage counts
                candidates = set()
                for c in remaining:
                    candidates.update(c)
                
                counts = {}
                for v in candidates:
                    counts[v] = sum(1 for c in remaining if v in c)
                
                # Get max count
                max_count = max(counts.values()) if counts else 0
                if max_count == 0:
                    break
                
                # Randomly pick among vertices with max count
                best_vertices = [v for v, cnt in counts.items() if cnt == max_count]
                v = rng.choice(best_vertices)
                
                selected.add(v)
                remaining = {c for c in remaining if v not in c}
            
            # Remove redundant vertices
            selected = remove_redundant(selected, cliques)
            
            # Local search
            selected = local_search_2exchange(selected, cliques, adj, n, max_iters=500)
            
            # Final redundancy removal
            selected = remove_redundant(selected, cliques)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = set(selected)
            
            if best_size == 0:
                break
        
        if best_solution is None:
            # Fallback: greedy once
            best_solution = greedy_hitting_set(cliques, adj, n)
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
