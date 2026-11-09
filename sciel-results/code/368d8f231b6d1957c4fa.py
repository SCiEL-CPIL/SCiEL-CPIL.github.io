"""Rewritten program: greedy + local search + exact MIS on conflict graph for large reciprocal-Schur-free subset of [1,180]."""
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill

    N = 180
    # Precompute all forbidden triples (a,b,c) with distinct elements where 1/a = 1/b + 1/c
    forbidden = set()
    for b in range(1, N+1):
        for c in range(b+1, N+1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = frozenset([a, b, c])
                    forbidden.add(triple)
    
    # Identify the "core" vertices involved in any forbidden triple
    core_vertices = set()
    for t in forbidden:
        core_vertices.update(t)
    
    # --- Part 1: Greedy + Local Search (Original Logic) ---
    S_greedy = set(range(1, N+1))
    active_triples = list(forbidden)
    
    while True:
        found = None
        for t in active_triples:
            if all(x in S_greedy for x in t):
                found = t
                break
        if found is None:
            break
        if time.time() - start_time > DEADLINE - 30:
            break
        
        best_elem = None
        best_count = -1
        for elem in found:
            count = 0
            for t2 in active_triples:
                if elem in t2 and all(x in S_greedy for x in t2):
                    count += 1
            if count > best_count:
                best_count = count
                best_elem = elem
        S_greedy.discard(best_elem)
    
    removed_greedy = set(range(1, N+1)) - S_greedy
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 30:
        improved = False
        for elem in list(removed_greedy):
            if time.time() - start_time > DEADLINE - 30:
                break
            S_test = S_greedy | {elem}
            valid = True
            for t in forbidden:
                if elem in t and all(x in S_test for x in t):
                    valid = False
                    break
            if valid:
                S_greedy = S_test
                removed_greedy.discard(elem)
                improved = True
                
    # Simple swap improvement
    if time.time() - start_time < DEADLINE - 30:
        for _ in range(2):
            if time.time() - start_time > DEADLINE - 30:
                break
            improved = False
            for keep in list(S_greedy):
                if time.time() - start_time > DEADLINE - 30:
                    break
                for add in list(removed_greedy):
                    if keep == add:
                        continue
                    S_test = (S_greedy - {keep}) | {add}
                    valid = True
                    for t in forbidden:
                        if all(x in S_test for x in t):
                            valid = False
                            break
                    if valid and len(S_test) >= len(S_greedy):
                        S_greedy = S_test
                        removed_greedy = set(range(1, N+1)) - S_greedy
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # --- Part 2: Exact MIS on Conflict Graph (Directive) ---
    # Build conflict graph on core_vertices
    # Edge (u,v) if u, v co-occur in any forbidden triple
    adj = {v: set() for v in core_vertices}
    for t in forbidden:
        u, v, w = list(t)
        adj[u].add(v); adj[u].add(w)
        adj[v].add(u); adj[v].add(w)
        adj[w].add(u); adj[w].add(v)
        
    # Implement Maximum Independent Set (MIS) using Branch and Bound
    # Since N is small for core, we can use a recursive solver
    # We want to maximize the size of the independent set in G_core.
    # An independent set in G_core is a set of vertices with no edges between them.
    # This means no two vertices in the set are in the same forbidden triple?
    # No, if u,v are in a triple, they are connected by an edge. So they cannot both be in the independent set.
    # So the independent set cannot contain any pair from a forbidden triple.
    # This is a valid subset of the original problem's solution space (it's stricter).
    
    def solve_mis(vertices, adj):
        if not vertices:
            return set()
        
        # Heuristic: pick vertex with highest degree in the induced subgraph?
        # Or just pick the first one.
        # Standard B&B: Branch on a vertex v.
        # 1. Include v: add v to set, remove v and its neighbors from graph.
        # 2. Exclude v: remove v from graph.
        # Take the max of the two.
        
        # Optimization: If the graph is empty, return empty.
        # If all vertices are isolated, return all.
        
        # To make it fast, we can use a simple recursion with memoization or just B&B with a good bound.
        # Bound: Greedy upper bound on the remaining graph?
        # For small N (<50), simple B&B is fine.
        
        v = next(iter(vertices))
        # Branch 1: Include v
        # Neighbors of v in the current vertex set
        neighbors = adj[v] & vertices
        remaining_after_include = vertices - neighbors - {v}
        sol1 = solve_mis(remaining_after_include, adj) | {v}
        
        # Branch 2: Exclude v
        remaining_after_exclude = vertices - {v}
        sol2 = solve_mis(remaining_after_exclude, adj)
        
        return sol1 if len(sol1) > len(sol2) else sol2

    # To avoid recursion depth issues or exponential blowup on worst-case, 
    # we can restrict the search or use a better heuristic.
    # Given the sparsity, it should be fast.
    # However, Python recursion limit might be an issue if core is large.
    # Let's check core size. If it's > 20, we might need to be careful.
    # Let's assume it's small. If not, we fall back to greedy only.
    
    S_mis = set()
    if len(core_vertices) < 40:
        try:
            # Use a non-recursive or optimized recursive approach?
            # Simple recursion is okay for < 40 nodes if pruned.
            # But pure B&B without strong bounds is $O(2^N)$.
            # Let's add a simple greedy upper bound to prune?
            # Or just trust the small N.
            # Actually, let's just run it. If it takes too long, we catch it.
            # But we can't catch time easily inside recursion.
            # Let's use a simple heuristic: 
            # If the graph is bipartite, MIS is easy? No.
            # Let's just run the recursive solver.
            core_list = list(core_vertices)
            # Map to indices for faster set operations?
            # The set operations above are fine for < 40 elements.
            
            mis_core = solve_mis(core_vertices, adj)
            # Elements not in core are isolated in conflict graph, so they are always in MIS
            S_mis = (set(range(1, N+1)) - core_vertices) | mis_core
        except Exception:
            S_mis = S_greedy # Fallback if it fails or times out (though we can't timeout easily here)
    else:
        S_mis = S_greedy

    # Choose the larger set
    if len(S_mis) > len(S_greedy):
        S_final = S_mis
    else:
        S_final = S_greedy

    # Final validation
    result_set = set(S_final)
    for t in forbidden:
        if all(x in result_set for x in t):
            for x in t:
                if x in result_set:
                    result_set.discard(x)
                    break
            break
    
    result = sorted(result_set)
    if not result:
        result = [91]
        
    return result
