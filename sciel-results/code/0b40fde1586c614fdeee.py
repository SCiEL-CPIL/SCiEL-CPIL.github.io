import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        max_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            pivot_candidates = list(P) + list(X)
            if not pivot_candidates:
                return
            
            # Choose pivot with max |P ∩ N(u)|
            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u
            
            # Recurse on P \ N(pivot)
            P_set = set(P)
            X_set = set(X)
            
            if best_pivot is not None:
                candidates = [v for v in P_set if v not in adj[best_pivot]]
            else:
                candidates = list(P_set)
                
            for v in candidates:
                new_R = R | {v}
                new_P = P_set & adj[v]
                new_X = X_set & adj[v]
                
                bk(new_R, new_P, new_X)
                
                P_set.remove(v)
                X_set.add(v)
                
                if time.time() > deadline - 5:
                    return

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def solve_hitting_set_ilp(n, cliques):
        """
        Solves the Hitting Set problem using Branch and Bound.
        Variables: x_v (0/1) for each vertex v.
        Minimize: sum(x_v)
        Subject to: sum(x_v for v in C) >= 1 for all cliques C.
        
        We use a simple LP relaxation bound (fractional cover) for pruning.
        Since n is small, we can also use a greedy upper bound for the root node.
        """
        if not cliques:
            return set()
        
        # Preprocess: Remove duplicate cliques (shouldn't happen with BK, but safe)
        cliques = list(set(cliques))
        m = len(cliques)
        
        # Map vertex to list of clique indices it covers
        v_to_cliques = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                v_to_cliques[v].append(i)
        
        # Greedy initial solution for upper bound (UB)
        def greedy_solution():
            uncovered = set(range(m))
            sol = set()
            # Copy v_to_cliques for modification? No, just track uncovered.
            # To speed up greedy, we can pre-sort or just iterate.
            # Standard greedy: pick vertex covering most uncovered cliques.
            while uncovered:
                best_v = None
                max_hits = 0
                for v in range(n):
                    if v in sol:
                        continue
                    hits = 0
                    for ci in v_to_cliques[v]:
                        if ci in uncovered:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                if best_v is None:
                    break
                sol.add(best_v)
                for ci in v_to_cliques[best_v]:
                    if ci in uncovered:
                        uncovered.discard(ci)
            return sol

        initial_sol = greedy_solution()
        best_size = len(initial_sol)
        best_sol = initial_sol
        
        # Branch and Bound
        # State: (assigned, unassigned, covered_cliques_mask, current_cost)
        # We represent covered cliques as a set of indices or a bitmask if m < 64.
        # m can be large, so we use a set of uncovered cliques.
        
        # Optimization: If a clique has size 1, the vertex MUST be selected.
        # If a clique has size 2, at least one of the two must be selected.
        
        # Let's do a recursive search with pruning.
        # Order vertices by "importance" (frequency in cliques) to find good solutions early.
        vertex_order = sorted(range(n), key=lambda v: len(v_to_cliques[v]), reverse=True)
        
        def bb_search(assigned, unassigned, uncovered, cost):
            nonlocal best_size, best_sol
            
            if time.time() > deadline - 10:
                return
                
            # Pruning: if current cost + lower_bound >= best_size, prune
            # Lower bound: max(0, ceil(|uncovered| / max_clique_size_in_uncovered?)) 
            # A better LB: For each uncovered clique, we need at least 1 vertex.
            # Actually, a simple LB is: if cost >= best_size, prune.
            if cost >= best_size:
                return
                
            if not uncovered:
                if cost < best_size:
                    best_size = cost
                    best_sol = set(assigned)
                return
            
            # Heuristic LB:
            # The maximum number of uncovered cliques any single remaining vertex can cover is max_cap.
            # We need at least ceil(len(uncovered) / max_cap) more vertices.
            max_cap = 0
            for v in unassigned:
                cap = 0
                for ci in v_to_cliques[v]:
                    if ci in uncovered:
                        cap += 1
                if cap > max_cap:
                    max_cap = cap
                    if max_cap >= len(uncovered):
                        break
            if max_cap == 0:
                # Unsolvable with remaining vertices? Should not happen if problem is valid.
                return
            
            import math
            lb = math.ceil(len(uncovered) / max_cap)
            if cost + lb >= best_size:
                return
            
            # Choose a vertex to branch on.
            # Strategy: Pick a vertex that appears in the most uncovered cliques?
            # Or pick a clique and branch on its vertices?
            # Branching on a clique is often better for hitting sets.
            # Pick the smallest uncovered clique to branch on.
            min_clique = None
            min_size = float('inf')
            for ci in uncovered:
                c = cliques[ci]
                if len(c) < min_size:
                    min_size = len(c)
                    min_clique = c
                    if min_size == 2:
                        break
            
            if min_clique is None:
                return
                
            # Branch on vertices in min_clique
            # We must pick at least one vertex from min_clique.
            # We can branch by forcing a vertex to be IN or OUT.
            # To reduce branching factor, we can use "include" branches for each v in clique,
            # and one "exclude all" branch? No, that's not valid.
            # Standard: For v in clique:
            #   1. Include v (add to assigned, remove v from unassigned, update uncovered)
            #   2. Exclude v (remove v from unassigned) -- but we must ensure at least one is included.
            
            # Better approach:
            # Iterate through vertices in the chosen clique.
            # For the i-th vertex v_i:
            #   Branch 1: Include v_i.
            #   Branch 2: Exclude v_i (and remember that we still need to cover the clique with remaining vertices).
            # To avoid exponential blowup of "exclude" branches, we can use a "branch on the first uncovered clique" strategy:
            # Pick v1, v2, ..., vk from the clique.
            # Branch 1: Include v1.
            # Branch 2: Exclude v1, Include v2.
            # Branch 3: Exclude v1, Exclude v2, Include v3.
            # ...
            # Branch k: Exclude v1...vk-1, Include vk.
            # This ensures the clique is covered.
            
            clique_vertices = list(min_clique)
            # Sort clique vertices by frequency in uncovered to find good solutions faster
            clique_vertices.sort(key=lambda v: sum(1 for ci in v_to_cliques[v] if ci in uncovered), reverse=True)
            
            excluded = set()
            for v in clique_vertices:
                # Branch: Include v
                new_assigned = assigned | {v}
                new_unassigned = unassigned - {v}
                
                # Update uncovered
                new_uncovered = set(uncovered)
                for ci in v_to_cliques[v]:
                    if ci in new_uncovered:
                        new_uncovered.discard(ci)
                        
                bb_search(new_assigned, new_unassigned, new_uncovered, cost + 1)
                
                # Exclude v for subsequent branches
                excluded.add(v)
                
                if time.time() > deadline - 10:
                    return
                
            # Note: We don't need a separate "exclude all" branch because the loop above covers all cases where at least one is included.
            # Wait, this logic is slightly flawed. 
            # If I exclude v1, I must include one of {v2, ..., vk}.
            # My loop does:
            # 1. Include v1.
            # 2. Exclude v1, Include v2.
            # 3. Exclude v1, Exclude v2, Include v3.
            # This covers all valid assignments for this clique.
            
        # Initial call
        all_vertices = set(range(n))
        bb_search(set(), all_vertices, set(range(m)), 0)
        
        return best_sol

    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 15:
            # Fallback: return all vertices (guaranteed valid)
            results.append(list(range(n)))
            continue
            
        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        
        # Filter cliques of size < 2
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
            
        # Solve with ILP/Branch and Bound
        sol = solve_hitting_set_ilp(n, cliques)
        
        # Fallback if B&B didn't finish or failed
        if sol is None or len(sol) > n:
             # Greedy fallback
             v_to_cliques = defaultdict(list)
             for i, c in enumerate(cliques):
                 for v in c:
                     v_to_cliques[v].append(i)
             uncovered = set(range(len(cliques)))
             greedy_sol = set()
             while uncovered:
                 best_v = None
                 max_hits = 0
                 for v in range(n):
                     if v in greedy_sol: continue
                     hits = sum(1 for ci in v_to_cliques[v] if ci in uncovered)
                     if hits > max_hits:
                         max_hits = hits
                         best_v = v
                 if best_v is None: break
                 greedy_sol.add(best_v)
                 for ci in v_to_cliques[best_v]:
                     if ci in uncovered:
                         uncovered.discard(ci)
             sol = greedy_sol
             
        results.append(sorted(sol))
        
    return results

# EVOLVE-BLOCK-END
