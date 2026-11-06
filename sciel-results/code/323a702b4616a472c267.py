import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for 150s limit
    
    results = []
    
    for n, p, seed in [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]:
        if time.time() - start_time > time_limit - 15:
            # Emergency: return empty or quick greedy
            results.append([])
            continue
        
        # 1. Generate Graph
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # 2. Enumerate Maximal Cliques using Bron-Kerbosch with pivoting
        cliques = []
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot: vertex in P ∪ X with max degree in P
            U = P | X
            if not U:
                return
            
            # Pick pivot u that maximizes |P ∩ N(u)|
            # This minimizes |P \ N(u)|, reducing branching
            best_pivot = None
            max_common = -1
            for u in U:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u
            
            # Iterate over P \ N(pivot)
            candidates = P - adj[best_pivot] if best_pivot is not None else P
            
            for v in list(candidates):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.discard(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        
        # 3. Precompute vertex-to-clique mapping
        if not cliques:
            results.append([])
            continue
        
        # Build vertex -> list of cliques containing that vertex
        vertex_cliques = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(i)
        
        # Relevant vertices
        relevant_vertices = set()
        for c in cliques:
            relevant_vertices.update(c)
        
        # 4. Solve Hitting Set Problem
        best_solution = None
        best_size = n
        
        inst_start = time.time()
        inst_budget = min(40.0, time_limit - 5 - (time.time() - start_time))
        
        search_rng = random.Random(seed + 1000)
        
        def greedy_hitting_set(cliques, vertex_cliques, relevant_vertices, rng, tie_break='max'):
            """
            Greedy hitting set with optional randomization.
            tie_break: 'max' for deterministic, 'random' for randomized tie-breaking
            """
            remaining = set(range(len(cliques)))
            solution = set()
            
            while remaining:
                # Find vertex that covers most remaining cliques
                best_v = None
                max_covers = 0
                best_covers_list = []
                
                for v in relevant_vertices:
                    covers = 0
                    for ci in vertex_cliques[v]:
                        if ci in remaining:
                            covers += 1
                    
                    if covers > max_covers:
                        max_covers = covers
                        best_v = v
                        best_covers_list = [v]
                    elif covers == max_covers and covers > 0:
                        best_covers_list.append(v)
                
                if best_v is None or max_covers == 0:
                    break
                
                # Tie-breaking
                if tie_break == 'random' and len(best_covers_list) > 1:
                    best_v = rng.choice(best_covers_list)
                
                solution.add(best_v)
                
                # Remove covered cliques
                new_remaining = set()
                for ci in remaining:
                    if best_v not in cliques[ci]:
                        new_remaining.add(ci)
                remaining = new_remaining
            
            return solution
        
        def reduce_solution(solution, cliques, vertex_cliques):
            """Remove redundant vertices from solution."""
            S = set(solution)
            changed = True
            while changed:
                changed = False
                for v in list(S):
                    S.remove(v)
                    # Check if all cliques are still hit
                    valid = True
                    for c in cliques:
                        if not (c & S):
                            valid = False
                            break
                    if valid:
                        changed = True
                    else:
                        S.add(v)
            return S
        
        def local_search_improve(solution, cliques, vertex_cliques, relevant_vertices, rng):
            """
            Iterated local search: try to swap vertices to improve solution.
            """
            S = set(solution)
            improved = True
            max_iters = 50
            iters = 0
            
            while improved and iters < max_iters:
                improved = False
                iters += 1
                
                # Try removing a vertex and adding a different one
                for v_remove in list(S):
                    S.remove(v_remove)
                    
                    # Find uncovered cliques
                    uncovered = [c for c in cliques if not (c & S)]
                    
                    if not uncovered:
                        # v_remove was redundant
                        improved = True
                        break
                    
                    # Try to find a single vertex that covers all uncovered
                    best_add = None
                    for v_add in relevant_vertices - S:
                        covers_all = True
                        for c in uncovered:
                            if v_add not in c:
                                covers_all = False
                                break
                        if covers_all:
                            best_add = v_add
                            break
                    
                    if best_add is not None:
                        S.add(best_add)
                        improved = True
                        break
                    else:
                        S.add(v_remove)
                
                # Try 2-opt: remove two, add two
                if not improved and len(S) >= 2:
                    S_list = list(S)
                    for i in range(len(S_list)):
                        for j in range(i + 1, len(S_list)):
                            v1, v2 = S_list[i], S_list[j]
                            S.remove(v1)
                            S.remove(v2)
                            
                            uncovered = [c for c in cliques if not (c & S)]
                            
                            if not uncovered:
                                improved = True
                                break
                            
                            # Try to find two vertices that cover all uncovered
                            found = False
                            for v1_add in relevant_vertices - S:
                                for v2_add in relevant_vertices - S - {v1_add}:
                                    covers_all = True
                                    for c in uncovered:
                                        if v1_add not in c and v2_add not in c:
                                            covers_all = False
                                            break
                                    if covers_all:
                                        S.add(v1_add)
                                        S.add(v2_add)
                                        improved = True
                                        found = True
                                        break
                                if found:
                                    break
                            
                            if not found:
                                S.add(v1)
                                S.add(v2)
                        
                        if improved:
                            break
                    
                    if improved:
                        break
            
            return S
        
        def perturbation(solution, relevant_vertices, rng, strength=2):
            """Randomly perturb the solution."""
            S = set(solution)
            for _ in range(strength):
                if S:
                    v = rng.choice(list(S))
                    S.remove(v)
                if relevant_vertices:
                    v = rng.choice(list(relevant_vertices))
                    S.add(v)
            return S
        
        # Run multiple strategies
        for i in range(20):  # Up to 20 iterations
            if time.time() - inst_start > inst_budget:
                break
            
            # Strategy 1: Deterministic greedy
            if i == 0:
                sol = greedy_hitting_set(cliques, vertex_cliques, relevant_vertices, search_rng, 'max')
            # Strategy 2: Randomized greedy
            elif i == 1:
                sol = greedy_hitting_set(cliques, vertex_cliques, relevant_vertices, search_rng, 'random')
            # Strategy 3+: Perturbation + local search
            else:
                if best_solution is None:
                    sol = greedy_hitting_set(cliques, vertex_cliques, relevant_vertices, search_rng, 'random')
                else:
                    sol = perturbation(best_solution, relevant_vertices, search_rng, strength=2)
            
            # Reduce
            sol = reduce_solution(sol, cliques, vertex_cliques)
            
            # Local search
            sol = local_search_improve(sol, cliques, vertex_cliques, relevant_vertices, search_rng)
            
            # Final reduction
            sol = reduce_solution(sol, cliques, vertex_cliques)
            
            if best_solution is None or len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
            
            # Early stop if we hit a good lower bound
            # Simple lower bound: at least 1 if there's any clique
            # Better: use LP relaxation or greedy bound
            # For now, if best_size <= 3, we can stop
            if best_size <= 3:
                break
        
        if best_solution is None:
            # Fallback
            best_solution = greedy_hitting_set(cliques, vertex_cliques, relevant_vertices, search_rng, 'max')
            best_solution = reduce_solution(best_solution, cliques, vertex_cliques)
        
        results.append(sorted(best_solution))
    
    return results
