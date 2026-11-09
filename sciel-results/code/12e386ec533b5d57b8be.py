import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    safety_margin = 5.0
    
    instances = [
        (30, 0.3, 201),
        (35, 0.25, 202),
        (40, 0.2, 203)
    ]
    
    results = []
    
    for n, p, seed in instances:
        # 1. Generate Graph
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # 2. Enumerate Maximal Cliques (Bron-Kerbosch)
        cliques = []
        
        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P U X
            if P:
                u = next(iter(P))
            elif X:
                u = next(iter(X))
            else:
                return
            
            candidates = [v for v in P if v not in adj[u]]
            
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bronk(set(), set(range(n)), set())
        
        if not cliques:
            results.append([])
            continue
            
        # 3. Solve Set Cover
        # Map clique index to set of vertices
        m = len(cliques)
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        best_solution = None
        best_size = float('inf')
        
        # Try multiple random restarts for tie-breaking in greedy
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline - safety_margin:
                break
                
            # Greedy Set Cover with random tie-breaking
            is_covered = [False] * m
            uncovered_count = m
            # cover_count[v] = number of uncovered cliques containing v
            cover_count = [0] * n
            for v in range(n):
                for c_idx in vertex_to_cliques[v]:
                    cover_count[v] += 1
            
            solution = []
            
            while uncovered_count > 0:
                max_cov = -1
                best_v = -1
                # Find vertices with max cover_count
                # To allow randomization, we can pick randomly among max
                candidates = []
                for v in range(n):
                    if cover_count[v] > max_cov:
                        max_cov = cover_count[v]
                        candidates = [v]
                    elif cover_count[v] == max_cov and max_cov > 0:
                        candidates.append(v)
                
                if not candidates:
                    break # Should not happen if uncovered_count > 0 and graph is valid
                    
                v = random.choice(candidates)
                solution.append(v)
                
                # Update coverage
                for c_idx in vertex_to_cliques[v]:
                    if not is_covered[c_idx]:
                        is_covered[c_idx] = True
                        uncovered_count -= 1
                        # Decrement cover_count for all vertices in this clique
                        for u in cliques[c_idx]:
                            cover_count[u] -= 1
            
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = solution.copy()
                
            # Local Search: Remove redundant vertices
            # We need to track support counts
            # support_count[c] = number of vertices in current solution that are in clique c
            support_count = [0] * m
            for v in best_solution:
                for c_idx in vertex_to_cliques[v]:
                    support_count[c_idx] += 1
            
            improved = True
            while improved:
                improved = False
                # Try removing each vertex
                for v in best_solution[:]: # Iterate over a copy
                    # Check if v is critical
                    can_remove = True
                    for c_idx in vertex_to_cliques[v]:
                        if support_count[c_idx] == 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        # Remove v
                        best_solution.remove(v)
                        for c_idx in vertex_to_cliques[v]:
                            support_count[c_idx] -= 1
                        improved = True
                        if time.time() > deadline - safety_margin:
                            break
                
                if time.time() > deadline - safety_margin:
                    break
            
            if len(best_solution) < best_size:
                best_size = len(best_solution)
                # best_solution is already updated in place, but we need to ensure we track the best
                # Actually, the local search modifies best_solution directly.
                # We should probably keep a separate variable for the best found so far.
                pass # The logic above updates best_solution. We just need to ensure we don't lose a better one from a previous restart.
                # Since we update best_solution in place during local search, and we only keep it if it was the best from greedy,
                # we might be overwriting a good solution with a worse one if the local search fails to improve?
                # No, local search only removes vertices, so size never increases.
                # But we need to compare against the global best.
                
        # The variable best_solution holds the result of the last restart that was the "best" at the time.
        # However, if a later restart produced a smaller greedy solution, best_solution was updated.
        # If a later restart produced a larger greedy solution, best_solution was NOT updated in the `if len(solution) < best_size` block.
        # BUT, the local search block modifies `best_solution` directly!
        # This is a bug in the logic above. I need to be more careful.
        
        # Let's restructure the loop to be safer.
        
        # Reset for correct implementation
        best_solution = None
        best_size = float('inf')
        
        for r in range(num_restarts):
            if time.time() > deadline - safety_margin:
                break
                
            is_covered = [False] * m
            uncovered_count = m
            cover_count = [0] * n
            for v in range(n):
                for c_idx in vertex_to_cliques[v]:
                    cover_count[v] += 1
            
            solution = []
            
            while uncovered_count > 0:
                max_cov = -1
                candidates = []
                for v in range(n):
                    if cover_count[v] > max_cov:
                        max_cov = cover_count[v]
                        candidates = [v]
                    elif cover_count[v] == max_cov and max_cov > 0:
                        candidates.append(v)
                
                if not candidates:
                    break
                    
                v = random.choice(candidates)
                solution.append(v)
                
                for c_idx in vertex_to_cliques[v]:
                    if not is_covered[c_idx]:
                        is_covered[c_idx] = True
                        uncovered_count -= 1
                        for u in cliques[c_idx]:
                            cover_count[u] -= 1
            
            # Local Search
            support_count = [0] * m
            for v in solution:
                for c_idx in vertex_to_cliques[v]:
                    support_count[c_idx] += 1
            
            improved = True
            while improved:
                improved = False
                for v in solution[:]:
                    can_remove = True
                    for c_idx in vertex_to_cliques[v]:
                        if support_count[c_idx] == 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        solution.remove(v)
                        for c_idx in vertex_to_cliques[v]:
                            support_count[c_idx] -= 1
                        improved = True
                        if time.time() > deadline - safety_margin:
                            break
                
                if time.time() > deadline - safety_margin:
                    break
            
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = solution.copy()
        
        if best_solution is None:
            # Fallback: return all vertices involved in edges
            touched = {x for e in edges for x in e}
            best_solution = sorted(touched)
            
        results.append(sorted(best_solution))
        
        if time.time() > deadline - safety_margin:
            # If we ran out of time, return what we have so far (might be incomplete list of 3)
            # But we must return 3 lists. If we didn't finish all 3, we need to fill the rest with valid (but large) solutions.
            # Since we process sequentially, if we break, we need to handle the remaining instances.
            break
    
    # If we broke out early, fill remaining results with a valid but potentially suboptimal solution
    while len(results) < 3:
        # Get the next instance parameters
        idx = len(results)
        n, p, seed = instances[idx]
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        touched = {x for e in edges for x in e}
        results.append(sorted(touched))
        
    return results
