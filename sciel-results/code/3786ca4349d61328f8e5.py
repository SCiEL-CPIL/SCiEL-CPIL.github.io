import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            P_X = P.union(X)
            if not P_X:
                return
            
            max_deg = -1
            pivot = None
            for u in P_X:
                deg = len(P.intersection(adj[u]))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            for v in list(P - adj[pivot]):
                bk(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
                P.remove(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
            
        m = len(cliques)
        
        # Precompute mappings
        vertex_to_cliques = [[] for _ in range(n)]
        clique_to_vertices = [set(c) for c in cliques]
        
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        best_solution = None
        best_size = float('inf')
        
        # Number of restarts
        restarts = 100
        
        for r in range(restarts):
            if time.time() > deadline - 5.0:
                break
                
            uncovered = set(range(m))
            solution = set()
            
            # Greedy Phase
            while uncovered:
                best_v = -1
                best_count = -1
                # To randomize ties, we can add noise or just pick random among max
                # For efficiency, we scan all vertices
                max_count = 0
                candidates = []
                for v in range(n):
                    if v in solution:
                        continue
                    count = 0
                    for ci in vertex_to_cliques[v]:
                        if ci in uncovered:
                            count += 1
                    if count > max_count:
                        max_count = count
                        candidates = [v]
                    elif count == max_count and count > 0:
                        candidates.append(v)
                        
                if not candidates:
                    break # Should not happen
                    
                v = random.choice(candidates)
                solution.add(v)
                
                for ci in vertex_to_cliques[v]:
                    uncovered.discard(ci)
                    
            # Local Search Phase
            # We try to minimize the size of the solution
            improved = True
            while improved:
                improved = False
                
                # 1. Redundancy Removal
                verts_to_check = list(solution)
                random.shuffle(verts_to_check)
                for v in verts_to_check:
                    if v not in solution:
                        continue
                    # Check if v is redundant
                    redundant = True
                    for ci in vertex_to_cliques[v]:
                        # Check if clique ci is covered by other vertices in solution
                        covered = False
                        for u in clique_to_vertices[ci]:
                            if u != v and u in solution:
                                covered = True
                                break
                        if not covered:
                            redundant = False
                            break
                    if redundant:
                        solution.remove(v)
                        improved = True
                        # Restart check from beginning? Or just continue?
                        # Continuing is usually fine, but restarting ensures we don't miss cascading redundancies
                        # However, for speed, we'll just continue and let the outer loop catch it if needed
                        # Actually, removing one might make another redundant. 
                        # To be safe and fast, we can break and restart the while loop
                        break 
                
                if improved:
                    continue # Restart the while loop to check redundancies again
                
                # 2. 1-for-1 Swap (to escape local optima for future reductions)
                # If we swap, size stays same, but might allow a 2-for-1 or redundancy later
                # We limit this to avoid infinite loops of swaps
                # For now, let's prioritize 2-for-1 which directly reduces size
                
                # 3. 2-for-1 Swap
                # Try to replace two vertices {v1, v2} with one vertex u
                # This reduces size by 1
                sol_list = list(solution)
                random.shuffle(sol_list)
                found_improvement = False
                for i in range(len(sol_list)):
                    if time.time() > deadline - 5.0:
                        break
                    v1 = sol_list[i]
                    if v1 not in solution:
                        continue
                    for j in range(i + 1, len(sol_list)):
                        v2 = sol_list[j]
                        if v2 not in solution:
                            continue
                        
                        # Temporarily remove v1 and v2
                        solution.remove(v1)
                        solution.remove(v2)
                        
                        # Find all cliques that are now uncovered
                        uncovered_cliques = set()
                        for ci in vertex_to_cliques[v1]:
                            if ci not in uncovered_cliques:
                                # Check if covered by remaining solution
                                covered = False
                                for u in clique_to_vertices[ci]:
                                    if u in solution:
                                        covered = True
                                        break
                                if not covered:
                                    uncovered_cliques.add(ci)
                        
                        for ci in vertex_to_cliques[v2]:
                            if ci not in uncovered_cliques:
                                covered = False
                                for u in clique_to_vertices[ci]:
                                    if u in solution:
                                        covered = True
                                        break
                                if not covered:
                                    uncovered_cliques.add(ci)
                        
                        # Try to find a single vertex u not in solution that covers all uncovered_cliques
                        if uncovered_cliques:
                            best_u = None
                            for u in range(n):
                                if u in solution:
                                    continue
                                # Check if u covers all in uncovered_cliques
                                covers_all = True
                                for ci in uncovered_cliques:
                                    if u not in clique_to_vertices[ci]:
                                        covers_all = False
                                        break
                                if covers_all:
                                    best_u = u
                                    break # First one found is fine, or we could pick best by other metrics
                                    
                            if best_u is not None:
                                solution.add(best_u)
                                improved = True
                                found_improvement = True
                                break
                        else:
                            # If no cliques were uncovered, it means v1 and v2 were redundant together?
                            # This case should have been caught by redundancy removal if they were individually redundant.
                            # If they are jointly redundant (i.e., removing both leaves all cliques covered), 
                            # then we can just remove both.
                            improved = True
                            found_improvement = True
                            break
                            
                        # Restore v1 and v2 if no improvement found
                        solution.add(v1)
                        solution.add(v2)
                        
                    if found_improvement:
                        break
                        
                if found_improvement:
                    continue # Restart local search loop

            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = set(solution)
                
        return sorted(best_solution) if best_solution else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10.0:
            # Fallback: return all touched vertices
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u,v)}
            out.append(sorted(touched))
        else:
            out.append(solve_instance(n, p, seed))
            
    return out
