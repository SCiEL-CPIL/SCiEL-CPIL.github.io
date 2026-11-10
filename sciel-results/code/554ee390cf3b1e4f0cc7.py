import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (150s limit)

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
        
        # Precompute vertex to cliques mapping
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        best_solution = None
        best_size = float('inf')
        
        # Increase restarts since we have more time budget
        restarts = 40
        
        for r in range(restarts):
            if time.time() > deadline - 10.0:
                break
                
            uncovered = set(range(m))
            solution = set()
            
            while uncovered:
                best_v = -1
                best_count = -1
                candidates = []
                for v in range(n):
                    if v in solution:
                        continue
                    count = 0
                    for ci in vertex_to_cliques[v]:
                        if ci in uncovered:
                            count += 1
                    if count > 0:
                        candidates.append((count, v))
                        
                if not candidates:
                    break
                    
                max_count = max(c[0] for c in candidates)
                tied = [c[1] for c in candidates if c[0] == max_count]
                v = random.choice(tied)
                
                solution.add(v)
                
                for ci in vertex_to_cliques[v]:
                    uncovered.discard(ci)
                    
            # Local Search: Redundancy Removal
            improved = True
            while improved:
                improved = False
                verts_to_check = list(solution)
                random.shuffle(verts_to_check)
                for v in verts_to_check:
                    if v not in solution:
                        continue
                    redundant = True
                    for ci in vertex_to_cliques[v]:
                        covered = False
                        for u in cliques[ci]:
                            if u != v and u in solution:
                                covered = True
                                break
                        if not covered:
                            redundant = False
                            break
                    if redundant:
                        solution.remove(v)
                        improved = True
                        
            # 2-exchange local search: try replacing one vertex with another
            improved2 = True
            while improved2 and time.time() < deadline - 10.0:
                improved2 = False
                verts_to_try = list(solution)
                random.shuffle(verts_to_try)
                for v in verts_to_try:
                    if v not in solution:
                        continue
                    # Try removing v
                    solution.remove(v)
                    
                    # Check if all cliques are still covered
                    valid = True
                    for ci in vertex_to_cliques[v]:
                        covered = False
                        for u in cliques[ci]:
                            if u in solution:
                                covered = True
                                break
                        if not covered:
                            valid = False
                            break
                    
                    if valid:
                        improved2 = True
                        # Don't re-add, keep it removed
                    else:
                        # Need to find a replacement or restore
                        solution.add(v)
                        # Try 1-for-1 exchange: find a vertex not in solution that covers all cliques v covers
                        # that aren't covered by solution without v
                        uncovered_by_v = []
                        for ci in vertex_to_cliques[v]:
                            covered = False
                            for u in cliques[ci]:
                                if u in solution:
                                    covered = True
                                    break
                            if not covered:
                                uncovered_by_v.append(ci)
                        
                        if uncovered_by_v:
                            # Find candidate vertices not in solution
                            best_replace = None
                            best_score = 0
                            for u in range(n):
                                if u in solution:
                                    continue
                                # Check if u covers all uncovered_by_v
                                covers_all = True
                                score = 0
                                for ci in uncovered_by_v:
                                    if u in cliques[ci]:
                                        score += 1
                                    else:
                                        covers_all = False
                                        break
                                if covers_all and score > best_score:
                                    best_score = score
                                    best_replace = u
                            
                            if best_replace is not None:
                                solution.remove(v)
                                solution.add(best_replace)
                                improved2 = True
                                # Continue to next v
                            else:
                                # No good replacement, v stays
                                pass
                        else:
                            # v was redundant, already handled
                            pass
                
                # After first pass, do another redundancy removal
                improved3 = True
                while improved3:
                    improved3 = False
                    verts_to_check = list(solution)
                    random.shuffle(verts_to_check)
                    for v in verts_to_check:
                        if v not in solution:
                            continue
                        redundant = True
                        for ci in vertex_to_cliques[v]:
                            covered = False
                            for u in cliques[ci]:
                                if u != v and u in solution:
                                    covered = True
                                    break
                            if not covered:
                                redundant = False
                                break
                        if redundant:
                            solution.remove(v)
                            improved3 = True
                
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = set(solution)
                
        return sorted(best_solution) if best_solution else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 15.0:
            # Fallback: return all touched vertices
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u,v)}
            out.append(sorted(touched))
        else:
            out.append(solve_instance(n, p, seed))
            
    return out
