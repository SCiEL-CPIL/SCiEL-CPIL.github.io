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
        
        restarts = 40
        
        for r in range(restarts):
            if time.time() > deadline - 10.0:
                break
                
            uncovered = set(range(m))
            solution = set()
            
            # Greedy Initialization
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
                    
            # Local Search
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                
                # 1. Redundancy Removal
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

                if not improved:
                    # 2. 1-for-1 Swap
                    verts_to_try = list(solution)
                    random.shuffle(verts_to_try)
                    for v in verts_to_try:
                        if v not in solution:
                            continue
                        solution.remove(v)
                        
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
                            improved = True
                        else:
                            solution.add(v)
                            # Try swap
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
                                best_replace = None
                                for u in range(n):
                                    if u in solution:
                                        continue
                                    covers_all = True
                                    for ci in uncovered_by_v:
                                        if u not in cliques[ci]:
                                            covers_all = False
                                            break
                                    if covers_all:
                                        best_replace = u
                                        break
                                
                                if best_replace is not None:
                                    solution.add(best_replace)
                                    improved = True
                                    
                if not improved:
                    # 3. 2-for-1 Exchange
                    sol_list = list(solution)
                    random.shuffle(sol_list)
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
                            
                            # Try removing v1 and v2
                            solution.remove(v1)
                            solution.remove(v2)
                            
                            # Find uncovered cliques
                            uncovered_set = set()
                            for ci in vertex_to_cliques[v1]:
                                covered = False
                                for u in cliques[ci]:
                                    if u in solution:
                                        covered = True
                                        break
                                if not covered:
                                    uncovered_set.add(ci)
                            for ci in vertex_to_cliques[v2]:
                                if ci in uncovered_set:
                                    continue
                                covered = False
                                for u in cliques[ci]:
                                    if u in solution:
                                        covered = True
                                        break
                                if not covered:
                                    uncovered_set.add(ci)
                            
                            if not uncovered_set:
                                improved = True
                                break
                            
                            # Find a vertex w that covers all in uncovered_set
                            best_w = None
                            for w in range(n):
                                if w in solution:
                                    continue
                                covers_all = True
                                for ci in uncovered_set:
                                    if w not in cliques[ci]:
                                        covers_all = False
                                        break
                                if covers_all:
                                    best_w = w
                                    break
                            
                            if best_w is not None:
                                solution.add(best_w)
                                improved = True
                                break
                            else:
                                # Restore
                                solution.add(v1)
                                solution.add(v2)
                        if improved:
                            break

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
