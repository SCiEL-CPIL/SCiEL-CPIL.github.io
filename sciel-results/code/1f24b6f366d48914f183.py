import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit)

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

    def max_independent_set_bound(cliques, vertex_to_cliques, n):
        """
        Compute a lower bound on the hitting set size using the maximum independent set
        in the clique-intersection graph. Two cliques are adjacent if they share a vertex.
        A maximum independent set gives a set of mutually disjoint cliques, each requiring
        at least one distinct vertex in any hitting set.
        """
        m = len(cliques)
        if m == 0:
            return 0
        
        # Build the complement of the clique-intersection graph
        # In the intersection graph: edge between i,j if cliques[i] & cliques[j] != empty
        # We want max independent set in intersection graph = max clique in complement
        # Complement: edge between i,j if cliques[i] & cliques[j] == empty (disjoint)
        
        # Build adjacency for the "disjointness" graph (complement of intersection)
        # Actually, for max independent set in intersection graph, we need max clique in complement
        # Let's just use Bron-Kerbosch on the intersection graph's complement
        
        # For efficiency with potentially large m, use a greedy heuristic + BK
        # First try greedy coloring bound
        
        # Build intersection graph adjacency
        inter_adj = [set() for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                if cliques[i] & cliques[j]:
                    inter_adj[i].add(j)
                    inter_adj[j].add(i)
        
        # Max independent set in inter_adj = Max clique in complement
        # Complement adj: edge if NOT in inter_adj
        comp_adj = [set() for _ in range(m)]
        for i in range(m):
            for j in range(m):
                if i != j and j not in inter_adj[i]:
                    comp_adj[i].add(j)
        
        # Use Bron-Kerbosch with pivot to find max clique in complement
        best_clique = set()
        
        def bk_max(R, P, X):
            nonlocal best_clique
            if len(R) + len(P) <= len(best_clique):
                return
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = set(R)
                return
            
            P_X = P.union(X)
            if not P_X:
                return
            
            # Pivot selection
            max_deg = -1
            pivot = None
            for u in P_X:
                deg = len(P.intersection(comp_adj[u]))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            for v in list(P - comp_adj[pivot]):
                bk_max(R.union({v}), P.intersection(comp_adj[v]), X.intersection(comp_adj[v]))
                P.remove(v)
                X.add(v)
        
        bk_max(set(), set(range(m)), set())
        return len(best_clique)

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
        
        # Compute lower bound via max independent set
        lb = max_independent_set_bound(cliques, vertex_to_cliques, n)
        
        best_solution = None
        best_size = float('inf')
        
        # Branch and bound hitting set solver
        def branch_and_bound(uncovered, solution, depth, max_depth=20):
            nonlocal best_solution, best_size, deadline
            if time.time() > deadline - 15.0:
                return
            if not uncovered:
                if len(solution) < best_size:
                    best_size = len(solution)
                    best_solution = set(solution)
                return
            if len(solution) >= best_size:
                return
            if depth > max_depth:
                return
            
            # Lower bound: each uncovered clique needs at least one new vertex
            # Use a simple bound: ceil(len(uncovered) / max_coverage)
            max_cov = 0
            for v in range(n):
                if v in solution:
                    continue
                cov = 0
                for ci in vertex_to_cliques[v]:
                    if ci in uncovered:
                        cov += 1
                max_cov = max(max_cov, cov)
            if max_cov == 0:
                return
            bound = len(solution) + math.ceil(len(uncovered) / max_cov)
            if bound >= best_size:
                return
            
            # Greedy lower bound refinement
            # Try to find a good vertex to branch on
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in solution:
                    continue
                count = 0
                for ci in vertex_to_cliques[v]:
                    if ci in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                return
            
            # Branch: include best_v
            new_uncovered = set(uncovered)
            for ci in vertex_to_cliques[best_v]:
                new_uncovered.discard(ci)
            branch_and_bound(new_uncovered, solution | {best_v}, depth + 1, max_depth)
            
            # Branch: exclude best_v (try to cover its cliques with other vertices)
            # Only if there's another vertex that can cover at least one of best_v's cliques
            if len(uncovered) > 1:
                new_uncovered2 = set(uncovered)
                # We need to cover all cliques that best_v covers using other vertices
                # This is complex; instead, just try the next best vertex
                second_v = -1
                second_count = -1
                for v in range(n):
                    if v in solution or v == best_v:
                        continue
                    count = 0
                    for ci in vertex_to_cliques[v]:
                        if ci in uncovered:
                            count += 1
                    if count > second_count:
                        second_count = count
                        second_v = v
                if second_v != -1 and second_count > 0:
                    new_uncovered2 = set(uncovered)
                    for ci in vertex_to_cliques[second_v]:
                        new_uncovered2.discard(ci)
                    branch_and_bound(new_uncovered2, solution | {second_v}, depth + 1, max_depth)
        
        # Greedy set cover with weighted scoring
        def greedy_weighted(uncovered, solution):
            nonlocal best_size, best_solution
            while uncovered:
                best_v = -1
                best_score = -1
                for v in range(n):
                    if v in solution:
                        continue
                    count = 0
                    for ci in vertex_to_cliques[v]:
                        if ci in uncovered:
                            count += 1
                    if count > best_score:
                        best_score = count
                        best_v = v
                if best_v == -1 or best_score == 0:
                    break
                solution.add(best_v)
                for ci in vertex_to_cliques[best_v]:
                    uncovered.discard(ci)
            return solution
        
        def local_search(solution):
            nonlocal deadline
            # Redundancy removal
            improved = True
            while improved and time.time() < deadline - 15.0:
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
            
            # 1-for-1 exchange
            improved2 = True
            while improved2 and time.time() < deadline - 15.0:
                improved2 = False
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
                        improved2 = True
                    else:
                        solution.add(v)
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
                            best_score = 0
                            for u in range(n):
                                if u in solution:
                                    continue
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
                
                # Another redundancy removal pass
                improved3 = True
                while improved3 and time.time() < deadline - 15.0:
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
            
            return solution
        
        # Run branch and bound first to get a good initial solution
        if m <= 200:
            branch_and_bound(set(range(m)), set(), 0, max_depth=15)
        
        # Multiple greedy restarts with local search
        restarts = 30
        
        for r in range(restarts):
            if time.time() > deadline - 10.0:
                break
            
            uncovered = set(range(m))
            solution = set()
            
            # Weighted greedy: prioritize vertices covering rare cliques
            while uncovered:
                best_v = -1
                best_score = -1
                for v in range(n):
                    if v in solution:
                        continue
                    count = 0
                    # Weight by inverse clique frequency for rarity
                    weight_sum = 0.0
                    for ci in vertex_to_cliques[v]:
                        if ci in uncovered:
                            count += 1
                            # Rarer cliques get higher weight
                            weight_sum += 1.0 / len(cliques[ci])
                    if count > 0:
                        score = count + 0.1 * weight_sum
                        if score > best_score:
                            best_score = score
                            best_v = v
                if best_v == -1:
                    break
                solution.add(best_v)
                for ci in vertex_to_cliques[best_v]:
                    uncovered.discard(ci)
            
            solution = local_search(solution)
            
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = set(solution)
            
            # If we hit the lower bound, we're optimal
            if best_size <= lb:
                break
        
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
