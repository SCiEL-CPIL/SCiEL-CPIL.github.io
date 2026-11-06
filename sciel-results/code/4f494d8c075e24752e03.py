import time
import math
import random as _random

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n, deadline):
        cliques = []
        
        def bk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])
            for v in candidates:
                if time.time() > deadline:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = bron_kerbosch(adj, n, time_budget * 0.3)
        if not cliques:
            return []
        
        m = len(cliques)
        vertex_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(ci)
        
        active_vertices = [v for v in range(n) if vertex_cliques[v]]
        if not active_vertices:
            return []
        
        # Dominance pruning
        active_vertices_sorted = sorted(active_vertices, key=lambda v: -len(vertex_cliques[v]))
        keep_vertices = set()
        for v in active_vertices_sorted:
            dominated = False
            for u in keep_vertices:
                if vertex_cliques[v].issubset(vertex_cliques[u]):
                    dominated = True
                    break
            if not dominated:
                keep_vertices.add(v)
        
        candidates = list(keep_vertices)
        if not candidates:
            return []
        
        num_candidates = len(candidates)
        num_cliques = m
        
        # Bitmask representation
        candidate_mask = [0] * num_candidates
        for i, v in enumerate(candidates):
            mask = 0
            for ci in vertex_cliques[v]:
                mask |= (1 << ci)
            candidate_mask[i] = mask
        
        clique_mask = [0] * num_cliques
        for i in range(num_candidates):
            mask = candidate_mask[i]
            while mask:
                lb = mask & (-mask)
                ci = lb.bit_length() - 1
                clique_mask[ci] |= (1 << i)
                mask ^= lb
        
        all_cliques_mask = (1 << num_cliques) - 1
        
        # Greedy initial solution
        def greedy_cover():
            selected = 0
            unc = all_cliques_mask
            while unc:
                best_idx = -1
                best_cov = 0
                for i in range(num_candidates):
                    if selected & (1 << i):
                        continue
                    cov = (unc & candidate_mask[i]).bit_count()
                    if cov > best_cov:
                        best_cov = cov
                        best_idx = i
                if best_idx == -1:
                    break
                selected |= (1 << best_idx)
                unc &= ~candidate_mask[best_idx]
            return selected
        
        init_sel = greedy_cover()
        best_size = bin(init_sel).count('1')
        best_selected = init_sel
        
        if best_size == 0:
            return []
        
        def sel_to_vertices(sel):
            verts = []
            for i in range(num_candidates):
                if sel & (1 << i):
                    verts.append(candidates[i])
            return verts
        
        best_vertices = sel_to_vertices(best_selected)
        
        # Lower bound: greedy packing of uncovered cliques
        def packing_lb(uncovered):
            if not uncovered:
                return 0
            remaining = uncovered
            count = 0
            used = 0
            while remaining:
                best_ci = -1
                min_avail = float('inf')
                r = remaining
                while r:
                    lb = r & (-r)
                    ci = lb.bit_length() - 1
                    avail = (clique_mask[ci] & ~used).bit_count()
                    if avail < min_avail:
                        min_avail = avail
                        best_ci = ci
                        if min_avail <= 1:
                            break
                    r ^= lb
                if best_ci == -1 or min_avail == 0:
                    break
                count += 1
                used |= clique_mask[best_ci]
                remaining &= ~(1 << best_ci)
            return count
        
        # Branch and Bound
        def branch_and_bound(selected, uncovered):
            nonlocal best_size, best_vertices, best_selected
            
            if not uncovered:
                sz = bin(selected).count('1')
                if sz < best_size:
                    best_size = sz
                    best_selected = selected
                    best_vertices = sel_to_vertices(selected)
                return
            
            sz = bin(selected).count('1')
            if sz >= best_size:
                return
            
            if time.time() > time_budget:
                return
            
            # Lower bound 1: ceil(unc / max_cov)
            max_cov = 0
            for i in range(num_candidates):
                if selected & (1 << i):
                    continue
                cov = (uncovered & candidate_mask[i]).bit_count()
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return
            
            lb1 = sz + math.ceil(bin(uncovered).count('1') / max_cov)
            
            # Lower bound 2: packing
            lb2 = sz + packing_lb(uncovered)
            
            lb = max(lb1, lb2)
            if lb >= best_size:
                return
            
            # Choose branch clique: fewest available candidates
            target_ci = -1
            min_avail = float('inf')
            r = uncovered
            while r:
                lb_bit = r & (-r)
                ci = lb_bit.bit_length() - 1
                avail = (clique_mask[ci] & ~selected).bit_count()
                if avail < min_avail:
                    min_avail = avail
                    target_ci = ci
                    if min_avail <= 1:
                        break
                r ^= lb_bit
            
            if target_ci == -1 or min_avail == 0:
                return
            
            avail_candidates_mask = clique_mask[target_ci] & ~selected
            
            # Order by coverage descending
            avail_list = []
            am = avail_candidates_mask
            while am:
                lb_bit = am & (-am)
                i = lb_bit.bit_length() - 1
                cov = (uncovered & candidate_mask[i]).bit_count()
                avail_list.append((cov, i))
                am ^= lb_bit
            
            avail_list.sort(key=lambda x: -x[0])
            
            for _, i in avail_list:
                new_selected = selected | (1 << i)
                new_uncovered = uncovered & ~candidate_mask[i]
                branch_and_bound(new_selected, new_uncovered)
        
        branch_and_bound(0, all_cliques_mask)
        
        # Iterative local search: try removals repeatedly
        def local_search_polish():
            nonlocal best_size, best_vertices, best_selected
            improved = True
            while improved:
                improved = False
                if time.time() > time_budget:
                    break
                for i in range(num_candidates):
                    if not (best_selected & (1 << i)):
                        continue
                    new_sel = best_selected & ~(1 << i)
                    covered = 0
                    for j in range(num_candidates):
                        if new_sel & (1 << j):
                            covered |= candidate_mask[j]
                    if covered == all_cliques_mask:
                        best_selected = new_sel
                        best_size -= 1
                        best_vertices = sel_to_vertices(best_selected)
                        improved = True
                        break
            return best_vertices
        
        local_search_polish()
        
        return sorted(best_vertices)

    # Allocate time: give more to larger/harder instances
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    total_budget = DEADLINE - time.time()
    # Weights based on problem size
    weights = [1.0, 1.2, 1.5]
    total_w = sum(weights)
    budgets = [total_budget * w / total_w for w in weights]
    
    results = []
    for (n, p, seed), tb in zip(instances, budgets):
        sol = solve_instance(n, p, seed, start_time + tb)
        results.append(sol)
    
    return results
