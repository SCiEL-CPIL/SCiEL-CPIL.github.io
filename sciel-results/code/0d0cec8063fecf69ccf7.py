import time
import math
import random as _random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the seed program."""
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting.
        """
        cliques = []
        
        def bk(R, P, X):
            if time.time() - start_time > DEADLINE:
                return
            
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u in P ∪ X to maximize |P ∩ adj[u]|
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
                if time.time() - start_time > DEADLINE:
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

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Enumerate maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        m = len(cliques)
        
        # Build vertex_cliques: map vertex index to set of clique indices it belongs to
        vertex_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(ci)
        
        # Identify active vertices (those in at least one clique)
        active_vertices = [v for v in range(n) if vertex_cliques[v]]
        
        if not active_vertices:
            return []
        
        # Preprocessing: Dominance Pruning
        # If C_u is a subset of C_v, u is dominated by v (v covers all cliques u covers)
        # So we can safely remove u from the candidate set.
        
        # Sort by number of cliques covered, descending
        active_vertices_sorted = sorted(active_vertices, key=lambda v: -len(vertex_cliques[v]))
        
        keep_vertices = set()
        for v in active_vertices_sorted:
            # Check if v is dominated by any already kept vertex
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

        # Build bitmask representation for faster operations
        # Each candidate i covers a set of cliques, represented as a bitmask
        # Each clique j is covered by a set of candidates, represented as a bitmask
        
        num_candidates = len(candidates)
        num_cliques = m
        
        # Use Python integers as arbitrary-precision bitmasks
        # candidate_mask[i] = bitmask of cliques covered by candidate i
        candidate_mask = [0] * num_candidates
        for i, v in enumerate(candidates):
            mask = 0
            for ci in vertex_cliques[v]:
                mask |= (1 << ci)
            candidate_mask[i] = mask
        
        # clique_mask[j] = bitmask of candidates that cover clique j
        clique_mask = [0] * num_cliques
        for i in range(num_candidates):
            mask = candidate_mask[i]
            while mask:
                lb = mask & (-mask)
                ci = lb.bit_length() - 1
                clique_mask[ci] |= (1 << i)
                mask ^= lb
        
        # Initial greedy upper bound using bitmasks
        def greedy_cover():
            selected = 0  # bitmask of selected candidates
            unc = (1 << num_cliques) - 1  # all cliques uncovered
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
        
        # Convert best_selected to actual vertex indices
        def sel_to_vertices(sel):
            verts = []
            for i in range(num_candidates):
                if sel & (1 << i):
                    verts.append(candidates[i])
            return verts
        
        best_vertices = sel_to_vertices(best_selected)
        
        # Precompute for disjoint-clique lower bound:
        # Find a set of pairwise disjoint cliques (no shared candidates)
        # The size of this set gives a lower bound on the solution size
        
        # Greedy disjoint clique selection for lower bound
        def disjoint_clique_lb(unc):
            """Find a set of pairwise disjoint cliques (in terms of covering candidates) from unc."""
            if not unc:
                return 0
            
            # Greedily pick cliques with fewest available candidates
            remaining = unc
            count = 0
            used_candidates = 0  # bitmask of candidates already "used"
            
            while remaining:
                best_ci = -1
                min_avail = float('inf')
                # Iterate through uncovered cliques
                r = remaining
                while r:
                    lb = r & (-r)
                    ci = lb.bit_length() - 1
                    avail = (clique_mask[ci] & ~used_candidates).bit_count()
                    if avail < min_avail:
                        min_avail = avail
                        best_ci = ci
                    if min_avail == 1:
                        break
                    r ^= lb
                
                if best_ci == -1 or min_avail == 0:
                    break
                
                count += 1
                used_candidates |= clique_mask[best_ci]
                remaining &= ~(1 << best_ci)
            
            return count
        
        # Branch and Bound with bitmask operations
        def branch_and_bound(selected, uncovered, depth=0):
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
            
            if time.time() - start_time > DEADLINE:
                return
            
            # Lower bound 1: max coverage of any single unselected candidate
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
            
            # Lower bound 2: disjoint clique packing
            lb2 = sz + disjoint_clique_lb(uncovered)
            
            lb = max(lb1, lb2)
            if lb >= best_size:
                return
            
            # Choose a clique to branch on
            # Heuristic: Choose the uncovered clique with the fewest available candidates
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
                    if min_avail == 1:
                        break
                r ^= lb_bit
            
            if target_ci == -1 or min_avail == 0:
                return
            
            # Get candidates that can cover target_ci, not yet selected
            avail_candidates_mask = clique_mask[target_ci] & ~selected
            
            # Order by coverage in descending order
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
                branch_and_bound(new_selected, new_uncovered, depth + 1)
        
        branch_and_bound(0, (1 << num_cliques) - 1)
        
        # Local search polish: try to improve by swapping vertices
        def local_search_polish():
            nonlocal best_size, best_vertices, best_selected
            
            if time.time() - start_time > DEADLINE:
                return
            
            # Try removing each selected vertex and see if coverage is still complete
            for i in range(num_candidates):
                if not (best_selected & (1 << i)):
                    continue
                
                # Try removing candidate i
                new_sel = best_selected & ~(1 << i)
                # Check if all cliques are still covered
                covered = 0
                for j in range(num_candidates):
                    if new_sel & (1 << j):
                        covered |= candidate_mask[j]
                
                if covered == ((1 << num_cliques) - 1):
                    # We can remove i
                    best_selected = new_sel
                    best_size -= 1
                    best_vertices = sel_to_vertices(best_selected)
                    # Restart the removal process since the solution changed
                    return True
            
            return False
        
        # Try local search a few times
        for _ in range(10):
            if not local_search_polish():
                break
            if time.time() - start_time > DEADLINE:
                break
        
        return sorted(best_vertices)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)
        
    return results
