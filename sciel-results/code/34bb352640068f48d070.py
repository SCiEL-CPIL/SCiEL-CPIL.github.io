import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj, P, X, C, maximal_cliques):
        if not P and not X:
            if len(C) >= 2:
                maximal_cliques.append(frozenset(C))
            return
        
        P_union_X = P | X
        if P_union_X:
            # Pivot selection: choose vertex u in P union X with max degree in P
            pivot = None
            max_inter = -1
            for v in P_union_X:
                inter = len(P & adj[v])
                if inter > max_inter:
                    max_inter = inter
                    pivot = v
            
            # Branch on P \ N(pivot)
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                new_P = P & adj[v]
                new_X = X & adj[v]
                new_C = C | {v}
                bron_kerbosch(adj, new_P, new_X, new_C, maximal_cliques)
                P.remove(v)
                X.add(v)

    def compute_disjoint_clique_lb(maximal_cliques):
        """
        Compute a lower bound on hitting set size using maximum matching
        in the clique-intersection graph (where edges connect cliques sharing a vertex).
        The maximum number of pairwise disjoint cliques is a lower bound.
        We approximate this with a greedy matching on the intersection graph.
        """
        n_cliques = len(maximal_cliques)
        if n_cliques <= 1:
            return 0
        
        # Build intersection graph: edge between i,j if cliques i and j share a vertex
        # For efficiency, use vertex_cliques mapping
        # Two cliques are disjoint if no vertex is in both
        
        # Greedy matching: repeatedly pick a pair of disjoint cliques
        # Use a simple greedy approach: sort cliques by size (ascending), 
        # then greedily match
        
        # First, compute for each clique which other cliques it intersects with
        # Use bitsets for efficiency if n_cliques is small enough
        
        # For our problem sizes (typically < 100 maximal cliques), we can use sets
        clique_vertices = [set(c) for c in maximal_cliques]
        
        # Greedy maximum matching approximation
        matched = [False] * n_cliques
        matching_size = 0
        
        # Try multiple orderings to get a better matching
        import random as _random
        rng = _random.Random(42)
        
        for _ in range(10):  # Try 10 random orderings
            order = list(range(n_cliques))
            rng.shuffle(order)
            
            temp_matched = [False] * n_cliques
            temp_matching = 0
            
            for i in order:
                if temp_matched[i]:
                    continue
                # Find an unmatched clique j that is disjoint from i
                for j in order:
                    if j <= i or temp_matched[j]:
                        continue
                    if clique_vertices[i].isdisjoint(clique_vertices[j]):
                        temp_matched[i] = True
                        temp_matched[j] = True
                        temp_matching += 1
                        break
            
            if temp_matching > matching_size:
                matching_size = temp_matching
        
        return matching_size

    def find_min_transversal(n, edges):
        if not edges:
            return []
        
        adj = build_adj(n, edges)
        maximal_cliques = []
        # Bron-Kerbosch
        bron_kerbosch(adj, set(range(n)), set(), set(), maximal_cliques)
        
        if not maximal_cliques:
            return []

        # Precompute: for each vertex, which cliques does it hit?
        # This allows quick updates.
        n_cliques = len(maximal_cliques)
        # Map each clique to an index
        clique_idx = {c: i for i, c in enumerate(maximal_cliques)}
        
        # For each vertex, list of clique indices it hits
        vertex_cliques = [[] for _ in range(n)]
        for i, c in enumerate(maximal_cliques):
            for v in c:
                vertex_cliques[v].append(i)
        
        # Compute lower bound using disjoint cliques
        lb_disjoint = compute_disjoint_clique_lb(maximal_cliques)
        
        # Greedy initialization
        unhit_mask = (1 << n_cliques) - 1  # All cliques unhit
        transversal = set()
        
        while unhit_mask:
            if time.time() > deadline:
                break
            # Count frequency of each vertex in unhit cliques
            freq = [0] * n
            # Iterate through unhit cliques using bitmask
            m = unhit_mask
            while m:
                # Get lowest set bit
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                for v in maximal_cliques[idx]:
                    freq[v] += 1
            
            if not any(freq):
                break
                
            best_v = None
            best_score = None
            for v in range(n):
                f = freq[v]
                if f == 0:
                    continue
                # Score: frequency is primary. Degree is secondary.
                score = (f, len(adj[v]), -v)
                if best_score is None or score > best_score:
                    best_score = score
                    best_v = v
            
            transversal.add(best_v)
            
            # Remove all cliques containing best_v from unhit_mask
            for ci in vertex_cliques[best_v]:
                unhit_mask &= ~(1 << ci)

        # Ensure validity if broken by time
        if unhit_mask:
            m = unhit_mask
            while m:
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                transversal.add(next(iter(maximal_cliques[idx])))
        
        # Greedy removal
        changed = True
        while changed:
            changed = False
            for v in list(transversal):
                if time.time() > deadline:
                    return sorted(transversal)
                redundant = True
                for ci in vertex_cliques[v]:
                    c = maximal_cliques[ci]
                    others = c - {v}
                    if not (others & transversal):
                        redundant = False
                        break
                if redundant:
                    transversal.remove(v)
                    changed = True
        
        # Branch and Bound for minimum hitting set
        # Use recursive search with pruning
        best_size = len(transversal)
        best_solution = set(transversal)
        
        # Precompute for each vertex: bitmask of cliques it hits
        vertex_clique_mask = [0] * n
        for i, c in enumerate(maximal_cliques):
            for v in c:
                vertex_clique_mask[v] |= (1 << i)
        
        # Precompute clique sizes
        clique_sizes = [len(c) for c in maximal_cliques]
        
        # For the disjoint clique lower bound in B&B, we need to track which cliques are "active"
        # and compute a lower bound based on pairwise disjointness among unhit cliques
        
        def compute_lb_for_unhit(unhit_mask):
            """Compute lower bound for hitting all cliques in unhit_mask."""
            if unhit_mask == 0:
                return 0
            
            # Simple lower bound: at least 1
            # Better: use disjoint clique matching on the subgraph of unhit cliques
            # For efficiency, use a greedy approach on the unhit cliques
            
            # Get list of unhit clique indices
            unhit_indices = []
            m = unhit_mask
            while m:
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                unhit_indices.append(idx)
            
            k = len(unhit_indices)
            if k <= 1:
                return k
            
            # Greedy matching on the subgraph induced by unhit cliques
            # Two cliques are disjoint if they share no vertex
            clique_vertices = [maximal_cliques[i] for i in unhit_indices]
            
            matched = [False] * k
            matching_size = 0
            
            # Try a few orderings
            import random as _random
            rng = _random.Random(123)
            
            for _ in range(5):
                order = list(range(k))
                rng.shuffle(order)
                
                temp_matched = [False] * k
                temp_matching = 0
                
                for i in order:
                    if temp_matched[i]:
                        continue
                    for j in order:
                        if j <= i or temp_matched[j]:
                            continue
                        if clique_vertices[i].isdisjoint(clique_vertices[j]):
                            temp_matched[i] = True
                            temp_matched[j] = True
                            temp_matching += 1
                            break
                
                if temp_matching > matching_size:
                    matching_size = temp_matching
            
            return matching_size
        
        def search(unhit_mask, current_size, remaining_budget):
            nonlocal best_size, best_solution
            
            if current_size >= best_size:
                return
            
            if not unhit_mask:
                # This shouldn't happen in our formulation since we track unhit mask
                return
            
            if time.time() > remaining_budget:
                return
            
            # Compute lower bound
            lb = compute_lb_for_unhit(unhit_mask)
            if current_size + lb >= best_size:
                return
            
            if current_size + 1 >= best_size:
                return
            
            # Find a clique to branch on (choose one with fewest candidate vertices)
            # Get list of unhit cliques
            unhit_list = []
            m = unhit_mask
            while m:
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                unhit_list.append(idx)
            
            if not unhit_list:
                return
            
            # Choose clique with minimum number of vertices
            best_clique_idx = None
            best_candidates = None
            min_cand_count = float('inf')
            
            for ci in unhit_list:
                c = maximal_cliques[ci]
                candidates = list(c)
                if len(candidates) < min_cand_count:
                    min_cand_count = len(candidates)
                    best_clique_idx = ci
                    best_candidates = candidates
                    if min_cand_count <= 1:
                        break
            
            if not best_candidates:
                return
            
            # Sort candidates by frequency in unhit cliques (descending)
            def candidate_score(v):
                # Count how many unhit cliques contain v
                mask = vertex_clique_mask[v]
                count = bin(mask & unhit_mask).count('1')
                return -count
            
            best_candidates.sort(key=candidate_score)
            
            for v in best_candidates:
                if time.time() > remaining_budget:
                    return
                
                # Update unhit mask: remove all cliques containing v
                new_unhit = unhit_mask & ~vertex_clique_mask[v]
                
                # If this leads to a complete solution
                if new_unhit == 0:
                    if current_size + 1 < best_size:
                        best_size = current_size + 1
                        # We need to reconstruct the solution, but for now just track size
                        # For simplicity, we'll use a different approach: track the actual set
                        pass
                else:
                    search(new_unhit, current_size + 1, remaining_budget)
        
        # Run B&B with time limit - track actual solution
        bb_start = time.time()
        bb_budget = min(deadline - bb_start, 40)  # Use up to 40s for B&B
        
        # Redefine search to track the actual solution
        best_solution_set = set(transversal)
        
        def search_with_solution(unhit_mask, current_set, current_size, remaining_budget):
            nonlocal best_size, best_solution_set
            
            if current_size >= best_size:
                return
            
            if not unhit_mask:
                if current_size < best_size:
                    best_size = current_size
                    best_solution_set = set(current_set)
                return
            
            if time.time() > remaining_budget:
                return
            
            # Compute lower bound
            lb = compute_lb_for_unhit(unhit_mask)
            if current_size + lb >= best_size:
                return
            
            if current_size + 1 >= best_size:
                return
            
            # Find a clique to branch on
            unhit_list = []
            m = unhit_mask
            while m:
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                unhit_list.append(idx)
            
            if not unhit_list:
                return
            
            # Choose clique with minimum number of vertices
            best_clique_idx = None
            best_candidates = None
            min_cand_count = float('inf')
            
            for ci in unhit_list:
                c = maximal_cliques[ci]
                candidates = list(c)
                if len(candidates) < min_cand_count:
                    min_cand_count = len(candidates)
                    best_clique_idx = ci
                    best_candidates = candidates
                    if min_cand_count <= 1:
                        break
            
            if not best_candidates:
                return
            
            # Sort candidates by frequency in unhit cliques (descending)
            def candidate_score(v):
                mask = vertex_clique_mask[v]
                count = bin(mask & unhit_mask).count('1')
                return -count
            
            best_candidates.sort(key=candidate_score)
            
            for v in best_candidates:
                if time.time() > remaining_budget:
                    return
                
                new_set = current_set | {v}
                new_unhit = unhit_mask & ~vertex_clique_mask[v]
                
                search_with_solution(new_unhit, new_set, current_size + 1, remaining_budget)
        
        if bb_budget > 0:
            search_with_solution((1 << n_cliques) - 1, set(), 0, bb_start + bb_budget)
        
        if best_size < len(transversal):
            transversal = best_solution_set
        else:
            # Try pair replacement
            improved = True
            while improved and time.time() < deadline:
                improved = False
                trans_list = list(transversal)
                for i in range(len(trans_list)):
                    for j in range(i + 1, len(trans_list)):
                        v1 = trans_list[i]
                        v2 = trans_list[j]
                        
                        candidate = transversal - {v1, v2}
                        
                        unhit_after_removal = []
                        for c in maximal_cliques:
                            if not (c & candidate):
                                unhit_after_removal.append(c)
                        
                        if not unhit_after_removal:
                            transversal = candidate
                            improved = True
                            break
                        
                        potential_w = set()
                        for c in unhit_after_removal:
                            potential_w.update(c)
                        
                        found_w = None
                        for w in potential_w:
                            if w == v1 or w == v2:
                                continue
                            ok = True
                            for c in unhit_after_removal:
                                if w not in c:
                                    ok = False
                                    break
                            if ok:
                                found_w = w
                                break
                        
                        if found_w is not None:
                            transversal = candidate | {found_w}
                            improved = True
                            break
                    
                    if improved:
                        break
            
            # Final greedy removal
            changed = True
            while changed:
                changed = False
                for v in list(transversal):
                    if time.time() > deadline:
                        break
                    redundant = True
                    for c in maximal_cliques:
                        if v in c:
                            others = c - {v}
                            if not (others & transversal):
                                redundant = False
                                break
                    if redundant:
                        transversal.remove(v)
                        changed = True

        return sorted(transversal)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        sol = find_min_transversal(n, edges)
        results.append(sol)
        if time.time() > deadline:
             # If we ran out of time, ensure we return what we have
             # But we need to return 3 lists. 
             # If we didn't finish all 3, we should return best effort or empty?
             # Better to return valid sets for completed ones and try fast heuristic for others?
             # Actually, 120s is plenty for N=40. The loop is very fast.
             pass

    return results
