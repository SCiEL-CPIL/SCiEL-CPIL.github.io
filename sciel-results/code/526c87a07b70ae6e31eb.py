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

    def find_min_transversal(n, edges):
        if not edges:
            return []
        
        adj = build_adj(n, edges)
        maximal_cliques = []
        # Bron-Kerbosch
        bron_kerbosch(adj, set(range(n)), set(), set(), maximal_cliques)
        
        if not maximal_cliques:
            return []

        n_cliques = len(maximal_cliques)
        
        # Map each clique to an index
        # Precompute: for each vertex, which cliques does it hit?
        vertex_cliques = [[] for _ in range(n)]
        for i, c in enumerate(maximal_cliques):
            for v in c:
                vertex_cliques[v].append(i)
        
        # Precompute vertex_clique_mask: bitmask of cliques each vertex hits
        vertex_clique_mask = [0] * n
        for i in range(n_cliques):
            bit = 1 << i
            for v in maximal_cliques[i]:
                vertex_clique_mask[v] |= bit
        
        # Greedy initialization with bitmask operations
        unhit_mask = (1 << n_cliques) - 1  # All cliques unhit
        transversal = set()
        
        while unhit_mask:
            if time.time() > deadline:
                break
            # Count frequency of each vertex in unhit cliques using bitmask
            freq = [0] * n
            # For each vertex, count how many unhit cliques it appears in
            for v in range(n):
                if vertex_clique_mask[v] == 0:
                    continue
                # Count bits in (vertex_clique_mask[v] & unhit_mask)
                x = vertex_clique_mask[v] & unhit_mask
                while x:
                    x &= x - 1
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
            unhit_mask &= ~vertex_clique_mask[best_v]

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
                # Check if v is needed: is there a clique containing v where no other transversal vertex appears?
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
        
        # Precompute for each clique: list of vertices (for quick access)
        clique_vertices = [list(c) for c in maximal_cliques]
        
        # Precompute: for each vertex, the bitmask of cliques it hits (already have vertex_clique_mask)
        
        def greedy_lower_bound(unhit_mask):
            """Compute a lower bound on the number of additional vertices needed.
            Use a greedy packing: find disjoint unhit cliques. Each requires a distinct vertex."""
            if not unhit_mask:
                return 0
            
            count = 0
            remaining = unhit_mask
            # Greedily pick cliques that are disjoint from already chosen ones
            # To be fast, just count bits in a simplified way: use frequency-based heuristic
            # Better: iteratively find a clique, remove all cliques sharing vertices with it
            
            # Simple approach: count unhit cliques, divide by max clique size (weak but fast)
            # Or: use a greedy set cover lower bound
            
            # Fast lower bound: at least ceil(number of unhit cliques / max_clique_size)
            # But this is weak. Let's use a better one:
            
            # Greedy packing: repeatedly find a clique, then remove all cliques that intersect it
            # This gives a valid lower bound (each packed clique needs a different vertex)
            
            packed = 0
            rem = unhit_mask
            while rem:
                # Pick the first unhit clique
                lowbit = rem & (-rem)
                idx = lowbit.bit_length() - 1
                rem ^= lowbit
                
                packed += 1
                
                # Remove all cliques that share any vertex with clique idx
                c = maximal_cliques[idx]
                # Compute union of vertex_clique_mask for all v in c
                shared_mask = 0
                for v in c:
                    shared_mask |= vertex_clique_mask[v]
                rem &= ~shared_mask
            
            return packed
        
        def search(unhit_mask, current_set_mask, current_size, remaining_budget):
            nonlocal best_size, best_solution
            
            if current_size >= best_size:
                return
            
            if not unhit_mask:
                best_size = current_size
                # Reconstruct solution from current_set_mask
                sol = set()
                m = current_set_mask
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    sol.add(idx)
                    m ^= lb
                best_solution = sol
                return
            
            if time.time() > remaining_budget:
                return
            
            # Lower bound: greedy packing
            lb = greedy_lower_bound(unhit_mask)
            if current_size + lb >= best_size:
                return
            
            # Find a clique to branch on (choose one with fewest candidate vertices not in current_set)
            min_cand_count = float('inf')
            best_clique_idx = -1
            best_candidates = []
            
            m = unhit_mask
            while m:
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                
                c = maximal_cliques[idx]
                # Count candidates in c that are NOT in current_set
                cand_count = 0
                for v in c:
                    if not (current_set_mask >> v) & 1:
                        cand_count += 1
                
                if cand_count < min_cand_count:
                    min_cand_count = cand_count
                    best_clique_idx = idx
                    best_candidates = [v for v in c if not (current_set_mask >> v) & 1]
                    if min_cand_count <= 1:
                        break
            
            if not best_candidates:
                return
            
            # Sort candidates by frequency in unhit cliques (descending)
            # Use vertex_clique_mask to compute frequency quickly
            def candidate_score(v):
                x = vertex_clique_mask[v] & unhit_mask
                # Count bits
                cnt = 0
                while x:
                    x &= x - 1
                    cnt += 1
                return -cnt  # Negative for descending sort
            
            best_candidates.sort(key=candidate_score)
            
            for v in best_candidates:
                if time.time() > remaining_budget:
                    return
                
                new_set_mask = current_set_mask | (1 << v)
                new_unhit = unhit_mask & ~vertex_clique_mask[v]
                
                search(new_unhit, new_set_mask, current_size + 1, remaining_budget)
        
        # Run B&B with time limit
        bb_start = time.time()
        # Allocate more time to B&B: use up to 80% of remaining time
        remaining = deadline - bb_start
        bb_budget = min(remaining, 90)  # Use up to 90s for B&B
        if bb_budget > 0:
            # Start with empty set, search from scratch
            search((1 << n_cliques) - 1, 0, 0, bb_start + bb_budget)
        
        if best_size < len(transversal):
            transversal = best_solution
        else:
            # Local search: try single vertex removal
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for v in list(transversal):
                    if time.time() > deadline:
                        break
                    candidate = transversal - {v}
                    # Check if candidate still hits all cliques
                    valid = True
                    for ci in vertex_cliques[v]:
                        c = maximal_cliques[ci]
                        if not (c & candidate):
                            valid = False
                            break
                    if valid:
                        transversal = candidate
                        improved = True
                        break
            
            # Local search: try pair replacement
            improved = True
            while improved and time.time() < deadline:
                improved = False
                trans_list = list(transversal)
                for i in range(len(trans_list)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(trans_list)):
                        if time.time() > deadline:
                            break
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
