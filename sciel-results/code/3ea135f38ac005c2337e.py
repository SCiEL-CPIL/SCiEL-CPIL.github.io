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
        # This is faster than iterating through vertex_cliques lists and checking bits
        vertex_clique_mask = [0] * n
        for i in range(n_cliques):
            bit = 1 << i
            for v in maximal_cliques[i]:
                vertex_clique_mask[v] |= bit
        
        def search(unhit_mask, current_set, current_size, remaining_budget):
            nonlocal best_size, best_solution
            
            if current_size >= best_size:
                return
            
            if not unhit_mask:
                best_size = current_size
                best_solution = set(current_set)
                return
            
            if time.time() > remaining_budget:
                return
            
            # Lower bound: at least 1 if unhit_mask exists
            if current_size + 1 >= best_size:
                return
            
            # Find a clique to branch on (choose one with fewest candidate vertices)
            # We need to iterate through unhit cliques.
            # To optimize, we can just pick the first one or one with min candidates.
            # Since n_cliques is small (< 1000 usually for these sizes), iterating is fine.
            
            # Heuristic: choose clique with minimum number of vertices not in current_set
            # Actually, standard B&B for hitting set often picks the "most constrained" set.
            # Let's pick the clique with the smallest number of available vertices.
            
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
                # Since current_set is small, checking membership is fast.
                # But we can also precompute which vertices are in current_set?
                # No, current_set changes.
                
                cand_count = 0
                # Optimization: if we already have a clique with 1 candidate, we can stop?
                # No, we need the actual candidates for branching.
                
                for v in c:
                    if v not in current_set:
                        cand_count += 1
                
                if cand_count < min_cand_count:
                    min_cand_count = cand_count
                    best_clique_idx = idx
                    # Store candidates
                    best_candidates = [v for v in c if v not in current_set]
                    if min_cand_count <= 1:
                        break # Can't get better than 1
            
            if not best_candidates:
                # Should not happen if unhit_mask is not empty and current_set is valid?
                # If a clique has no candidates, it means all its vertices are in current_set,
                # which means it should have been removed from unhit_mask.
                # This implies a bug or state inconsistency.
                return
            
            # Sort candidates by frequency in unhit cliques (descending) to find good solutions early
            # Precompute freq for these candidates?
            # For small candidate lists, just calculating is fine.
            
            def candidate_score(v):
                count = 0
                mm = unhit_mask
                while mm:
                    lb = mm & (-mm)
                    idx2 = lb.bit_length() - 1
                    mm ^= lb
                    if v in maximal_cliques[idx2]:
                        count += 1
                return -count # Negative for max-heap sort (we want highest freq first)
            
            best_candidates.sort(key=candidate_score)
            
            for v in best_candidates:
                if time.time() > remaining_budget:
                    return
                
                # Add v
                new_set = current_set | {v}
                
                # Update unhit_mask
                # Remove all cliques containing v
                new_unhit = unhit_mask
                for ci in vertex_cliques[v]:
                    new_unhit &= ~(1 << ci)
                
                search(new_unhit, new_set, current_size + 1, remaining_budget)
                
                # Note: We don't remove v from consideration for other branches in this loop
                # because we are branching on a specific clique. We try each vertex in that clique.
                # The recursion handles the state.

        # Run B&B with time limit
        bb_start = time.time()
        bb_budget = min(deadline - bb_start, 30)  # Use up to 30s for B&B
        if bb_budget > 0:
            search((1 << n_cliques) - 1, set(), 0, bb_start + bb_budget)
        
        if best_size < len(transversal):
            transversal = best_solution
        else:
            # Try pair replacement (local search for size reduction)
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
