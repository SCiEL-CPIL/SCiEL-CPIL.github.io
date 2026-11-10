import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

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

    def find_min_transversal(n, edges, time_budget):
        if not edges:
            return []
        
        adj = build_adj(n, edges)
        maximal_cliques = []
        bron_kerbosch(adj, set(range(n)), set(), set(), maximal_cliques)
        
        if not maximal_cliques:
            return []

        n_cliques = len(maximal_cliques)
        
        # Precompute: for each vertex, which cliques does it hit?
        vertex_cliques = [[] for _ in range(n)]
        for i, c in enumerate(maximal_cliques):
            for v in c:
                vertex_cliques[v].append(i)
        
        # Precompute: for each vertex, bitmask of cliques it hits
        vertex_clique_mask = [0] * n
        for i in range(n_cliques):
            bit = 1 << i
            for v in maximal_cliques[i]:
                vertex_clique_mask[v] |= bit
        
        # Greedy initialization with multiple strategies
        def greedy_init():
            unhit_mask = (1 << n_cliques) - 1
            transversal = set()
            
            while unhit_mask:
                if time.time() > time_budget:
                    break
                
                # Count frequency of each vertex in unhit cliques
                freq = [0] * n
                m = unhit_mask
                while m:
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
                    score = (f, len(adj[v]), -v)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_v = v
                
                transversal.add(best_v)
                for ci in vertex_cliques[best_v]:
                    unhit_mask &= ~(1 << ci)
            
            # Ensure validity
            if unhit_mask:
                m = unhit_mask
                while m:
                    lowbit = m & (-m)
                    idx = lowbit.bit_length() - 1
                    m ^= lowbit
                    transversal.add(next(iter(maximal_cliques[idx])))
            
            return transversal

        def greedy_removal(transversal):
            changed = True
            while changed:
                changed = False
                for v in list(transversal):
                    if time.time() > time_budget:
                        return transversal
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
            return transversal

        def local_search(transversal):
            improved = True
            while improved and time.time() < time_budget:
                improved = False
                trans_list = list(transversal)
                
                # Try removing single vertices
                for v in trans_list:
                    candidate = transversal - {v}
                    unhit = []
                    for c in maximal_cliques:
                        if not (c & candidate):
                            unhit.append(c)
                    if not unhit:
                        transversal = candidate
                        improved = True
                        break
                
                if improved:
                    continue
                
                # Try 2-for-1 exchanges
                for i in range(len(trans_list)):
                    if time.time() > time_budget:
                        return transversal
                    for j in range(i + 1, len(trans_list)):
                        v1 = trans_list[i]
                        v2 = trans_list[j]
                        candidate = transversal - {v1, v2}
                        unhit = []
                        for c in maximal_cliques:
                            if not (c & candidate):
                                unhit.append(c)
                        if not unhit:
                            transversal = candidate
                            improved = True
                            break
                        potential_w = set()
                        for c in unhit:
                            potential_w.update(c)
                        for w in potential_w:
                            if w == v1 or w == v2:
                                continue
                            ok = True
                            for c in unhit:
                                if w not in c:
                                    ok = False
                                    break
                            if ok:
                                transversal = candidate | {w}
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if improved:
                    continue
                
                # Try 3-for-2 exchanges
                for i in range(len(trans_list)):
                    if time.time() > time_budget:
                        return transversal
                    for j in range(i + 1, len(trans_list)):
                        for k in range(j + 1, len(trans_list)):
                            v1, v2, v3 = trans_list[i], trans_list[j], trans_list[k]
                            candidate = transversal - {v1, v2, v3}
                            unhit = []
                            for c in maximal_cliques:
                                if not (c & candidate):
                                    unhit.append(c)
                            if not unhit:
                                transversal = candidate
                                improved = True
                                break
                            potential_w = set()
                            for c in unhit:
                                potential_w.update(c)
                            # Try single replacement
                            for w in potential_w:
                                if w in (v1, v2, v3):
                                    continue
                                ok = True
                                for c in unhit:
                                    if w not in c:
                                        ok = False
                                        break
                                if ok:
                                    transversal = candidate | {w}
                                    improved = True
                                    break
                            if improved:
                                break
                            # Try two replacements
                            for w1 in potential_w:
                                if w1 in (v1, v2, v3):
                                    continue
                                for w2 in potential_w:
                                    if w2 in (v1, v2, v3) or w2 == w1:
                                        continue
                                    ok = True
                                    for c in unhit:
                                        if w1 not in c and w2 not in c:
                                            ok = False
                                            break
                                    if ok:
                                        transversal = candidate | {w1, w2}
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            return transversal

        # Branch and Bound
        def branch_and_bound(initial_transversal, time_budget):
            best_size = len(initial_transversal)
            best_solution = set(initial_transversal)
            
            # Sort cliques by size (smallest first) for better pruning
            clique_order = sorted(range(n_cliques), key=lambda i: len(maximal_cliques[i]))
            
            def lower_bound(unhit_mask, current_size):
                # Simple lower bound: greedy set cover on unhit cliques
                if not unhit_mask:
                    return 0
                remaining = unhit_mask
                count = 0
                while remaining:
                    # Find vertex covering most unhit cliques
                    best_v = -1
                    best_count = 0
                    for v in range(n):
                        count = 0
                        m = remaining
                        while m:
                            lb = m & (-m)
                            idx = lb.bit_length() - 1
                            m ^= lb
                            if v in maximal_cliques[idx]:
                                count += 1
                        if count > best_count:
                            best_count = count
                            best_v = v
                    if best_v == -1 or best_count == 0:
                        break
                    count += 1
                    # Remove cliques covered by best_v
                    for ci in vertex_cliques[best_v]:
                        remaining &= ~(1 << ci)
                return count
            
            def search(unhit_mask, current_set, current_size):
                nonlocal best_size, best_solution
                
                if time.time() > time_budget:
                    return
                
                if current_size >= best_size:
                    return
                
                if not unhit_mask:
                    best_size = current_size
                    best_solution = set(current_set)
                    return
                
                # Lower bound check
                lb = lower_bound(unhit_mask, current_size)
                if current_size + lb >= best_size:
                    return
                
                # Find clique with fewest candidates
                min_cand_count = float('inf')
                best_clique_idx = -1
                best_candidates = []
                
                m = unhit_mask
                while m:
                    lowbit = m & (-m)
                    idx = lowbit.bit_length() - 1
                    m ^= lowbit
                    
                    c = maximal_cliques[idx]
                    cand_count = 0
                    cands = []
                    for v in c:
                        if v not in current_set:
                            cand_count += 1
                            cands.append(v)
                    
                    if cand_count < min_cand_count:
                        min_cand_count = cand_count
                        best_clique_idx = idx
                        best_candidates = cands
                        if min_cand_count <= 1:
                            break
                
                if not best_candidates:
                    return
                
                # Sort candidates by frequency in unhit cliques (descending)
                def candidate_score(v):
                    count = 0
                    mm = unhit_mask
                    while mm:
                        lb = mm & (-mm)
                        idx2 = lb.bit_length() - 1
                        mm ^= lb
                        if v in maximal_cliques[idx2]:
                            count += 1
                    return -count
                
                best_candidates.sort(key=candidate_score)
                
                for v in best_candidates:
                    if time.time() > time_budget:
                        return
                    
                    new_set = current_set | {v}
                    new_unhit = unhit_mask
                    for ci in vertex_cliques[v]:
                        new_unhit &= ~(1 << ci)
                    
                    search(new_unhit, new_set, current_size + 1)
            
            search((1 << n_cliques) - 1, set(), 0)
            
            return best_solution

        # Main algorithm
        transversal = greedy_init()
        transversal = greedy_removal(transversal)
        
        # Branch and bound with time budget
        bb_budget = min(time_budget - (time.time() - start_time), 20)
        if bb_budget > 0:
            bb_solution = branch_and_bound(transversal, start_time + bb_budget)
            if len(bb_solution) < len(transversal):
                transversal = bb_solution
            transversal = greedy_removal(transversal)
        
        # Local search with remaining time
        ls_budget = time_budget - (time.time() - start_time)
        if ls_budget > 0:
            transversal = local_search(transversal)
            transversal = greedy_removal(transversal)
        
        return sorted(transversal)

    results = []
    total_time_budget = 140 - (time.time() - start_time)
    # Allocate time: give more to harder instances
    time_alloc = [total_time_budget * 0.25, total_time_budget * 0.3, total_time_budget * 0.45]
    
    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        if time.time() > deadline:
            # Emergency: return empty lists for remaining
            while len(results) < 3:
                results.append([])
            break
        edges = instance(n, p, seed)
        sol = find_min_transversal(n, edges, start_time + time_alloc[idx])
        results.append(sol)

    return results
