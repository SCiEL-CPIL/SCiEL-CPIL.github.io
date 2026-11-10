import time
from random import Random
from itertools import combinations

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
            pivot = None
            max_inter = -1
            for v in P_union_X:
                inter = len(P & adj[v])
                if inter > max_inter:
                    max_inter = inter
                    pivot = v
            
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
        
        # Precompute bitmask for each vertex
        vertex_clique_mask = [0] * n
        for i, c in enumerate(maximal_cliques):
            for v in c:
                vertex_clique_mask[v] |= (1 << i)
        
        full_mask = (1 << n_cliques) - 1
        
        def is_valid(candidate_set):
            for c in maximal_cliques:
                if not (c & candidate_set):
                    return False
            return True
        
        def local_search(transversal):
            """Apply greedy removal and k-for-1 replacements until no improvement."""
            improved = True
            max_passes = 50
            pass_count = 0
            
            while improved and pass_count < max_passes:
                improved = False
                pass_count += 1
                
                # 1. Greedy Removal
                for v in list(transversal):
                    candidate = transversal - {v}
                    valid = True
                    for c in maximal_cliques:
                        if v in c and not (c & candidate):
                            valid = False
                            break
                    if valid:
                        transversal = candidate
                        improved = True
                
                if not transversal:
                    break

                # 2. k-for-1 Replacement (k=2, k=3)
                trans_list = list(transversal)
                size = len(trans_list)
                
                # Try k=2
                if size >= 2:
                    for i in range(size):
                        for j in range(i + 1, size):
                            v1, v2 = trans_list[i], trans_list[j]
                            base = transversal - {v1, v2}
                            
                            # Find cliques unhit by base
                            unhit = []
                            for c in maximal_cliques:
                                if not (c & base):
                                    unhit.append(c)
                            
                            if not unhit:
                                transversal = base
                                improved = True
                                break
                            
                            # Find w that hits all unhit cliques
                            potential_w = set(unhit[0])
                            for c in unhit[1:]:
                                potential_w &= c
                            potential_w.discard(v1)
                            potential_w.discard(v2)
                            
                            if potential_w:
                                w = min(potential_w)  # deterministic tie-break
                                transversal = base | {w}
                                improved = True
                                break
                        if improved:
                            break
                
                # Try k=3
                if not improved and size >= 3:
                    for combo in combinations(range(size), 3):
                        v1, v2, v3 = trans_list[combo[0]], trans_list[combo[1]], trans_list[combo[2]]
                        base = transversal - {v1, v2, v3}
                        
                        unhit = []
                        for c in maximal_cliques:
                            if not (c & base):
                                unhit.append(c)
                        
                        if not unhit:
                            transversal = base
                            improved = True
                            break
                        
                        potential_w = set(unhit[0])
                        for c in unhit[1:]:
                            potential_w &= c
                        potential_w.discard(v1)
                        potential_w.discard(v2)
                        potential_w.discard(v3)
                        
                        if potential_w:
                            w = min(potential_w)
                            transversal = base | {w}
                            improved = True
                            break
                    if improved:
                        break
            
            return transversal
        
        # Multiple randomized greedy initializations
        best_solution = None
        best_size = float('inf')
        
        # Run several diverse greedy initializations
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > time_budget:
                break
            
            rng = Random(1000 + restart)
            unhit_mask = full_mask
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
                
                # Score vertices: frequency primary, add random perturbation for diversity
                candidates = [(v, freq[v]) for v in range(n) if freq[v] > 0]
                
                # Add small random perturbation to break ties differently each restart
                perturbed = []
                for v, f in candidates:
                    perturbation = rng.random() * 0.1
                    perturbed.append((v, f + perturbation))
                
                perturbed.sort(key=lambda x: -x[1])
                
                # Pick from top 3 with some randomness
                top_k = min(3, len(perturbed))
                if top_k == 1:
                    best_v = perturbed[0][0]
                else:
                    # Weighted random selection from top k
                    weights = [perturbed[i][1] for i in range(top_k)]
                    total = sum(weights)
                    r = rng.random() * total
                    cumulative = 0
                    best_v = perturbed[0][0]
                    for i in range(top_k):
                        cumulative += weights[i]
                        if r <= cumulative:
                            best_v = perturbed[i][0]
                            break
                
                transversal.add(best_v)
                
                # Remove all cliques containing best_v
                mask_v = vertex_clique_mask[best_v]
                unhit_mask &= ~mask_v
            
            # Ensure validity
            if unhit_mask:
                m = unhit_mask
                while m:
                    lowbit = m & (-m)
                    idx = lowbit.bit_length() - 1
                    m ^= lowbit
                    transversal.add(next(iter(maximal_cliques[idx])))
            
            # Apply local search
            transversal = local_search(transversal)
            
            if len(transversal) < best_size:
                best_size = len(transversal)
                best_solution = set(transversal)
            
            if best_size == 0:
                break
        
        if best_solution is None:
            return []
        
        # B&B polish on the best solution if time remains
        remaining_time = time_budget - time.time()
        if remaining_time > 5 and best_size > 1:
            # Simple B&B to try to improve
            def bb_search(unhit_mask, current_set, current_size, budget):
                nonlocal best_size, best_solution
                
                if current_size >= best_size:
                    return
                
                if not unhit_mask:
                    if current_size < best_size:
                        best_size = current_size
                        best_solution = set(current_set)
                    return
                
                if time.time() > budget:
                    return
                
                # Lower bound: at least 1 more needed
                if current_size + 1 >= best_size:
                    return
                
                # Choose clique with fewest candidates
                unhit_list = []
                m = unhit_mask
                while m:
                    lowbit = m & (-m)
                    idx = lowbit.bit_length() - 1
                    m ^= lowbit
                    unhit_list.append(idx)
                
                if not unhit_list:
                    return
                
                # Pick clique with minimum size
                best_ci = min(unhit_list, key=lambda ci: len(maximal_cliques[ci]))
                candidates = list(maximal_cliques[best_ci])
                
                # Sort by frequency in unhit cliques
                def score(v):
                    mask = vertex_clique_mask[v]
                    return -bin(mask & unhit_mask).count('1')
                
                candidates.sort(key=score)
                
                for v in candidates:
                    if time.time() > budget:
                        return
                    
                    new_set = current_set | {v}
                    new_unhit = unhit_mask & ~vertex_clique_mask[v]
                    
                    bb_search(new_unhit, new_set, current_size + 1, budget)
            
            bb_budget = time.time() + min(remaining_time - 2, 20)
            bb_search(full_mask, set(), 0, bb_budget)
        
        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        # Allocate time proportionally to problem size
        remaining = deadline - time.time()
        if remaining <= 0:
            results.append([])
            continue
        
        # Give more time to larger instances
        if n == 30:
            budget = time.time() + min(remaining * 0.3, 30)
        elif n == 35:
            budget = time.time() + min(remaining * 0.35, 40)
        else:
            budget = time.time() + min(remaining * 0.35, 50)
        
        sol = find_min_transversal(n, edges, budget)
        results.append(sol)
    
    return results
