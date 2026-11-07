import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def get_score(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def fast_get_score(values, edge_indices):
        sums = set()
        prods = set()
        for u, v in edge_indices:
            su = values[u]
            sv = values[v]
            sums.add(su + sv)
            prods.add(su * sv)
        return max(len(sums), len(prods)), sums, prods

    def backtrack_edge_search(values, target_k, max_backtracks=5000):
        """
        Attempt to construct a 35-edge subgraph such that 
        distinct sums <= K and distinct products <= K.
        Uses backtracking with heuristic edge selection.
        """
        if time.time() > deadline:
            return None, float('inf')

        current_edges = []
        current_sums = set()
        current_prods = set()
        
        # Precompute all possible edges sorted by "collision potential"
        # We want edges that reuse existing sums/products
        # But initially sets are empty, so we start with random or small values
        
        # Strategy:
        # 1. Pick a random edge.
        # 2. For the next edge, prefer one that shares a sum or product with existing edges.
        # 3. If no such edge exists or if we exceed K, backtrack.
        
        # To make this efficient, we maintain a list of candidate edges.
        
        def search(depth, edges, sums, prods):
            if time.time() > deadline:
                return None
            
            if depth == m:
                if len(sums) <= target_k and len(prods) <= target_k:
                    return list(edges)
                return None
            
            # Get all unused edges
            used_set = set(edges)
            candidates = [e for e in all_edges if e not in used_set]
            
            if not candidates:
                return None
            
            # Heuristic: Score candidates
            # Score = - (1 if sum in sums else 0) - (1 if prod in prods else 0)
            # Lower is better (more collisions)
            
            scored_candidates = []
            for e in candidates:
                u, v = e
                s = values[u] + values[v]
                p = values[u] * values[v]
                
                score = 0
                if s in sums:
                    score -= 2
                if p in prods:
                    score -= 2
                
                # Also penalize if it adds a NEW sum/prod and we are close to limit
                if len(sums) >= target_k and s not in sums:
                    score += 5
                if len(prods) >= target_k and p not in prods:
                    score += 5
                    
                scored_candidates.append((score, random.random(), e))
            
            # Sort by score (best first), then random tie-breaker
            scored_candidates.sort(key=lambda x: (x[0], x[1]))
            
            # Try top N candidates
            # If we are deep, try fewer to save time? Or more?
            # Let's try top 10
            
            for _, _, e in scored_candidates[:10]:
                u, v = e
                s = values[u] + values[v]
                p = values[u] * values[v]
                
                # Check feasibility
                new_sums_count = len(sums) + (0 if s in sums else 1)
                new_prods_count = len(prods) + (0 if p in prods else 1)
                
                if new_sums_count > target_k or new_prods_count > target_k:
                    continue
                
                new_edges = edges + [e]
                new_sums = sums.copy()
                new_prods = prods.copy()
                new_sums.add(s)
                new_prods.add(p)
                
                result = search(depth + 1, new_edges, new_sums, new_prods)
                if result is not None:
                    return result
            
            return None
        
        # Start search
        # To speed up, we can seed with a few random edges first
        # Or just start from scratch.
        
        # Optimization: If target_k is very small, this might fail quickly.
        # We will call this for increasing K.
        
        return search(0, [], set(), set())

    def polish_edges(values, edges, max_iters=5000):
        """Local search on edges only, keeping values fixed."""
        current_edges = set(edges)
        current_edge_list = list(current_edges)
        current_score, _, _ = fast_get_score(values, current_edge_list)

        best_score = current_score
        best_edges = set(current_edges)

        for i in range(max_iters):
            if time.time() > deadline:
                break

            e_out_idx = random.randint(0, m - 1)
            e_out = current_edge_list[e_out_idx]

            while True:
                e_in = random.choice(all_edges)
                if e_in not in current_edges:
                    break

            new_edge_list = list(current_edge_list)
            new_edge_list[e_out_idx] = e_in

            new_score, _, _ = fast_get_score(values, new_edge_list)

            if new_score <= current_score:
                current_edges = set(new_edge_list)
                current_edge_list = new_edge_list
                current_score = new_score

                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)

        return best_score, best_edges

    def polish_values(values, edges, max_iters=5000):
        """Local search on values only, keeping edges fixed."""
        current_values = list(values)
        edge_list = list(edges)
        current_score, _, _ = fast_get_score(current_values, edge_list)

        best_score = current_score
        best_values = list(current_values)

        for i in range(max_iters):
            if time.time() > deadline:
                break

            idx = random.randint(0, n - 1)
            # Try swapping
            if random.random() < 0.6:
                idx2 = random.randint(0, n - 1)
                if idx2 == idx:
                    continue
                new_values = list(current_values)
                new_values[idx], new_values[idx2] = new_values[idx2], new_values[idx]
            else:
                delta = random.randint(-20, 20)
                new_val = current_values[idx] + delta
                if not (1 <= new_val <= 10000) or new_val in current_values:
                    continue
                new_values = list(current_values)
                new_values[idx] = new_val

            new_score, _, _ = fast_get_score(new_values, edge_list)

            if new_score <= current_score:
                current_values = new_values
                current_score = new_score

                if current_score < best_score:
                    best_score = current_score
                    best_values = new_values

        return best_score, best_values

    def joint_sa(initial_values, initial_edges, max_iter=20000, T_start=2.0, T_end=0.01):
        current_values = list(initial_values)
        current_edges = set(initial_edges)
        current_edge_list = list(current_edges)

        current_score, current_sums, current_prods = fast_get_score(current_values, current_edge_list)

        best_score = current_score
        best_values = list(current_values)
        best_edges = set(current_edges)

        adj = [set() for _ in range(n)]
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)

        for i in range(max_iter):
            if time.time() > deadline:
                break

            T = T_start * (T_end / T_start) ** (i / max_iter)
            move_type = random.random()

            if move_type < 0.45:
                # Edge Move: Swap one edge for another
                e_out_idx = random.randint(0, m - 1)
                e_out = current_edge_list[e_out_idx]

                while True:
                    e_in = random.choice(all_edges)
                    if e_in not in current_edges:
                        break

                new_edge_list = list(current_edge_list)
                new_edge_list[e_out_idx] = e_in
                new_edges = set(new_edge_list)

                new_score, new_sums, new_prods = fast_get_score(current_values, new_edge_list)

                delta_score = new_score - current_score
                if delta_score <= 0 or random.random() < math.exp(-delta_score / max(T, 1e-9)):
                    current_edges = new_edges
                    current_edge_list = new_edge_list
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods
                    adj[e_out[0]].discard(e_out[1])
                    adj[e_out[1]].discard(e_out[0])
                    adj[e_in[0]].add(e_in[1])
                    adj[e_in[1]].add(e_in[0])

                    if current_score < best_score:
                        best_score = current_score
                        best_values = list(current_values)
                        best_edges = set(current_edges)

            elif move_type < 0.85:
                # Value Move: Swap two values
                idx = random.randint(0, n - 1)
                idx2 = random.randint(0, n - 1)
                if idx == idx2:
                    continue

                new_values = list(current_values)
                new_values[idx], new_values[idx2] = new_values[idx2], new_values[idx]

                new_score, new_sums, new_prods = fast_get_score(new_values, current_edge_list)

                delta_score = new_score - current_score
                if delta_score <= 0 or random.random() < math.exp(-delta_score / max(T, 1e-9)):
                    current_values = new_values
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods

                    if current_score < best_score:
                        best_score = current_score
                        best_values = list(current_values)
                        best_edges = set(current_edges)

            else:
                # Value Move: Targeted collision-seeking
                idx = random.randint(0, n - 1)
                neighbors = list(adj[idx])
                if not neighbors:
                    continue

                found = False
                for _ in range(3):
                    nb = random.choice(neighbors)
                    v_nb = current_values[nb]

                    candidates = []
                    for s in current_sums:
                        cand = s - v_nb
                        if 1 <= cand <= 10000 and cand != current_values[idx]:
                            if cand not in current_values:
                                candidates.append(cand)
                    for p in current_prods:
                        if p % v_nb == 0:
                            cand = p // v_nb
                            if 1 <= cand <= 10000 and cand != current_values[idx]:
                                if cand not in current_values:
                                    candidates.append(cand)

                    if candidates:
                        new_val = random.choice(candidates[:10])
                        new_values = list(current_values)
                        new_values[idx] = new_val

                        new_score, new_sums, new_prods = fast_get_score(new_values, current_edge_list)

                        delta_score = new_score - current_score
                        if delta_score <= 0 or random.random() < math.exp(-delta_score / max(T, 1e-9)):
                            current_values = new_values
                            current_score = new_score
                            current_sums = new_sums
                            current_prods = new_prods

                            if current_score < best_score:
                                best_score = current_score
                                best_values = list(current_values)
                                best_edges = set(current_edges)
                            found = True
                            break

                if not found:
                    delta = random.randint(-10, 10)
                    new_val = current_values[idx] + delta
                    if 1 <= new_val <= 10000 and new_val not in current_values:
                        new_values = list(current_values)
                        new_values[idx] = new_val
                        new_score, new_sums, new_prods = fast_get_score(new_values, current_edge_list)

                        delta_score = new_score - current_score
                        if delta_score <= 0 or random.random() < math.exp(-delta_score / max(T, 1e-9)):
                            current_values = new_values
                            current_score = new_score
                            current_sums = new_sums
                            current_prods = new_prods

                            if current_score < best_score:
                                best_score = current_score
                                best_values = list(current_values)
                                best_edges = set(current_edges)

        return best_score, best_values, best_edges

    best_global_score = float('inf')
    best_global_values = None
    best_global_edges = None

    random.seed(42)

    # Generate diverse initial configurations
    initial_configs = []

    # Config 1: Small range
    v1 = random.sample(range(1, 51), n)
    e1 = set(random.sample(all_edges, m))
    initial_configs.append((v1, e1))

    # Config 2: Medium range
    v2 = random.sample(range(1, 101), n)
    e2 = set(random.sample(all_edges, m))
    initial_configs.append((v2, e2))

    # Config 3: Arithmetic progression
    v3 = [i * 10 + 1 for i in range(n)]
    e3 = set(random.sample(all_edges, m))
    initial_configs.append((v3, e3))

    # Config 4: Powers of 2
    v4 = [2**i for i in range(n)]
    if max(v4) <= 10000:
        e4 = set(random.sample(all_edges, m))
        initial_configs.append((v4, e4))

    # Config 5: Random large
    v5 = random.sample(range(1, 10001), n)
    e5 = set(random.sample(all_edges, m))
    initial_configs.append((v5, e5))

    # Config 6: Near-duplicates
    v6 = list(range(1, 8)) + [10, 12, 15, 20, 25, 30, 40]
    random.shuffle(v6)
    e6 = set(random.sample(all_edges, m))
    initial_configs.append((v6, e6))

    # Config 7: Multiplicative structure
    v7 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    random.shuffle(v7)
    e7 = set(random.sample(all_edges, m))
    initial_configs.append((v7, e7))

    # Run SA for each config
    time_per_config = (deadline - time.time()) / len(initial_configs)

    for vals, edges in initial_configs:
        if time.time() > deadline:
            break

        max_iter = int(time_per_config * 800)
        max_iter = min(max_iter, 30000)
        max_iter = max(max_iter, 2000)

        score, best_v, best_e = joint_sa(vals, edges, max_iter=max_iter, T_start=2.0, T_end=0.01)

        # Polish pass
        if time.time() < deadline - 5.0:
            polish_iters = int((deadline - time.time()) * 0.3 * 500)
            polish_iters = max(polish_iters, 1000)
            p_score, p_edges = polish_edges(best_v, best_e, max_iters=polish_iters)
            if p_score < score:
                score = p_score
                best_e = p_edges

            if time.time() < deadline - 2.0:
                pv_iters = int((deadline - time.time()) * 0.2 * 500)
                pv_iters = max(pv_iters, 500)
                pv_score, p_values = polish_values(best_v, best_e, max_iters=pv_iters)
                if pv_score < score:
                    score = pv_score
                    best_v = p_values

        if score < best_global_score:
            best_global_score = score
            best_global_values = best_v
            best_global_edges = best_e

    # NEW STRATEGY: Backtracking Search for specific K
    # Try to find a solution with K distinct sums/products
    # Start from a reasonable K (e.g., 20) and go down? Or up from 10?
    # Let's try a few K values
    
    if best_global_values is None:
        best_global_values = list(range(1, 15))
        best_global_edges = set(all_edges[:35])
        best_global_score = get_score(best_global_values, best_global_edges)

    # Use the best values found so far to try backtracking
    # Or try backtracking with specific value sets designed for collisions
    
    # Value sets that encourage collisions:
    # 1. All 1s? No, distinct.
    # 2. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
    # 3. Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (too big? 8192<10000 ok)
    # 4. Multiples of a base: 10, 20, 30...
    
    candidate_value_sets = [
        best_global_values,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2**i for i in range(14)],
        [10*i + 1 for i in range(14)],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    ]
    
    for vals in candidate_value_sets:
        if time.time() > deadline - 10:
            break
            
        # Try K = 15, 16, 17, 18
        for k in range(15, 20):
            if time.time() > deadline - 5:
                break
                
            # Backtracking can be slow. Limit it.
            # We'll just run it once for the best K we can afford.
            # Actually, backtracking with randomization and limits is better.
            
            # Let's modify backtrack_edge_search to be faster and limited
            # It already has time checks.
            
            result_edges = backtrack_edge_search(vals, k)
            
            if result_edges is not None:
                score, _, _ = fast_get_score(vals, result_edges)
                if score < best_global_score:
                    best_global_score = score
                    best_global_values = vals
                    best_global_edges = set(result_edges)
                    
                    # Polish it
                    if time.time() < deadline - 2:
                        p_score, p_edges = polish_edges(best_global_values, best_global_edges, max_iters=2000)
                        if p_score < best_global_score:
                            best_global_score = p_score
                            best_global_edges = p_edges

    # Final polish on best result
    if best_global_values is not None and time.time() < deadline - 1.0:
        remaining = deadline - time.time() - 1.0
        pe_iters = int(remaining * 0.5 * 500)
        pe_iters = max(pe_iters, 500)
        ps, pe = polish_edges(best_global_values, best_global_edges, max_iters=pe_iters)
        if ps < best_global_score:
            best_global_score = ps
            best_global_edges = pe

        if time.time() < deadline - 0.5:
            pv_iters = int((deadline - time.time() - 0.5) * 500)
            pv_iters = max(pv_iters, 200)
            pvs, pv = polish_values(best_global_values, best_global_edges, max_iters=pv_iters)
            if pvs < best_global_score:
                best_global_score = pvs
                best_global_values = pv

    # Fallback
    if best_global_values is None:
        best_global_values = list(range(1, 15))
        best_global_edges = set(all_edges[:35])
        best_global_score = get_score(best_global_values, best_global_edges)

    # Ensure types
    final_values = [int(v) for v in best_global_values]
    final_edges = [[int(u), int(v)] for u, v in best_global_edges]

    return {
        'values': final_values,
        'edges': final_edges
    }
